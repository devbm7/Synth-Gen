from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, create_model, field_validator
from typing import Dict, List, Any, Optional, Type, Union, Annotated
import uvicorn
import time
from datetime import datetime
import logging
import json
import os
import uuid
from enum import Enum
from ollama import chat
import asyncio
from pathlib import Path
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/api.log")
    ]
)

logger = logging.getLogger("data-gen-api")

# Constants
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(
    title="Synthetic Data Generator API",
    description="API for generating synthetic data using LLMs",
    version="1.0.0"
)

# Data Types Enum
class DataType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    NAME = "name"
    COUNTRY = "country"
    STATE = "state"
    CITY = "city"
    ZIP = "zip"
    URL = "url"
    LIST = "list"


class DataColumn(BaseModel):
    name: str = Field(..., description="Column name")
    type: DataType = Field(..., description="Data type")
    description: Optional[str] = Field(None, description="Description of the data")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Additional constraints for validation")


class DataRequest(BaseModel):
    columns: List[DataColumn] = Field(..., description="Column definitions")
    rows: int = Field(..., description="Number of rows to generate", gt=0)
    model: str = Field("gemma3:latest", description="LLM model to use")
    batch_size: int = Field(10, description="Batch size for generation", gt=0)
    parallel: bool = Field(False, description="Enable parallel processing")


class TaskStatus(BaseModel):
    task_id: str
    status: str
    message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    result_file: Optional[str] = None


# Store for background tasks
tasks = {}


def construct_dynamic_model(columns: List[DataColumn]) -> Type[BaseModel]:
    """Dynamically create a Pydantic model based on column definitions."""
    field_definitions = {}
    for column in columns:
        field_type = str
        
        if column.type == DataType.INTEGER:
            field_type = int
        elif column.type == DataType.FLOAT:
            field_type = float
        elif column.type == DataType.BOOLEAN:
            field_type = bool
        elif column.type == DataType.DATE or column.type == DataType.DATETIME:
            field_type = datetime
        elif column.type == DataType.LIST:
            field_type = List[str]
        
        # Add constraints
        field_kwargs = {}
        if column.constraints:
            field_kwargs.update(column.constraints)
        
        field_definitions[column.name] = (field_type, Field(**field_kwargs))
    
    # Create the dynamic model
    return create_model("DynamicDataModel", **field_definitions)


def generate_prompt(columns: List[DataColumn], rows: int) -> str:
    """Generate a prompt for the LLM based on column definitions."""
    prompt = f"Generate {rows} rows of synthetic data in JSON format. "
    prompt += "The data should follow this structure:\n\n"
    
    for column in columns:
        prompt += f"- {column.name} ({column.type}): "
        if column.description:
            prompt += f"{column.description} "
        
        if column.constraints:
            constraints_text = ", ".join([f"{k}={v}" for k, v in column.constraints.items()])
            prompt += f"with constraints: {constraints_text} "
        
        prompt += "\n"
    
    prompt += "\nRespond with a JSON array of objects, where each object has all the fields listed above. "
    prompt += "Make sure all data is realistic and diverse. Do not include extra fields."
    
    return prompt


async def generate_data_task(task_id: str, request: DataRequest):
    """Background task to generate data using LLM."""
    try:
        logger.info(f"Starting task {task_id} to generate {request.rows} rows")
        tasks[task_id].status = "running"
        
        # Create dynamic model for validation
        data_model = construct_dynamic_model(request.columns)
        
        # Fix: Properly annotate the data field when creating the list model
        list_model = create_model(
            "DataList", 
            data=(List[data_model], Field(..., description="List of data items"))
        )
        
        # Generate prompt
        prompt = generate_prompt(request.columns, request.rows)
        logger.debug(f"Generated prompt: {prompt}")
        
        # Call LLM
        response = chat(
            messages=[{"role": "user", "content": prompt}],
            model=request.model,
            format=list_model.model_json_schema()
        )
        
        logger.info(f"Received response from LLM for task {task_id}")
        
        # Validate response
        try:
            data = list_model.model_validate_json(response.message.content).data
            
            # Save result to file
            result_file = f"{DATA_DIR}/{task_id}.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump({"data": [item.model_dump() for item in data]}, f, default=str, indent=2)
            
            tasks[task_id].status = "completed"
            tasks[task_id].completed_at = datetime.now()
            tasks[task_id].result_file = result_file
            
            logger.info(f"Task {task_id} completed successfully, saved to {result_file}")
            
        except Exception as e:
            logger.error(f"Validation error in task {task_id}: {e}")
            tasks[task_id].status = "failed"
            tasks[task_id].message = f"Validation error: {str(e)}"
            tasks[task_id].completed_at = datetime.now()
            
    except Exception as e:
        logger.error(f"Error in task {task_id}: {e}")
        tasks[task_id].status = "failed"
        tasks[task_id].message = str(e)
        tasks[task_id].completed_at = datetime.now()


@app.post("/generate", response_model=TaskStatus)
async def generate_data(request: DataRequest, background_tasks: BackgroundTasks):
    """Endpoint to generate synthetic data using LLM."""
    task_id = str(uuid.uuid4())
    
    # Create task record
    tasks[task_id] = TaskStatus(
        task_id=task_id,
        status="pending",
        created_at=datetime.now()
    )
    
    # Start background task
    background_tasks.add_task(generate_data_task, task_id, request)
    
    return tasks[task_id]


@app.get("/task/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get status of a task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return tasks[task_id]


@app.get("/tasks", response_model=List[TaskStatus])
async def list_tasks():
    """List all tasks."""
    return list(tasks.values())


@app.get("/data/{task_id}")
async def get_task_data(task_id: str):
    """Get data generated by a task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    if task.status != "completed":
        raise HTTPException(status_code=400, detail=f"Task is not completed (status: {task.status})")
    
    if not task.result_file or not os.path.exists(task.result_file):
        raise HTTPException(status_code=404, detail="Result file not found")
    
    try:
        with open(task.result_file, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading result file for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading result file: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now()}


@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task and its data."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    if task.result_file and os.path.exists(task.result_file):
        try:
            os.remove(task.result_file)
        except Exception as e:
            logger.error(f"Error deleting result file for task {task_id}: {e}")
    
    del tasks[task_id]
    return {"status": "deleted", "task_id": task_id}


@app.post("/convert_to_csv/{task_id}")
async def convert_to_csv(task_id: str):
    """Convert task result to CSV."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    if task.status != "completed":
        raise HTTPException(status_code=400, detail=f"Task is not completed (status: {task.status})")
    
    if not task.result_file or not os.path.exists(task.result_file):
        raise HTTPException(status_code=404, detail="Result file not found")
    
    try:
        # Read JSON data
        with open(task.result_file, "r") as f:
            json_data = json.load(f)
        
        data_entries = json_data["data"]
        
        # Define CSV file path
        csv_file_path = f"{DATA_DIR}/{task_id}.csv"
        
        # Write data to CSV file
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
            if data_entries:
                writer = csv.DictWriter(csv_file, fieldnames=data_entries[0].keys())
                writer.writeheader()
                writer.writerows(data_entries)
        
        logger.info(f"Data successfully written to {csv_file_path}")
        return {"status": "success", "csv_file": csv_file_path}
    
    except Exception as e:
        logger.error(f"Error converting to CSV for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error converting to CSV: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)