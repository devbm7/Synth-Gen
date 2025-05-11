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
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel, Field, create_model, field_validator
from typing import Dict, List, Any, Optional, Type, Union, Annotated
import pandas as pd
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
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="Synthetic Data Generator API",
    description="API for generating synthetic data using LLMs",
    version="1.0.1"
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

class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    column_info: List[dict]
    row_count: int

class AppendDataRequest(BaseModel):
    file_id: str
    rows: int = Field(..., description="Number of rows to generate", gt=0)
    model: str = Field("gemma3:latest", description="LLM model to use")
    batch_size: int = Field(10, description="Batch size for generation", gt=0)
    parallel: bool = Field(False, description="Enable parallel processing")



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
            format=list_model.model_json_schema(),
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

# Add new functions to extract column information from CSV
def extract_column_info(filepath: str) -> List[dict]:
    """Extract column information from a CSV file."""
    try:
        # Read first few rows to infer data types
        df = pd.read_csv(filepath, nrows=100)
        
        columns = []
        for col_name in df.columns:
            col_data = df[col_name]
            col_type = "string"  # Default type
            
            # Infer data type
            if pd.api.types.is_integer_dtype(col_data):
                col_type = "integer"
            elif pd.api.types.is_float_dtype(col_data):
                col_type = "float"
            elif pd.api.types.is_bool_dtype(col_data):
                col_type = "boolean"
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                col_type = "datetime"
            elif col_name.lower() in ["email", "mail"]:
                col_type = "email"
            elif col_name.lower() in ["phone", "telephone", "mobile"]:
                col_type = "phone"
            elif col_name.lower() in ["address", "addr"]:
                col_type = "address"
            elif col_name.lower() in ["name", "fullname", "full_name"]:
                col_type = "name"
            elif col_name.lower() in ["country"]:
                col_type = "country"
            elif col_name.lower() in ["state", "province"]:
                col_type = "state"
            elif col_name.lower() in ["city"]:
                col_type = "city"
            elif col_name.lower() in ["zip", "zipcode", "postal", "postalcode"]:
                col_type = "zip"
            elif col_name.lower() in ["url", "website", "link"]:
                col_type = "url"
            
            # Create constraints based on data
            constraints = {}
            if col_type in ["integer", "float"]:
                constraints["ge"] = float(col_data.min()) if not pd.isna(col_data.min()) else None
                constraints["le"] = float(col_data.max()) if not pd.isna(col_data.max()) else None
            elif col_type == "string":
                # Get average string length, rounded up
                avg_len = col_data.str.len().mean() if not col_data.empty else 20
                constraints["max_length"] = int(avg_len * 1.5) if not pd.isna(avg_len) else 100
            
            # Remove None values from constraints
            constraints = {k: v for k, v in constraints.items() if v is not None}
            
            columns.append({
                "name": col_name,
                "type": col_type,
                "description": f"Generate data similar to the existing {col_name} column",
                "constraints": constraints if constraints else None
            })
            
        return columns
    except Exception as e:
        logger.error(f"Error extracting column info: {e}")
        raise e

# Add to app.py - New endpoint to upload a CSV file
@app.post("/upload_csv", response_model=FileUploadResponse)
async def upload_csv(file: UploadFile = File(...)):
    """Upload a CSV file and extract column information."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        file_path = f"{UPLOAD_DIR}/{file_id}.csv"
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Extract column information
        column_info = extract_column_info(file_path)
        
        # Count rows in the CSV
        with open(file_path, "r", encoding="utf-8") as f:
            row_count = sum(1 for _ in csv.reader(f)) - 1  # Subtract 1 for header
        
        return FileUploadResponse(
            file_id=file_id,
            filename=file.filename,
            column_info=column_info,
            row_count=row_count
        )
    except Exception as e:
        logger.error(f"Error uploading CSV: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add new function to generate a prompt that maintains consistency with existing data
def generate_append_prompt(columns: List[Dict], rows: int, sample_data: List[Dict]) -> str:
    """Generate a prompt for the LLM based on column definitions and sample data."""
    prompt = f"Generate {rows} rows of synthetic data that are similar to the sample data provided. "
    prompt += "The data should follow this structure:\n\n"
    
    # Add column information
    for column in columns:
        prompt += f"- {column['name']} ({column['type']}): "
        if column.get('description'):
            prompt += f"{column['description']} "
        
        if column.get('constraints'):
            constraints_text = ", ".join([f"{k}={v}" for k, v in column['constraints'].items()])
            prompt += f"with constraints: {constraints_text} "
        
        prompt += "\n"
    
    # Add sample data for reference
    prompt += "\nHere are some sample rows from the existing data for reference:\n"
    for i, sample in enumerate(sample_data[:5]):  # Use up to 5 samples
        prompt += f"\nSample {i+1}: {json.dumps(sample)}"
    
    prompt += "\n\nRespond with a JSON array of objects, where each object has all the fields listed above. "
    prompt += "Make sure all data is realistic, diverse, and follows the pattern of the sample data. "
    prompt += "Do not include extra fields."
    
    return prompt

# Add new background task for appending data
async def append_data_task(task_id: str, file_id: str, rows: int, model: str, batch_size: int, parallel: bool):
    """Background task to generate and append data to an existing CSV."""
    try:
        logger.info(f"Starting append task {task_id} to generate {rows} rows")
        tasks[task_id].status = "running"
        
        # Get the original file path
        original_file = f"{UPLOAD_DIR}/{file_id}.csv"
        if not os.path.exists(original_file):
            raise FileNotFoundError(f"Original CSV file not found: {original_file}")
        
        # Read the original CSV to get column info and sample data
        df = pd.read_csv(original_file)
        sample_data = df.head(5).to_dict('records')
        
        # Extract column information
        columns = extract_column_info(original_file)
        
        # Create dynamic model for validation based on columns
        data_model = construct_dynamic_model([DataColumn(**col) for col in columns])
        
        # Fix: Properly annotate the data field when creating the list model
        list_model = create_model(
            "DataList", 
            data=(List[data_model], Field(..., description="List of data items"))
        )
        
        # Generate prompt with sample data
        prompt = generate_append_prompt(columns, rows, sample_data)
        logger.debug(f"Generated prompt for append task: {prompt}")
        
        # Call LLM
        response = chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            format=list_model.model_json_schema(),
        )
        
        logger.info(f"Received response from LLM for append task {task_id}")
        
        # Validate response
        try:
            data = list_model.model_validate_json(response.message.content).data
            
            # First save generated data to a temporary JSON file for reference
            temp_result_file = f"{DATA_DIR}/{task_id}_temp.json"
            with open(temp_result_file, "w", encoding="utf-8") as f:
                json.dump({"data": [item.model_dump() for item in data]}, f, default=str, indent=2)
            
            # Convert to DataFrame and append to original CSV
            new_data_df = pd.DataFrame([item.model_dump() for item in data])
            
            # Ensure the output has the same columns as the original
            new_data_df = new_data_df[df.columns]
            
            # Append to the original CSV
            new_data_df.to_csv(original_file, mode='a', header=False, index=False)
            
            # Also save a copy of the appended result
            result_file = f"{DATA_DIR}/{task_id}_appended.csv"
            df_combined = pd.concat([df, new_data_df])
            df_combined.to_csv(result_file, index=False)
            
            tasks[task_id].status = "completed"
            tasks[task_id].completed_at = datetime.now()
            tasks[task_id].result_file = result_file
            
            logger.info(f"Append task {task_id} completed successfully, appended to {original_file} and saved full result to {result_file}")
            
        except Exception as e:
            logger.error(f"Validation error in append task {task_id}: {e}")
            tasks[task_id].status = "failed"
            tasks[task_id].message = f"Validation error: {str(e)}"
            tasks[task_id].completed_at = datetime.now()
            
    except Exception as e:
        logger.error(f"Error in append task {task_id}: {e}")
        tasks[task_id].status = "failed"
        tasks[task_id].message = str(e)
        tasks[task_id].completed_at = datetime.now()

# Add new endpoint for appending data
@app.post("/append_data", response_model=TaskStatus)
async def append_data(request: AppendDataRequest, background_tasks: BackgroundTasks):
    """Endpoint to generate and append synthetic data to an existing CSV."""
    file_path = f"{UPLOAD_DIR}/{request.file_id}.csv"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    task_id = str(uuid.uuid4())
    
    # Create task record
    tasks[task_id] = TaskStatus(
        task_id=task_id,
        status="pending",
        created_at=datetime.now()
    )
    
    # Start background task
    background_tasks.add_task(
        append_data_task, 
        task_id, 
        request.file_id, 
        request.rows, 
        request.model, 
        request.batch_size, 
        request.parallel
    )
    
    return tasks[task_id]

# Add endpoint to download the original file with appended data
@app.get("/download_appended/{file_id}")
async def download_appended(file_id: str):
    """Get the original file with appended data."""
    file_path = f"{UPLOAD_DIR}/{file_id}.csv"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        with open(file_path, "r") as f:
            content = f.read()
        return {"content": content, "filename": f"appended_{file_id}.csv"}
    except Exception as e:
        logger.error(f"Error reading appended file: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)