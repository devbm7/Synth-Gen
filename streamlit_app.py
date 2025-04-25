import streamlit as st
import requests
import json
import pandas as pd
import time
import os
import base64
from datetime import datetime
from io import StringIO, BytesIO
from typing import List, Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API configuration
API_URL = "http://localhost:8000"  # Change if API is hosted elsewhere
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Set page config
st.set_page_config(
    page_title="Synthetic Data Generator",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper functions
def get_download_link(file_path, link_text):
    """Generate a download link for a file."""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        filename = os.path.basename(file_path)
        return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{link_text}</a>'
    except Exception as e:
        logger.error(f"Error creating download link: {e}")
        return None

# Function to call the API
def call_api(endpoint, method="GET", data=None, params=None):
    """Make API call with error handling."""
    url = f"{API_URL}/{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, params=params)
        elif method == "POST":
            response = requests.post(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)
        else:
            st.error(f"Unsupported method: {method}")
            return None
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        logger.error(f"API Error ({url}): {e}")
        return None

# Define data types
DATA_TYPES = [
    "string", "integer", "float", "boolean", "date", "datetime",
    "email", "phone", "address", "name", "country", "state", 
    "city", "zip", "url", "list"
]

# App state
if 'columns' not in st.session_state:
    st.session_state.columns = []

if 'task_id' not in st.session_state:
    st.session_state.task_id = None

if 'data' not in st.session_state:
    st.session_state.data = None

if 'csv_file' not in st.session_state:
    st.session_state.csv_file = None

# App title
st.title("🧪 Synthetic Data Generator")
st.markdown("Generate realistic synthetic data using LLM models")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # LLM Model
    model = st.text_input("LLM Model", "gemma3:latest")
    
    # Number of rows
    num_rows = st.number_input("Number of rows", min_value=1, value=10)
    
    # Batch size
    batch_size = st.number_input("Batch size", min_value=1, value=10)
    
    # Parallel processing
    parallel = st.checkbox("Enable parallel processing")
    
    # Active task
    if st.session_state.task_id:
        st.subheader("Active Task")
        st.write(f"Task ID: {st.session_state.task_id}")
        
        if st.button("Cancel Task"):
            call_api(f"tasks/{st.session_state.task_id}", method="DELETE")
            st.session_state.task_id = None
            st.session_state.data = None
            st.session_state.csv_file = None
            st.rerun()

# Main content
tab1, tab2, tab3 = st.tabs(["Column Definition", "Results", "Recent Tasks"])

# Column Definition Tab
with tab1:
    st.header("Define Data Columns")
    
    # Column definition form
    with st.form("column_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            column_name = st.text_input("Column Name")
        
        with col2:
            column_type = st.selectbox("Data Type", DATA_TYPES)
        
        column_description = st.text_area("Description (optional)", "", help="Describe this column to help the LLM generate appropriate data")
        
        # Advanced constraints based on data type
        with st.expander("Advanced Constraints"):
            constraints = {}
            
            if column_type in ["integer", "float"]:
                min_value = st.number_input("Minimum Value", value=None, step=1)
                max_value = st.number_input("Maximum Value", value=None, step=1)
                if min_value is not None:
                    constraints["ge"] = min_value
                if max_value is not None:
                    constraints["le"] = max_value
            
            elif column_type == "string":
                min_length = st.number_input("Minimum Length", value=None, min_value=0, step=1)
                max_length = st.number_input("Maximum Length", value=None, min_value=0, step=1)
                if min_length is not None:
                    constraints["min_length"] = min_length
                if max_length is not None:
                    constraints["max_length"] = max_length
        
        submitted = st.form_submit_button("Add Column")
        
        if submitted and column_name:
            st.session_state.columns.append({
                "name": column_name,
                "type": column_type,
                "description": column_description if column_description else None,
                "constraints": constraints if constraints else None
            })
    
    # Display current columns
    if st.session_state.columns:
        st.subheader("Current Columns")
        
        col_data = pd.DataFrame(st.session_state.columns)
        st.dataframe(col_data)
        
        if st.button("Clear All Columns"):
            st.session_state.columns = []
            st.rerun()
        
        # Generate data button
        if st.button("Generate Data"):
            if not st.session_state.columns:
                st.error("Please add at least one column")
            else:
                with st.spinner("Submitting generation request..."):
                    # Prepare request data
                    request_data = {
                        "columns": st.session_state.columns,
                        "rows": num_rows,
                        "model": model,
                        "batch_size": batch_size,
                        "parallel": parallel
                    }
                    
                    # Submit to API
                    result = call_api("generate", method="POST", data=request_data)
                    
                    if result and "task_id" in result:
                        st.session_state.task_id = result["task_id"]
                        st.success(f"Generation task submitted! Task ID: {result['task_id']}")
                        st.rerun()

# Results Tab
with tab2:
    st.header("Generated Data")
    
    if st.session_state.task_id:
        # Check task status
        task_status = call_api(f"task/{st.session_state.task_id}")
        
        if task_status:
            st.info(f"Status: {task_status['status']}")
            
            if task_status["status"] == "completed":
                # Fetch data if not already fetched
                if not st.session_state.data:
                    with st.spinner("Fetching generated data..."):
                        st.session_state.data = call_api(f"data/{st.session_state.task_id}")
                
                if st.session_state.data and "data" in st.session_state.data:
                    # Display as dataframe
                    df = pd.DataFrame(st.session_state.data["data"])
                    st.dataframe(df)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # JSON download
                        if st.button("Download JSON"):
                            # Create a JSON string
                            json_string = json.dumps(st.session_state.data, indent=2)
                            
                            # Create a download button
                            st.download_button(
                                label="Download JSON File",
                                data=json_string,
                                file_name=f"synthetic_data_{st.session_state.task_id}.json",
                                mime="application/json"
                            )
                    
                    with col2:
                        # CSV conversion and download
                        if st.button("Convert to CSV"):
                            with st.spinner("Converting to CSV..."):
                                csv_result = call_api(f"convert_to_csv/{st.session_state.task_id}", method="POST")
                                
                                if csv_result and "csv_file" in csv_result:
                                    st.session_state.csv_file = csv_result["csv_file"]
                                    st.success("CSV conversion successful!")
                    
                    # Display CSV download if available
                    if st.session_state.csv_file:
                        try:
                            # Read CSV to DataFrame
                            csv_df = pd.read_csv(st.session_state.csv_file)
                            st.subheader("CSV Preview")
                            st.dataframe(csv_df)
                            
                            # CSV download
                            with open(st.session_state.csv_file, "r") as file:
                                csv_string = file.read()
                            
                            st.download_button(
                                label="Download CSV File",
                                data=csv_string,
                                file_name=f"synthetic_data_{st.session_state.task_id}.csv",
                                mime="text/csv"
                            )
                        except Exception as e:
                            st.error(f"Error reading CSV file: {e}")
            
            elif task_status["status"] == "running" or task_status["status"] == "pending":
                st.warning("Data generation is in progress... This may take some time.")
                # Add auto-refresh for status updates
                st.empty()
                time.sleep(3)
                st.rerun()
            
            elif task_status["status"] == "failed":
                st.error(f"Generation failed: {task_status.get('message', 'Unknown error')}")
    else:
        st.info("No active generation task. Go to the Column Definition tab to start a new task.")

# Recent Tasks Tab
with tab3:
    st.header("Recent Tasks")
    
    if st.button("Refresh Tasks"):
        st.rerun()
    
    # Fetch all tasks
    tasks = call_api("tasks")
    
    if tasks:
        # Create a dataframe from tasks
        task_df = pd.DataFrame([
            {
                "Task ID": t["task_id"],
                "Status": t["status"],
                "Created": t["created_at"],
                "Completed": t["completed_at"] if t["completed_at"] else "",
                "Result File": t["result_file"] if t["result_file"] else ""
            }
            for t in tasks
        ])
        
        if not task_df.empty:
            st.dataframe(task_df)
            
            # Allow loading a previous task
            task_id = st.selectbox("Select a task to load", task_df["Task ID"].tolist())
            
            if st.button("Load Selected Task"):
                st.session_state.task_id = task_id
                st.session_state.data = None
                st.session_state.csv_file = None
                st.rerun()
        else:
            st.info("No tasks available")
    else:
        st.info("No tasks available or unable to fetch task list")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Your Synthetic Data Generator")