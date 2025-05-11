# Synthetic Data Generator

A powerful tool for generating realistic synthetic data using Large Language Models (LLMs). This project consists of a FastAPI backend and a Streamlit frontend that work together to create customizable synthetic datasets for development, testing, and training purposes.

![Synthetic Data Generator](resources/diagram.png)

## Features

- **Custom Column Definitions**: Define your data structure with customizable columns, data types, and constraints
- **LLM-Powered Generation**: Use state-of-the-art language models to create realistic synthetic data
- **Multiple Data Types**: Support for strings, integers, floats, booleans, dates, emails, addresses, and more
- **Background Processing**: Asynchronous task processing for handling large data generation jobs
- **CSV and JSON Export**: Download your generated data in CSV or JSON formats
- **Resumable Generation**: Upload existing CSV files and generate additional matching data
- **Task Management**: View and manage all your generation tasks
- **API First Design**: Modular architecture with a complete REST API backend

## System Requirements

- Python 3.8+
- [Ollama](https://ollama.ai/) for local LLM support
- FastAPI
- Streamlit
- Pandas
- Pydantic

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/synthetic-data-generator.git
cd synthetic-data-generator
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Make sure you have Ollama installed and running with compatible models.

## Configuration

The application uses default configurations that can be modified in the source code:

- API port: 8000 (in `app.py`)
- API URL: http://localhost:8000 (in `streamlit_app.py`)
- Default LLM model: "gemma3:latest" (can be changed in the UI)

## Directory Structure

```
synthetic-data-generator/
├── app.py                # FastAPI backend
├── streamlit_app.py      # Streamlit frontend
├── data/                 # Generated data storage
├── logs/                 # API logs
├── uploads/              # Uploaded file storage
└── requirements.txt      # Project dependencies
```

## Usage

### Starting the Application

1. Start the FastAPI backend:

```bash
uvicorn app:app --reload
```

2. In a new terminal, start the Streamlit frontend:

```bash
streamlit run streamlit_app.py
```

3. Open your browser and navigate to http://localhost:8501 to access the UI.

### Generating Data

#### Method 1: Define Columns and Generate

1. Navigate to the "Column Definition" tab
2. Define your columns with name, data type, description, and constraints
3. Configure generation parameters in the sidebar (model, number of rows, etc.)
4. Click "Generate Data"
5. Switch to the "Results" tab to view and download your generated data

#### Method 2: Upload CSV and Append Data

1. Navigate to the "Resume File Gen" tab
2. Upload an existing CSV file
3. Click "Process CSV" to extract the schema
4. Configure generation parameters
5. Click "Generate and Append Data"
6. Download the combined result once complete

### API Documentation

The API documentation is available at http://localhost:8000/docs when the backend is running.

## Key Endpoints

| Endpoint                      | Method | Description                                         |
| ----------------------------- | ------ | --------------------------------------------------- |
| `/generate`                 | POST   | Generate synthetic data based on column definitions |
| `/task/{task_id}`           | GET    | Get the status of a generation task                 |
| `/data/{task_id}`           | GET    | Get the generated data for a completed task         |
| `/convert_to_csv/{task_id}` | POST   | Convert JSON result to CSV                          |
| `/upload_csv`               | POST   | Upload and analyze a CSV file                       |
| `/append_data`              | POST   | Generate additional data matching an uploaded CSV   |

## Data Types

The system supports the following data types:

- string
- integer
- float
- boolean
- date
- datetime
- email
- phone
- address
- name
- country
- state
- city
- zip
- url
- list

## Constraints

You can add various constraints to your columns:

- For numeric types: minimum and maximum values
- For string types: minimum and maximum length
- Custom constraints can be added through the API

## Error Handling

The application includes comprehensive error handling:

- API errors are logged to `logs/api.log`
- Failed tasks include error messages
- UI displays appropriate error notifications

## How It Works

1. The frontend allows you to define your data schema
2. The backend converts this schema into a prompt for the LLM
3. The LLM generates data in a structured format
4. The backend validates and processes the data
5. The results are stored and made available for download

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with FastAPI and Streamlit
- Powered by Ollama and open source language models
