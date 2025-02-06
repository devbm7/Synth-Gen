import logging.config
from ollama import chat
import logging
from pydantic import BaseModel
import time
from datetime import datetime
import json

LOG_DIR = "logs"
DATA_DIR = "data"
MODEL_NAME = "llama3.2:3b"

logging.basicConfig(
    filename=f"{LOG_DIR}/app.log",
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)

def load_existing_data(file_path: str) -> list:
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_data(data: dict, file_path: str):
    existing_data = load_existing_data(file_path)
    timestamp_data = {'data': data, 'timestamp': datetime.now().isoformat()}
    if isinstance(existing_data, list):
        existing_data.append(timestamp_data)
    else:
        existing_data = [timestamp_data]
    
    with open(file_path, 'w') as file:
        json.dump(existing_data, file, indent=4)

class Data(BaseModel):
    ID: int
    column1: str

class DataList(BaseModel):
    data: list[Data]

def main():
    logging.debug("Script Started")
    current_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
    logging.info(f"EXECUTION TIME: {formatted_time}\n\n")
    response = chat(
        messages=[
            {
                'role':'user',
                'content':'Generate 5 datapoints for given dataclass. Here is more information about the dataclass. "ID" represents the identifier for the data entry. "column1" is about english first names.'
            }
        ],
        model=MODEL_NAME,
        format=DataList.model_json_schema()     
    )
    logging.info(f"RESPONSE: {response}")
    logging.info("TOKEN SPEED: %s tokens/s", format((response.eval_count / response.eval_duration)*(10**9), ".5g"))
    data = DataList.model_validate_json(response.message.content)
    logging.info(f"\n{data}")
    save_data(data.model_dump(), file_path=f"{DATA_DIR}/data.json")
    logging.debug("Scrpit Ended")

if __name__ == '__main__':
    main()