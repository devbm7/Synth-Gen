import logging.config
from ollama import chat
import logging
from pydantic import BaseModel
from time import time

LOG_DIR = "logs"
DATA_DIR = "data"
MODEL_NAME = "llama3.2:3b"

logging.basicConfig(
    filename=f"{LOG_DIR}/app.log",
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

class Data(BaseModel):
    ID: int
    column1: str

class DataList:
    data: list[Data]

def main():
    print("Hi")