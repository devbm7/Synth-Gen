from ollama import chat
from pydantic import BaseModel
import time
from datetime import datetime
import logging
import json
from pathlib import Path

LOG_DIR = "logs"
DATA_DIR = "data"
MODEL_NAME = "llama3.2:1b"

logging.basicConfig(
    filename=f"{LOG_DIR}/custom-gen.log",
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

class Country(BaseModel):
    country_name: str
    country_capital: str
    country_official_languages: list[str]

class CountryList(BaseModel):
    countries: list[Country]

class ToolCall:
    def __init__(self, function):
        self.function = function

class Function:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments

def add_two_numbers(a: int, b: int) -> int:
    '''
    Addion of    two numbers

    Args:
    a: int: First number
    b: int: Second number

    Returns:
    int: Sum of two numbers
    '''
    return a + b

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

def main():
    logging.debug("Script Started")
    current_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
    logging.info(f"EXECUTION TIME: {formatted_time}\n\n")
    try:
        response = chat(
            messages=[
                {
                    'role': 'user',
                    'content': "Give me 100 data points for given data class. Respond to JSON format.",
                }
            ],
            model=MODEL_NAME,
            format=CountryList.model_json_schema()
        )
        logging.info("TOKEN SPEED: %s tokens/s", format((response.eval_count / response.eval_duration)*(10**9), ".5g"))
        logging.info("FULL RESPONSE: %s", response)
        country = CountryList.model_validate_json(response.message.content)
        logging.info(f"\n{country}")
        # json_data = json.dumps(country.model_dump(), indent=4)
        # with open(f"{DATA_DIR}/countries.json", "w") as file:
        #     file.write(json_data)
        #     file.close()
        save_data(country.model_dump(), f"{DATA_DIR}/countries.json")

        ### Using ToolCall and Function classes for function calls to model
        do_function_calling = False
        if do_function_calling:
            response = chat(model="llama3.2:1b", tools=[add_two_numbers], messages=[{"role": "user", "content": "What is addition of 2 and 5?"}])
            logging.debug("DEBUG ADD TWO NUMBERS RESPONSE: \n", response, "\n\n")
            tool_call = response.message.tool_calls[0]
            function = Function(tool_call.function.name, tool_call.function.arguments)
            function_name = function.name
            function_args = function.arguments
            if function_name == "add_two_numbers":
                result = add_two_numbers(function_args["a"], function_args["b"])
                logging.info(f"Addition of {function_args} is {result}")

        logging.debug("Script Ended")
    except Exception as e:
        logging.error("Error: \n", e)
        return

if __name__ == '__main__':
    main()