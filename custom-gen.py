from ollama import chat
from pydantic import BaseModel, ValidationError
import time
from datetime import datetime
import logging
import json
from pathlib import Path
import os

LOG_DIR = "logs"
DATA_DIR = "data"
MODEL_NAME = "gemma3:latest"

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(
    filename=f"{LOG_DIR}/custom-gen.log",
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)


class Movies(BaseModel):
    movie_name: str
    movie_release_date: datetime
    movie_genre: list[str]
    movie_director: str
    movie_lead_cast: list[str]
    movie_production_cost: float
    movie_total_revenue: float


class MoviesList(BaseModel):
    movies: list[Movies]


class Demographics(BaseModel):
    country_name: str
    state: str
    state_code: int
    total_population: int
    male_population: int
    female_population: int


class DemographicsList(BaseModel):
    demographics: list[Demographics]


def load_existing_data(file_path: str) -> list:
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_data(data: dict, file_path: str):
    existing_data = load_existing_data(file_path)
    timestamp_data = {"data": data, "timestamp": datetime.now().isoformat()}
    if isinstance(existing_data, list):
        existing_data.append(timestamp_data)
    else:
        existing_data = [timestamp_data]

    with open(file_path, "w") as file:
        json.dump(existing_data, file, indent=4)


def main():
    logging.debug("Script Started")
    current_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
    logging.info(f"EXECUTION TIME: {formatted_time}\n\n")
    try:
        prompt = (
            "Give me 10 diverse data points for demographics, responding in JSON format."
            "Each data point should include: country_name (string),"
            "state (string), state_code (integer), total_population (integer),"
            "male_population (integer), and female_population (integer)."
            "Ensure the total_population is the sum of male_population and"
            "female_population.  Vary the countries and states."
        )

        response = chat(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_NAME,
            format=DemographicsList.model_json_schema(),
        )
        logging.info(
            "TOKEN SPEED: %s tokens/s",
            format((response.eval_count / response.eval_duration) * (10**9), ".5g"),
        )
        logging.info("FULL RESPONSE: %s", response)

        try:
            country = DemographicsList.model_validate_json(
                response.message.content
            )
            logging.info(f"\n{country}")
            save_data(country.model_dump(), f"{DATA_DIR}/demo.json")
        except ValidationError as e:
            logging.error(f"Pydantic Validation Error: {e}")
            return 1  # Indicate failure

        logging.debug("Script Ended")
        return 0  # Indicate success

    except Exception as e:
        logging.error(f"Error: {e}")
        return 1  # Indicate failure


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
