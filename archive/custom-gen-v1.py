from ollama import chat
from pydantic import BaseModel, ValidationError, field_validator
import time
from datetime import datetime
import logging
import json
from pathlib import Path
import os
import argparse
from typing import Optional, List, Dict, Any
import concurrent.futures
from tqdm import tqdm


class Config:
    """Configuration settings for the data generation script."""
    LOG_DIR = "logs"
    DATA_DIR = "data"
    DEFAULT_MODEL = "gemma3:latest"
    MAX_RETRIES = 3


# Ensure directories exist
os.makedirs(Config.LOG_DIR, exist_ok=True)
os.makedirs(Config.DATA_DIR, exist_ok=True)

# Set up logging with rotation
logging.basicConfig(
    filename=f"{Config.LOG_DIR}/custom-gen.log",
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    encoding='utf-8',  # Ensure proper encoding for all characters
)

# Add console handler for better debugging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)


class Movies(BaseModel):
    movie_name: str
    movie_release_date: datetime
    movie_genre: List[str]
    movie_director: str
    movie_lead_cast: List[str]
    movie_production_cost: float
    movie_total_revenue: float

    @field_validator('movie_production_cost', 'movie_total_revenue')
    @classmethod
    def validate_positive_numbers(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Financial values must be positive")
        return v


class MoviesList(BaseModel):
    movies: List[Movies]


class Demographics(BaseModel):
    country_name: str
    state: str
    state_code: int
    total_population: int
    male_population: int
    female_population: int

    @field_validator('total_population', 'male_population', 'female_population')
    @classmethod
    def validate_positive_population(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Population values must be positive")
        return v
    
    @field_validator('total_population')
    @classmethod
    def validate_population_sum(cls, v: int, info) -> int:
        # Access the values that have been validated so far
        values = info.data
        if 'male_population' in values and 'female_population' in values:
            expected_total = values['male_population'] + values['female_population']
            if v != expected_total:
                raise ValueError(f"Total population ({v}) must equal the sum of male ({values['male_population']}) and female ({values['female_population']}) populations")
        return v


class DemographicsList(BaseModel):
    demographics: List[Demographics]


def load_existing_data(file_path: str) -> list:
    """Load existing data from a JSON file with proper error handling."""
    try:
        path = Path(file_path)
        if not path.exists():
            logging.info(f"File {file_path} does not exist. Creating new data set.")
            return []
        
        with open(file_path, "r", encoding='utf-8') as file:
            data = json.load(file)
            logging.info(f"Successfully loaded {len(data)} records from {file_path}")
            return data
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {file_path}: {e}")
        backup_path = f"{file_path}.backup-{int(time.time())}"
        logging.info(f"Creating backup of corrupted file at {backup_path}")
        os.rename(file_path, backup_path)
        return []
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return []


def save_data(data: Dict[str, Any], file_path: str) -> bool:
    """Save data to a JSON file with error handling and atomic write operations."""
    existing_data = load_existing_data(file_path)
    timestamp_data = {"data": data, "timestamp": datetime.now().isoformat()}
    
    if isinstance(existing_data, list):
        existing_data.append(timestamp_data)
    else:
        existing_data = [timestamp_data]
    
    # Use atomic write operations
    temp_file = f"{file_path}.tmp"
    try:
        with open(temp_file, "w", encoding='utf-8') as file:
            json.dump(existing_data, file, indent=4, ensure_ascii=False)
        
        # Rename is an atomic operation on most file systems
        os.replace(temp_file, file_path)
        logging.info(f"Successfully saved data to {file_path}")
        return True
    except Exception as e:
        logging.error(f"Error saving data to {file_path}: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return False


def generate_data(data_type: str, count: int, model_name: str) -> Dict[str, Any]:
    """Generate data of the specified type using the LLM."""
    prompts = {
        "demographics": (
            f"Give me {count} diverse data points for demographics, responding in JSON format. "
            "Each data point should include: country_name (string), "
            "state (string), state_code (integer), total_population (integer), "
            "male_population (integer), and female_population (integer). "
            "Ensure the total_population is the sum of male_population and "
            "female_population. Vary the countries and states."
        ),
        "movies": (
            f"Generate {count} fictional movie entries in JSON format. "
            "Each entry should include: movie_name (string), "
            "movie_release_date (ISO format date), movie_genre (list of strings), "
            "movie_director (string), movie_lead_cast (list of strings), "
            "movie_production_cost (float in millions), "
            "movie_total_revenue (float in millions). "
            "Vary the genres, release years, and financial metrics."
        )
    }
    
    if data_type not in prompts:
        raise ValueError(f"Unknown data type: {data_type}")
    
    schema_map = {
        "demographics": DemographicsList.model_json_schema(),
        "movies": MoviesList.model_json_schema()
    }
    
    validation_map = {
        "demographics": DemographicsList,
        "movies": MoviesList
    }
    
    for attempt in range(Config.MAX_RETRIES):
        try:
            logging.info(f"Generating {data_type} data (attempt {attempt+1}/{Config.MAX_RETRIES})")
            
            response = chat(
                messages=[{"role": "user", "content": prompts[data_type]}],
                model=model_name,
                format=schema_map[data_type]
            )
            
            logging.info(
                "TOKEN SPEED: %s tokens/s",
                format((response.eval_count / response.eval_duration) * (10**9), ".5g"),
            )
            
            # Log response content for debugging
            logging.debug(f"Raw response content: {response.message.content}")
            
            # Check if response content is valid JSON
            try:
                json.loads(response.message.content)
            except json.JSONDecodeError as e:
                logging.error(f"Response is not valid JSON: {e}")
                continue
                
            # Validate the response
            try:
                validated_data = validation_map[data_type].model_validate_json(
                    response.message.content
                )
            except Exception as e:
                logging.error(f"Failed to validate response: {e}")
                continue
            
            logging.info(f"Successfully generated and validated {data_type} data")
            return validated_data.model_dump()
            
        except ValidationError as e:
            logging.error(f"Validation error (attempt {attempt+1}): {e}")
        except Exception as e:
            logging.error(f"Error generating data (attempt {attempt+1}): {e}")
    
    logging.error(f"Failed to generate valid {data_type} data after {Config.MAX_RETRIES} attempts")
    raise RuntimeError(f"Failed to generate valid {data_type} data")


def batch_generate(data_type: str, total_count: int, batch_size: int, model_name: str, parallel: bool = False) -> List[Dict[str, Any]]:
    """Generate data in batches, optionally in parallel."""
    results = []
    
    # Calculate number of batches
    num_batches = (total_count + batch_size - 1) // batch_size
    batches = [min(batch_size, total_count - i * batch_size) for i in range(num_batches)]
    
    if parallel and num_batches > 1:
        logging.info(f"Generating {total_count} {data_type} entries in {num_batches} parallel batches")
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_batches, os.cpu_count() or 4)) as executor:
            futures = [executor.submit(generate_data, data_type, size, model_name) for size in batches]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Generating {data_type}"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logging.error(f"Batch generation failed: {e}")
    else:
        logging.info(f"Generating {total_count} {data_type} entries in {num_batches} sequential batches")
        for size in tqdm(batches, desc=f"Generating {data_type}"):
            try:
                result = generate_data(data_type, size, model_name)
                results.append(result)
            except Exception as e:
                logging.error(f"Batch generation failed: {e}")
    
    return results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate synthetic data using LLMs')
    parser.add_argument('--type', '-t', choices=['demographics', 'movies'], default='demographics',
                        help='Type of data to generate')
    parser.add_argument('--count', '-c', type=int, default=10,
                        help='Number of entries to generate')
    parser.add_argument('--model', '-m', default=Config.DEFAULT_MODEL,
                        help=f'LLM model to use (default: {Config.DEFAULT_MODEL})')
    parser.add_argument('--batch-size', '-b', type=int, default=10,
                        help='Batch size for generation (default: 10)')
    parser.add_argument('--parallel', '-p', action='store_true',
                        help='Enable parallel batch processing')
    parser.add_argument('--output', '-o', default=None,
                        help='Output file name (defaults to data/{type}.json)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    return parser.parse_args()


def main():
    """Main function."""
    start_time = time.time()
    args = parse_arguments()
    
    # Adjust logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
    
    logging.debug("Script started")
    output_file = args.output or f"{Config.DATA_DIR}/{args.type}.json"
    
    try:
        results = batch_generate(
            data_type=args.type,
            total_count=args.count,
            batch_size=args.batch_size,
            model_name=args.model,
            parallel=args.parallel
        )
        
        for result in results:
            if not save_data(result, output_file):
                logging.error("Failed to save some results")
                return 1
        
        elapsed_time = time.time() - start_time
        logging.info(f"Script completed successfully in {elapsed_time:.2f} seconds")
        logging.info(f"Generated data saved to {output_file}")
        return 0
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        logging.error(f"Script failed after {elapsed_time:.2f} seconds: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)