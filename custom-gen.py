from ollama import chat
from pydantic import BaseModel

class Country(BaseModel):
    country_name: str
    country_capital: str
    country_official_languages: list[str]

class CountryList(BaseModel):
    countries: list[Country]

def add_two_numbers(a: int, b: int) -> int:
    '''
    Add two numbers

    Args:
    a: int: First number
    b: int: Second number

    Returns:
    int: Sum of two numbers
    '''
    return a + b

def main():
    response = chat(
        messages=[
            {
                'role': 'user',
                'content': "Tell me about two north american countries. Respond to JSON format.",
            }
        ],
        model='llama3.2:1b',
        format=CountryList.model_json_schema()
    )
    print("DEBUG FULL RESPONSE: \n", response, "\n\n")
    print("DEBUG RESPONSE MESSAGE: \n", response.message, "\n\n")
    print("DEBUG TOKEN SPEED (tokens/s): ", f"{format((response.eval_count / response.eval_duration)*(10**9), ".5g")}", "\n\n")
    country = CountryList.model_validate_json(response.message.content)
    print(country)

    print("DEBUG ADD TWO NUMBERS: ", add_two_numbers(3, 4))
    response = chat(model="llama3.2:1b", tools=[add_two_numbers], messages=[{"role": "user", "content": "What is addition of 2 and 5?"}])
    print(response.message)

if __name__ == '__main__':
    main()