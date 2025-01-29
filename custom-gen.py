from ollama import chat
from pydantic import BaseModel

class Country(BaseModel):
    country_name: str
    country_capital: str
    languages: list[str]

class CountryList(BaseModel):
    countries: list[Country]

def main():
    response = chat(
        messages=[
            # {
            #     'role': 'user',
            #     'content': "Generate a list of containing two countries with their capitals and languages.",
            # },
            {
                'role': 'user',
                'content': "Tell me about two north american countries.",
            }
        ],
        model='phi4:latest',
        format=CountryList.model_json_schema()
    )
    country = CountryList.model_validate_json(response.message.content)
    print(country)

if __name__ == '__main__':
    main()