from ollama import chat
from pydantic import BaseModel
import time

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
    Add two numbers

    Args:
    a: int: First number
    b: int: Second number

    Returns:
    int: Sum of two numbers
    '''
    return a + b

def main():
    print("=====================================")
    current_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
    print(f"EXECUTION TIME: {formatted_time}\n\n")
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
    print("DEBUG TOKEN SPEED (tokens/s): ", f"{format((response.eval_count / response.eval_duration)*(10**9), ".5g")}", "\n\n")
    country = CountryList.model_validate_json(response.message.content)
    print(country)

    ### Using ToolCall and Function classes for function calls to model
    response = chat(model="llama3.2:1b", tools=[add_two_numbers], messages=[{"role": "user", "content": "What is addition of 2 and 5?"}])
    print("DEBUG ADD TWO NUMBERS RESPONSE: \n", response, "\n\n")
    tool_call = response.message.tool_calls[0]
    print(tool_call)
    function = Function(tool_call.function.name, tool_call.function.arguments)
    function_name = function.name
    function_args = function.arguments
    if function_name == "add_two_numbers":
        result = add_two_numbers(function_args["a"], function_args["b"])
        print(result)

    print("=====================================\n")

if __name__ == '__main__':
    main()