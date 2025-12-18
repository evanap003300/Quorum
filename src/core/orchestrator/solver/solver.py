import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_KEY")
)

def solve(text: str) -> str:
    """
    Takes in text describing a problem and returns a solution from the LLM.

    Args:
        text: The problem description to solve

    Returns:
        The LLM's solution response
    """
    completion = client.chat.completions.create(
        model="openai/gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an expert problem solver. Your job is to analyze problems and provide thorough, accurate solutions. Be precise and detailed in your responses."
            },
            {
                "role": "user",
                "content": text
            }
        ]
    )
    return completion.choices[0].message.content

print(solve("What is 9 + 10"))