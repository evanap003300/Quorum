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

def plan(text: str) -> str:
    """
    Takes in text describing a problem and returns a planning strategy from the LLM.

    Args:
        text: The problem description to plan for

    Returns:
        The LLM's planning response
    """
    completion = client.chat.completions.create(
        model="google/gemini-3-pro-preview",
        messages=[
            {
                "role": "system",
                "content": "You are an expert planner. Your job is to break down complex problems into clear, actionable steps. Provide a detailed plan that outlines the approach to solve the given problem."
            },
            {
                "role": "user",
                "content": text
            }
        ]
    )
    return completion.choices[0].message.content

print(plan("What is 9 + 10"))