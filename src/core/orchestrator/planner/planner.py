import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_KEY")
)

# Needs to return the state object with steps
def plan(question: str) -> str:
    planner_prompt = """
        You are an expert planner. Your job is to analyze problems and provide thorough, accurate solutions. Be precise and detailed in your responses.
    """

    completion = client.chat.completions.create(
        model="google/gemini-3-pro-preview",
        messages=[
            {
                "role": "system",
                "content": planner_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return completion.choices[0].message.content
