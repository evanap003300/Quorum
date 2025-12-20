import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_KEY")
)

def solve(plan: str) -> str:
    solver_prompt = """
        You are an expert problem solver. Your job is to analyze problems and provide thorough, accurate solutions. Be precise and detailed in your responses.
    """

    completion = client.chat.completions.create(
        model="openai/gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": solver_prompt
            },
            {
                "role": "user",
                "content": plan
            }
        ]
    )

    return completion.choices[0].message.content
