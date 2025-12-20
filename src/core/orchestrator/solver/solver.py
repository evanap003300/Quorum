import os
import json
import asyncio
import sys
import importlib.util
from dotenv import load_dotenv
from openai import OpenAI

# Import python interpreter from hyphenated directory
spec = importlib.util.spec_from_file_location(
    "python_interpreter_e2b",
    os.path.join(os.path.dirname(__file__), "python_interpreter-e2b", "main.py")
)
python_interpreter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(python_interpreter_module)
run_python = python_interpreter_module.run

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_KEY")
)

# Define the Python interpreter tool
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "python_interpreter",
            "description": "Execute Python code in a sandboxed environment. Use this to run calculations, analyze data, or perform computations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute"
                    }
                },
                "required": ["code"]
            }
        }
    }
]

def solve(plan: str) -> str:
    solver_prompt = """
        You are an expert problem solver with access to a Python interpreter. Your job is to analyze problems and provide thorough, accurate solutions.

        When you need to perform calculations, run code, analyze data, or verify results, use the python_interpreter tool to execute Python code.

        Be precise and detailed in your responses. Show your work and explain your reasoning.
    """

    messages = [
        {
            "role": "system",
            "content": solver_prompt
        },
        {
            "role": "user",
            "content": plan
        }
    ]

    # Agentic loop for tool calling
    while True:
        completion = client.chat.completions.create(
            model="openai/gpt-4.1-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        # Check if the model wants to use a tool
        if completion.choices[0].finish_reason == "tool_calls":
            assistant_message = completion.choices[0].message
            messages.append(assistant_message)

            # Process tool calls
            for tool_call in assistant_message.tool_calls:
                if tool_call.function.name == "python_interpreter":
                    args = json.loads(tool_call.function.arguments)
                    result = asyncio.run(run_python(args["code"]))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
        else:
            # Model is done - return the response
            return completion.choices[0].message.content
