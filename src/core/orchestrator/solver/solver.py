import os
import json
import asyncio
from typing import Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI
import importlib.util
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from planner.schema import Step, StateObject

# Import python interpreter
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

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "python_interpreter",
            "description": "Execute Python code to perform calculations. Always print the final result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            }
        }
    }
]


def solve_step(step: Step, state: StateObject) -> Tuple[bool, Optional[float], Optional[str], Optional[str]]:
    """
    Execute a single atomic step.
    
    Args:
        step: The step to execute
        state: Current problem state with variable values
        
    Returns:
        Tuple of (success, value, unit, error_message)
        - success: True if step executed successfully
        - value: Computed numerical value (None if failed)
        - unit: Unit of the value (None if failed)
        - error_message: Error description (None if success)
    """
    
    try:
        # Build prompt based on step type
        if step.operation == "extract":
            prompt = _build_extract_prompt(step, state)
        elif step.operation == "calculate":
            prompt = _build_calculate_prompt(step, state)
        elif step.operation == "convert":
            prompt = _build_convert_prompt(step, state)
        else:
            return False, None, None, f"Unknown operation: {step.operation}"
        
        # Execute with LLM tool loop
        value, unit, code = _execute_with_llm(prompt)
        
        return True, value, unit, None
        
    except Exception as e:
        return False, None, None, str(e)


def _build_extract_prompt(step: Step, state: StateObject) -> str:
    """Build prompt for extract operations"""
    return f"""Extract the value for '{step.output}' from this problem:

Problem: {state.problem_text}

Variable: {step.output} ({state.variables[step.output].description})
Expected unit: {step.expected_unit}
Hint: {step.justification}

Write Python code that:
1. Defines the value based on the problem text
2. Assigns it to variable '{step.output}'
3. Prints: "<value> <unit>"

Example for "from rest":
```python
v0 = 0
unit = "m/s"
print(f"{{v0}} {{unit}}")
```

Generate the code."""


def _build_calculate_prompt(step: Step, state: StateObject) -> str:
    """Build prompt for calculate operations"""
    
    # Get input values
    inputs_text = ""
    for input_var in step.inputs:
        var = state.variables[input_var]
        inputs_text += f"{input_var} = {var.value} {var.unit}\n"
    
    return f"""Perform this calculation:

Inputs:
{inputs_text}

Task: {step.description}
Formula: {step.formula}
Expected output unit: {step.expected_unit}

Write Python code that:
1. Defines the input variables
2. Calculates '{step.output}' using the formula
3. Prints: "<value> <unit>"

Example:
```python
import math

v0 = 0
a = 2
t = 5

v = v0 + a * t
unit = "m/s"

print(f"{{v}} {{unit}}")
```

Generate the code."""


def _build_convert_prompt(step: Step, state: StateObject) -> str:
    """Build prompt for convert operations"""
    
    input_var = step.inputs[0]
    source = state.variables[input_var]
    
    return f"""Convert this value:

Input: {source.value} {source.unit}
Target unit: {step.expected_unit}

Write Python code to convert and print: "<value> <unit>"

You can use basic conversion factors or the pint library.

Generate the code."""


def _execute_with_llm(prompt: str) -> Tuple[float, str, str]:
    """
    Execute the LLM tool loop to generate and run code.
    
    Returns:
        Tuple of (value, unit, code)
    """
    
    messages = [
        {
            "role": "system",
            "content": "You are a code generator for physics calculations. Generate clean Python code."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    code = None
    output = None
    max_attempts = 3
    
    for attempt in range(max_attempts):
        completion = client.chat.completions.create(
            model="google/gemini-3-pro-preview",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.1
        )
        
        choice = completion.choices[0]
        
        if choice.finish_reason == "tool_calls":
            assistant_message = choice.message
            messages.append(assistant_message)
            
            for tool_call in assistant_message.tool_calls:
                if tool_call.function.name == "python_interpreter":
                    args = json.loads(tool_call.function.arguments)
                    code = args["code"]
                    
                    # Execute the code
                    output = asyncio.run(run_python(code))
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": output
                    })
        else:
            # Done
            break
    
    if not output:
        raise ValueError("No output generated from code execution")
    
    # Parse output: expect "value unit" format
    value, unit = _parse_output(output)
    
    return value, unit, code


def _parse_output(output: str) -> Tuple[float, str]:
    """
    Parse execution output to extract value and unit.
    Expected format: "10.0 m/s"
    """
    
    parts = output.strip().split(None, 1)  # Split on first whitespace
    
    if len(parts) < 2:
        # Try to parse as just a number
        try:
            value = float(parts[0])
            return value, "dimensionless"
        except:
            raise ValueError(f"Could not parse output: {output}")
    
    try:
        value = float(parts[0])
        unit = parts[1].strip()
        return value, unit
    except ValueError as e:
        raise ValueError(f"Could not parse output '{output}': {e}")


# Simple test
if __name__ == "__main__":
    from planner import plan
    
    problem = "A car accelerates from rest at 2 m/s² for 5 seconds. What is its final velocity?"
    
    state, plan_obj = plan(problem)
    
    print("Testing first step...")
    step = plan_obj.steps[0]
    
    success, value, unit, error = solve_step(step, state)
    
    if success:
        print(f"✓ {step.output} = {value} {unit}")
    else:
        print(f"✗ Failed: {error}")