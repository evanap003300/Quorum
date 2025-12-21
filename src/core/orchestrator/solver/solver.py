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

# Pricing per 1M tokens (convert to per-token in calculations)
MODEL_PRICING = {
    "google/gemini-3-pro-preview": {
        "input": 2.0,      # $2 per 1M input tokens
        "output": 12.0,    # $12 per 1M output tokens
    },
    "openai/gpt-4.1-mini": {
        "input": 0.4,      # $0.4 per 1M input tokens
        "output": 1.6,     # $1.6 per 1M output tokens
    }
}

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


def solve_step(step: Step, state: StateObject) -> Tuple[bool, Optional[float], Optional[str], Optional[str], float]:
    """
    Execute a single atomic step.

    Args:
        step: The step to execute
        state: Current problem state with variable values

    Returns:
        Tuple of (success, value, unit, error_message, cost)
        - success: True if step executed successfully
        - value: Computed numerical value (None if failed)
        - unit: Unit of the value (None if failed)
        - error_message: Error description (None if success)
        - cost: USD cost of this step
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
            return False, None, None, f"Unknown operation: {step.operation}", 0.0

        # Execute with LLM tool loop
        value, unit, code, cost = _execute_with_llm(prompt)

        return True, value, unit, None, cost

    except Exception as e:
        return False, None, None, str(e), 0.0


def _build_extract_prompt(step: Step, state: StateObject) -> str:
    """Build prompt for extract operations"""
    return f"""Extract the value for '{step.output}' from this problem:

Problem: {state.problem_text}

Variable: {step.output} ({state.variables[step.output].description})
Expected unit: {step.expected_unit}
Hint: {step.justification}

IMPORTANT RULES:
1. For physical constants (e.g., speed of light, gravity, Planck constant, Boltzmann constant), you MUST use the imported constants libraries, NOT approximations
2. Use the exact value from sci_constants or astro_constants, never hardcoded values
3. For problem-given values (like temperatures, distances), extract from the problem text
4. Always print the output as: "<value> <unit>"

Available constants libraries (already imported):
- sci_constants: scipy.constants
  * sci_constants.c = speed of light (299792458 m/s, NOT 3e8)
  * sci_constants.h = Planck constant (6.62607015e-34 J*s)
  * sci_constants.k = Boltzmann constant (1.380649e-23 J/K)
  * sci_constants.G = gravitational constant
  * sci_constants.g = standard gravity (9.80665 m/s^2)
  * sci_constants.e = elementary charge
  * Many more available
- astro_constants: astropy.constants (e.g., astro_constants.G, astro_constants.c)
- element: mendeleev (e.g., element('Fe').atomic_mass)

Example for "from rest":
```python
v0 = 0
unit = "m/s"
print(f"{{v0}} {{unit}}")
```

Example using a constant:
```python
c = sci_constants.c
unit = "m/s"
print(f"{{c}} {{unit}}")
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

IMPORTANT RULES:
1. Use the provided input values exactly as given
2. If any physical constants are needed in the formula but not in inputs, use sci_constants or astro_constants
3. Never use approximations - always use library constants for physical values
4. Perform the calculation precisely using the formula
5. Print the result as: "<value> <unit>"

Available constants libraries (already imported):
- sci_constants: scipy.constants (e.g., sci_constants.c, sci_constants.h, sci_constants.G)
- astro_constants: astropy.constants for astronomical constants
- u: astropy.units for unit handling

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

Available tools:
- pint library (already imported in previous examples)
- scipy.constants (sci_constants) for physical constants
- astropy.units (u) for advanced unit handling

Generate the code."""


def _calculate_cost(completion, model: str) -> float:
    """
    Calculate the cost of a completion based on input/output tokens.

    Args:
        completion: OpenAI completion object with usage info
        model: Model name to look up pricing

    Returns:
        Cost in USD
    """
    if model not in MODEL_PRICING:
        return 0.0

    pricing = MODEL_PRICING[model]
    usage = completion.usage

    # Calculate cost: (tokens * price_per_million) / 1_000_000
    input_cost = (usage.prompt_tokens * pricing["input"]) / 1_000_000
    output_cost = (usage.completion_tokens * pricing["output"]) / 1_000_000

    return input_cost + output_cost


def _execute_with_llm(prompt: str) -> Tuple[float, str, str, float]:
    """
    Execute the LLM tool loop to generate and run code.

    Returns:
        Tuple of (value, unit, code, cost)
        - value: Extracted numerical value
        - unit: Unit of the value
        - code: The code that was executed
        - cost: Total cost in USD for this step
    """

    messages = [
        {
            "role": "system",
            "content": "You are a code generator for physics calculations. Generate clean Python code and use the python_interpreter tool to execute it."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    code = None
    output = None
    max_attempts = 3
    total_cost = 0.0
    model = "openai/gpt-4.1-mini"

    for attempt in range(max_attempts):
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            temperature=0.1
        )

        # Track cost
        total_cost += _calculate_cost(completion, model)

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
                        "content": output if output else "(No output generated)"
                    })

            # If we got output, break out of retry loop
            if output:
                break
        else:
            # finish_reason is "stop" - model didn't use tool
            # Try again, the model might use the tool on next attempt
            if attempt < max_attempts - 1:
                continue
            else:
                raise ValueError(f"Model did not use tool after {max_attempts} attempts")

    if not output:
        raise ValueError("No output generated from code execution")

    # Parse output: expect "value unit" format
    value, unit = _parse_output(output)

    return value, unit, code, total_cost


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