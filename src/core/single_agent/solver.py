"""Single-agent solver using Gemini 3 Pro with direct code execution."""

import asyncio
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv

# Configure path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

try:
    from e2b_code_interpreter import Sandbox
except ImportError:
    Sandbox = None

# Import code interpreter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'orchestrator'))
from tools.evaluation.code_interpreter import run as run_python_impl

# Import centralized pricing config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'orchestrator', 'config'))
from pricing import MODEL_PRICING

load_dotenv()

# Configure Gemini client
if GENAI_AVAILABLE:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# System prompt for the agent
SYSTEM_PROMPT = """You are an expert physics and mathematics problem solver. You excel at solving complex problems step-by-step by reasoning carefully and using Python code for all calculations.

## Your Capabilities

You have access to a Python interpreter with these libraries pre-loaded:
- numpy as np (numerical computations)
- sympy as sp (symbolic mathematics)
- scipy.constants as sci_constants (physical constants like c, G, g)
- pint (unit conversions with ureg)
- astropy.units and astropy.constants (astrophysics constants)

The Python environment also has commonly used constants:
- Speed of light: sci_constants.c (m/s)
- Gravitational constant: sci_constants.G (m³/(kg·s²))
- Gravitational acceleration: sci_constants.g (m/s²)
- Planck constant: sci_constants.h (J·s)
- And many others available via sci_constants

## Problem-Solving Approach

When solving a problem, follow these steps:

1. **READ**: Carefully analyze the problem text to identify:
   - What information is given (initial conditions, constants, parameters)
   - What is being asked for (target quantity and its expected unit)
   - The domain (physics, chemistry, mathematics, etc.)
   - Any constraints or special conditions

2. **PLAN**: Mentally outline your solution strategy:
   - What equations or methods apply?
   - What intermediate calculations are needed?
   - In what order should calculations be performed?
   - Should you work symbolically or numerically?

3. **EXECUTE**: Use Python to perform calculations:
   - Extract and validate given values
   - Set up equations (symbolic with sympy or numeric with numpy)
   - Perform calculations step by step
   - Print intermediate results to verify correctness
   - Always use proper units throughout

4. **VERIFY**: Check your final answer:
   - Are the units correct and consistent?
   - Is the magnitude reasonable (not wildly off)?
   - Does the answer address what was asked?
   - Have you handled all special cases?

## Tool Usage Guidelines

**When to Use Python:**
- Use the python_interpreter tool for ALL numerical calculations
- Use it for unit conversions, symbolic derivations, and complex arithmetic
- Never attempt mental math - always verify calculations with code

**Code Structure:**
- Start each code block with clear comments explaining what you're doing
- Define variables with meaningful names that match the problem notation
- Print intermediate values to show your work
- Use proper unit handling (sympy units or pint)
- Always print the final result clearly

**Example Code Pattern:**
```python
# Extract given values
m = 5.0  # mass in kg
a = 2.0  # acceleration in m/s²
t = 10.0  # time in s

# Calculate result
v = a * t  # v = at for motion from rest

# Print result
print(f"Final velocity: {v} m/s")
```

**Error Handling:**
- If you get an error, read the error message carefully
- Explain what went wrong
- Adjust your code and try again
- Keep iterating until you get a valid result

## Critical Rules

1. **ALWAYS USE UNITS**: Never work with unitless numbers when units are required
   - Example good: `5 * u.meter`, `10 * u.second`
   - Example bad: just `5` or `10`

2. **VALIDATE UNITS**: Check that intermediate results have correct units
   - Use unit_converter if different units need to be compared
   - Ensure final answer has the expected unit

3. **PYTHON IS MANDATORY**: Use the python_interpreter tool for all calculations
   - This includes simple arithmetic
   - This ensures accuracy and reproducibility

4. **CLEAR OUTPUT FORMAT**: End your response with a clearly formatted answer
   - Use: ANSWER: <value> <unit>
   - Example: ANSWER: 98.5 m/s
   - For symbolic: ANSWER: m*v₀²/(2*d) J

5. **SHOW YOUR WORK**: Print values at each step so your reasoning is clear
   - This helps verify correctness
   - This makes it easy to spot errors

6. **RECOVER FROM ERRORS**: If code fails:
   - Don't give up immediately
   - Examine the error
   - Try a different approach
   - Iterate until you find a solution

## Examples

### Example 1: Simple Kinematics
Problem: "A car accelerates from rest at 2 m/s² for 5 seconds. What is its final velocity?"

Solution:
```python
import numpy as np

# Given values
a = 2.0  # acceleration in m/s²
t = 5.0  # time in s
v0 = 0   # starts from rest

# Calculate final velocity using v = v0 + at
v = v0 + a * t

print(f"Initial velocity: {v0} m/s")
print(f"Acceleration: {a} m/s²")
print(f"Time: {t} s")
print(f"Final velocity: {v} m/s")
```

ANSWER: 10.0 m/s

### Example 2: Unit Conversion
Problem: "Convert 72 km/h to m/s"

Solution:
```python
import pint
ureg = pint.UnitRegistry()

# Create quantity with units
speed = 72 * ureg.kilometer / ureg.hour

# Convert to m/s
speed_ms = speed.to(ureg.meter / ureg.second)

print(f"Speed in km/h: {speed}")
print(f"Speed in m/s: {speed_ms}")
print(f"Speed (numeric): {speed_ms.magnitude}")
```

ANSWER: 20.0 m/s

Now you're ready to solve problems. Go step-by-step, use Python for all calculations, and clearly state your final answer in the required format."""

# Python tool definition for Gemini
PYTHON_TOOL = {
    "function_declarations": [
        {
            "name": "python_interpreter",
            "description": "Execute Python code for calculations. Always use this for arithmetic, unit conversion, and symbolic math. Print results to see output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Always print intermediate and final results."
                    }
                },
                "required": ["code"]
            }
        }
    ]
}

# Pricing for Gemini 3 Pro
PRICING = {
    "input": 2.0 / 1_000_000,    # $2 per 1M tokens → per token
    "output": 12.0 / 1_000_000,   # $12 per 1M tokens → per token
}


async def run_python(code: str, sandbox: Optional["Sandbox"] = None) -> str:
    """Execute Python code in a sandbox."""
    return await run_python_impl(code, sandbox)


def _calculate_cost(
    input_tokens: int, output_tokens: int, model: str = "gemini-3-pro-preview"
) -> float:
    """Calculate cost for API call using centralized pricing."""
    if model not in MODEL_PRICING:
        return 0.0

    pricing = MODEL_PRICING[model]
    return ((input_tokens * pricing["input"]) + (output_tokens * pricing["output"])) / 1_000_000


def _extract_answer(response: str) -> Tuple[Union[float, str], str]:
    """Extract final answer and unit from agent response.

    Args:
        response: Full agent response text

    Returns:
        Tuple of (value, unit) where value is float or string

    Raises:
        ValueError: If no valid answer found
    """
    # Look for "ANSWER: <value> <unit>" pattern
    pattern = r'ANSWER:\s*([^\s]+)\s+(.+?)(?:\n|$)'
    match = re.search(pattern, response, re.IGNORECASE)

    if not match:
        raise ValueError(f"Could not find ANSWER in response:\n{response}")

    value_str = match.group(1).strip()
    unit = match.group(2).strip()

    # Try to parse as float, otherwise keep as symbolic expression
    try:
        value = float(value_str)
    except ValueError:
        # Symbolic expression (e.g., "m*v/(F+mg)")
        value = value_str

    return value, unit


async def _run_agent_loop(
    problem: str,
    sandbox: Optional["Sandbox"] = None,
    max_iterations: int = 7,
    model: str = "gemini-3-pro-preview",
) -> Tuple[str, List[str], float]:
    """Run agent conversation loop with tool calling.

    Args:
        problem: Problem text to solve
        sandbox: Optional existing sandbox for code execution
        max_iterations: Maximum number of tool calls allowed
        model: Model name to use (default: gemini-3-pro-preview)

    Returns:
        Tuple of (final_response, code_list, total_cost)
    """
    # Initialize Gemini client
    model_name = model

    # Create GenerativeModel with tools
    from google.generativeai.types import FunctionDeclaration, Tool

    # Convert tool definition format
    tool = Tool(
        function_declarations=PYTHON_TOOL["function_declarations"]
    )

    client = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_PROMPT,
        tools=[tool],
        generation_config={
            "temperature": 0.1,
        },
    )

    # Start chat session
    chat = client.start_chat()

    code_executed = []
    total_cost = 0.0
    iteration = 0

    # Agentic loop
    while iteration < max_iterations:
        iteration += 1

        try:
            # Send message to chat
            if iteration == 1:
                # First message is the problem
                response = chat.send_message(problem)
            else:
                # Subsequent messages are function responses
                response = chat.send_message(genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name="python_interpreter",
                        response={"result": last_output}
                    )
                ))

            # Track tokens and cost
            if hasattr(response, 'usage_metadata'):
                total_cost += _calculate_cost(
                    response.usage_metadata.prompt_token_count,
                    response.usage_metadata.candidates_token_count,
                    model_name
                )

            # Check if model used a tool
            function_call = None
            if (response.candidates and
                response.candidates[0].content.parts and
                hasattr(response.candidates[0].content.parts[0], 'function_call')):
                function_call = response.candidates[0].content.parts[0].function_call

            if function_call is not None:
                if function_call.name == "python_interpreter":
                    code = function_call.args.get("code", "")

                    if code:
                        code_executed.append(code)

                        try:
                            # Execute code in sandbox
                            last_output = await run_python(code, sandbox)
                        except Exception as e:
                            last_output = f"Error executing code: {str(e)}"

                        # Continue loop to send output back
                        continue

            # No more tool calls - agent finished with text response
            if response.text:
                return response.text, code_executed, total_cost

        except Exception as e:
            error_msg = f"Error in agent loop: {str(e)}"
            import traceback
            error_msg += f"\n{traceback.format_exc()}"
            return error_msg, code_executed, total_cost

    # Max iterations exceeded
    return f"Max iterations ({max_iterations}) exceeded", code_executed, total_cost


async def solve_problem(
    problem: str = "",
    max_iterations: int = 7,
    timeout: int = 120,
    sandbox: Optional["Sandbox"] = None,
    model: str = "gemini-3-pro-preview",
) -> Dict[str, Any]:
    """Solve a physics/math problem using single-agent Gemini approach.

    Args:
        problem: Problem text to solve
        max_iterations: Maximum number of tool calls allowed (default 7)
        timeout: Timeout in seconds (currently unused, for API compatibility)
        sandbox: Optional pre-initialized sandbox for code execution
        model: Model name to use (default: gemini-3-pro-preview)

    Returns:
        Result dict with keys:
        - success: bool - Whether solving was attempted
        - final_answer: float | str - Numeric or symbolic answer
        - final_unit: str - Unit of answer
        - reasoning_trace: str - Full agent response
        - code_executed: List[str] - All code snippets executed
        - num_iterations: int - Number of tool calls made
        - total_time: float - Total execution time in seconds
        - total_cost: float - Total API cost in USD
        - error: Optional[str] - Error message if any
    """
    start_time = time.time()

    # Validate inputs
    if not problem:
        return {
            "success": False,
            "final_answer": None,
            "final_unit": None,
            "reasoning_trace": "",
            "code_executed": [],
            "num_iterations": 0,
            "total_time": 0.0,
            "total_cost": 0.0,
            "error": "No problem text provided"
        }

    if not GENAI_AVAILABLE:
        return {
            "success": False,
            "final_answer": None,
            "final_unit": None,
            "reasoning_trace": "",
            "code_executed": [],
            "num_iterations": 0,
            "total_time": 0.0,
            "total_cost": 0.0,
            "error": "google-generativeai not available"
        }

    try:
        # Run agent loop
        response_text, code_executed, total_cost = await _run_agent_loop(
            problem,
            sandbox=sandbox,
            max_iterations=max_iterations,
            model=model,
        )

        # Extract answer
        try:
            final_answer, final_unit = _extract_answer(response_text)
        except ValueError as e:
            return {
                "success": False,
                "final_answer": None,
                "final_unit": None,
                "reasoning_trace": response_text,
                "code_executed": code_executed,
                "num_iterations": len(code_executed),
                "total_time": time.time() - start_time,
                "total_cost": total_cost,
                "error": str(e)
            }

        # Success
        return {
            "success": True,
            "final_answer": final_answer,
            "final_unit": final_unit,
            "reasoning_trace": response_text,
            "code_executed": code_executed,
            "num_iterations": len(code_executed),
            "total_time": time.time() - start_time,
            "total_cost": total_cost,
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "final_answer": None,
            "final_unit": None,
            "reasoning_trace": "",
            "code_executed": [],
            "num_iterations": 0,
            "total_time": time.time() - start_time,
            "total_cost": 0.0,
            "error": f"Exception: {str(e)}"
        }
