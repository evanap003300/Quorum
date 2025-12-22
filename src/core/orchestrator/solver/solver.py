import os
import json
import asyncio
from typing import List, Optional, Tuple, Union
from dotenv import load_dotenv
from openai import OpenAI
import importlib.util
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from planner.schema import Step, StateObject

try:
    from e2b_code_interpreter import Sandbox
except ImportError:
    Sandbox = None

# Import python interpreter
spec = importlib.util.spec_from_file_location(
    "python_interpreter_e2b",
    os.path.join(os.path.dirname(__file__), "python_interpreter-e2b", "main.py")
)
python_interpreter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(python_interpreter_module)
_run_python_impl = python_interpreter_module.run

async def run_python(code: str, sandbox: Optional["Sandbox"] = None) -> str:
    """
    Execute Python code in a sandbox.

    Args:
        code: Python code to execute
        sandbox: Optional existing sandbox. If None, creates and destroys per call.

    Returns:
        Output from code execution
    """
    return await _run_python_impl(code, sandbox)

load_dotenv()

# Prefer direct OpenAI API for better latency, fall back to OpenRouter
if os.getenv("OPENAI_API_KEY"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    _using_direct_api = True
else:
    # Fallback to OpenRouter
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_KEY")
    )
    _using_direct_api = False

# Pricing per 1M tokens (convert to per-token in calculations)
MODEL_PRICING = {
    "google/gemini-3-pro-preview": {
        "input": 2.0,      # $2 per 1M input tokens
        "output": 12.0,    # $12 per 1M output tokens
    },
    "openai/gpt-4.1-mini": {
        "input": 0.4,      # $0.4 per 1M input tokens
        "output": 1.6,     # $1.6 per 1M output tokens
    },
    "gpt-4.1-mini-2025-04-14": {
        "input": 0.4,      # $0.40 per 1M input tokens
        "output": 1.6,     # $1.60 per 1M output tokens
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


def solve_step(step: Step, state: StateObject, sandbox: Optional["Sandbox"] = None) -> Tuple[bool, Optional[Union[float, str, dict]], Optional[Union[str, dict]], Optional[str], float]:
    """
    Execute a single atomic step.

    Args:
        step: The step to execute
        state: Current problem state with variable values
        sandbox: Optional existing sandbox to reuse. If None, creates new sandbox per execution.

    Returns:
        For single-output steps: Tuple of (success, value, unit, error_message, cost)
        For multi-output steps: Tuple of (success, dict_of_values, dict_of_units, error_message, cost)
        - success: True if step executed successfully
        - value: Computed value(s) - float/str for single, dict for multiple outputs
        - unit: Unit(s) - str for single, dict for multiple outputs
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

        # Check if this is a multi-output extraction step
        outputs = step.get_outputs()
        is_multi_output = len(outputs) > 1 and step.operation == "extract"

        # Execute with LLM tool loop
        if is_multi_output:
            values, units, code, cost = _execute_with_llm_multi_output(prompt, outputs, step, state, sandbox)
        else:
            value, unit, code, cost = _execute_with_llm(prompt, sandbox)
            values, units = value, unit

        return True, values, units, None, cost

    except Exception as e:
        return False, None, None, str(e), 0.0


def _build_extract_prompt(step: Step, state: StateObject) -> str:
    """Build prompt for extract operations (single or multiple variables)"""
    outputs = step.get_outputs()

    if len(outputs) == 1:
        # Single output - original format
        var = state.variables[outputs[0]]
        base_prompt = f"""Extract {outputs[0]} ({var.description}) from this problem:

Problem: {state.problem_text}

Expected unit: {step.get_unit(outputs[0])}

Print: VALUE UNIT

Use sci_constants for constants (not approximations).
Extract from problem text for given values."""

        if step.is_symbolic:
            symbolic_section = f"""

SYMBOLIC: Print the symbol name and unit.
Format: {outputs[0]} {step.get_unit(outputs[0])}
"""
            return base_prompt + symbolic_section
        else:
            return base_prompt
    else:
        # Multiple outputs - batch extraction
        variables_info = ""
        for var_name in outputs:
            var = state.variables[var_name]
            unit = step.get_unit(var_name)
            variables_info += f"  {var_name}: {var.description} ({unit})\n"

        base_prompt = f"""Extract values from this problem:

Problem: {state.problem_text}

Extract these variables:
{variables_info}

Print output as: VAR_NAME VALUE UNIT (one per line)
- For numeric: "v0 0 m/s"
- For symbolic: "L L m"

Use sci_constants for physical constants (c, g, etc). No approximations.

Generate code that prints only the results (3+ lines, nothing else)."""

        if step.is_symbolic:
            symbolic_section = ""
            return base_prompt
        else:
            return base_prompt


def _build_calculate_prompt(step: Step, state: StateObject) -> str:
    """Build prompt for calculate operations"""

    # Get input values
    inputs_text = ""
    for input_var in step.inputs:
        var = state.variables[input_var]
        inputs_text += f"{input_var} = {var.value} {var.unit}\n"

    base_prompt = f"""Calculate {step.description}:

Inputs:
{inputs_text}

Formula: {step.formula}
Output unit: {step.expected_unit}

Print result as: VALUE UNIT

Use sci_constants for constants. Simplify symbolic results.
Use only the provided inputs - compute nothing else."""

    if step.is_symbolic:
        symbolic_section = f"""

SYMBOLIC CALCULATION:
- Define symbolic variables: sp.symbols('L V u ...', positive=True, real=True)
- Apply formula and simplify: sp.simplify(result)
- CRITICAL: Preserve units! Result must have unit {step.expected_unit}
- If you get dimensionless, you dropped a factor
- Print: EXPR UNIT (e.g., "L*log(V/u)/V s")
"""
        return base_prompt + symbolic_section
    else:
        return base_prompt


def _build_convert_prompt(step: Step, state: StateObject) -> str:
    """Build prompt for convert operations"""

    input_var = step.inputs[0]
    source = state.variables[input_var]

    base_prompt = f"""Convert this value:

Input: {source.value} {source.unit}
Target unit: {step.expected_unit}

Write Python code to convert and print: "<value> <unit>"

Available tools:
- pint library (already imported in previous examples)
- scipy.constants (sci_constants) for physical constants
- astropy.units (u) for advanced unit handling

Generate the code."""

    if step.is_symbolic:
        symbolic_section = """

IMPORTANT: This is a SYMBOLIC conversion.
- The input is a symbolic expression (string), not a numeric value
- Use sympy for symbolic unit conversion
- Simplify with sp.simplify() before printing
- Format: print(f"{{str(sp.simplify(result))}} {{target_unit}}")
"""
        return base_prompt + symbolic_section
    else:
        return base_prompt


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


def _execute_with_llm(prompt: str, sandbox: Optional["Sandbox"] = None) -> Tuple[Union[float, str], str, str, float]:
    """
    Execute the LLM tool loop to generate and run code.

    Args:
        prompt: The prompt to send to LLM
        sandbox: Optional existing sandbox to reuse

    Returns:
        Tuple of (value, unit, code, cost)
        - value: Numerical value (float) or symbolic expression (str)
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
    # Use gpt-4.1-mini-2025-04-14 for better code generation and variable naming
    model = "gpt-4.1-mini-2025-04-14" if _using_direct_api else "openai/gpt-4.1-mini"

    for attempt in range(max_attempts):
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            temperature=0.1  # Slight randomness helps avoid degenerate code generation
        )

        # Track cost
        total_cost += _calculate_cost(completion, model)

        choice = completion.choices[0]

        assistant_message = choice.message

        if choice.finish_reason == "tool_calls" and assistant_message.tool_calls:
            # Model used a tool - execute it
            messages.append(assistant_message)

            for tool_call in assistant_message.tool_calls:
                if tool_call.function.name == "python_interpreter":
                    args = json.loads(tool_call.function.arguments)
                    code = args["code"]

                    # Execute the code
                    output = asyncio.run(run_python(code, sandbox))

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": output if output else "(No output generated)"
                    })

            # If we got output, break out of retry loop
            if output:
                break
        else:
            # Model finished without using a tool
            # Try to extract result from the text response
            if assistant_message.content:
                extracted = _extract_result_from_text(assistant_message.content)
                if extracted:
                    output = extracted
                    code = "(No code generated - direct text response)"
                    break

            # If we couldn't extract anything, retry or give up
            if attempt < max_attempts - 1:
                # Add the message to history for context on next attempt
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content or "(No response)"
                })
                messages.append({
                    "role": "user",
                    "content": "Please try again. Generate a clear result in the format: '<value> <unit>'"
                })
                continue
            else:
                # Last attempt failed - raise error
                raise ValueError("Could not generate valid output from model after 3 attempts")

    if not output:
        raise ValueError("No output generated from code execution")

    # Parse output: expect "value unit" format
    value, unit = _parse_output(output)

    return value, unit, code, total_cost


def _execute_with_llm_multi_output(prompt: str, outputs: List[str], step: Step, state: StateObject, sandbox: Optional["Sandbox"] = None) -> Tuple[dict, dict, str, float]:
    """
    Execute the LLM tool loop for multi-output extraction.

    Args:
        prompt: The extraction prompt
        outputs: List of variable names to extract
        step: The step being executed
        state: Current state for looking up variable info
        sandbox: Optional existing sandbox to reuse

    Returns:
        Tuple of (values_dict, units_dict, code, cost)
        - values_dict: {"var_name": value, ...}
        - units_dict: {"var_name": "unit", ...}
        - code: The code that was executed
        - cost: Total cost in USD for this step
    """

    messages = [
        {
            "role": "system",
            "content": "You are a code generator for physics calculations. Generate clean Python code that extracts multiple values and prints them in the required format. Use the python_interpreter tool to execute it."
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
    # Use gpt-4.1-mini-2025-04-14 for better code generation and variable naming
    model = "gpt-4.1-mini-2025-04-14" if _using_direct_api else "openai/gpt-4.1-mini"

    for attempt in range(max_attempts):
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            temperature=0.1  # Slight randomness helps avoid degenerate code generation
        )

        # Track cost
        total_cost += _calculate_cost(completion, model)

        choice = completion.choices[0]

        assistant_message = choice.message

        if choice.finish_reason == "tool_calls" and assistant_message.tool_calls:
            # Model used a tool - execute it
            messages.append(assistant_message)

            for tool_call in assistant_message.tool_calls:
                if tool_call.function.name == "python_interpreter":
                    args = json.loads(tool_call.function.arguments)
                    code = args["code"]

                    # Execute the code
                    output = asyncio.run(run_python(code, sandbox))

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": output if output else "(No output generated)"
                    })

            # If we got output, break out of retry loop
            if output:
                break
        else:
            # Model finished without using a tool
            # Try to extract result from the text response
            if assistant_message.content:
                extracted = _extract_result_from_text(assistant_message.content)
                if extracted:
                    output = extracted
                    code = "(No code generated - direct text response)"
                    break

            # If we couldn't extract anything, retry or give up
            if attempt < max_attempts - 1:
                # Add the message to history for context on next attempt
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content or "(No response)"
                })
                messages.append({
                    "role": "user",
                    "content": "Please try again. Generate clear results with each on a separate line in format: '<var_name> <value> <unit>'"
                })
                continue
            else:
                # Last attempt failed - raise error
                raise ValueError("Could not generate valid output from model after 3 attempts")

    if not output:
        raise ValueError("No output generated from code execution")

    # Parse output: expect multiple lines of "var_name value unit" format
    values_dict, units_dict = _parse_multi_output(output, outputs)

    # Validate that all expected variables were parsed
    missing_vars = [v for v in outputs if values_dict.get(v) is None]
    if missing_vars:
        raise ValueError(f"Failed to extract: {missing_vars}. Output:\n{output}")

    return values_dict, units_dict, code, total_cost


def _extract_result_from_text(text: str) -> Optional[str]:
    """
    Extract a result from plain text response.
    Looks for patterns like "M kg", "10.0 m/s", etc.
    Assumes the first meaningful line or last line contains the result.

    Returns:
        Extracted result string, or None if no result found
    """
    lines = text.strip().split('\n')

    # Try to find a line that looks like "value unit"
    for line in lines:
        line = line.strip()
        # Skip empty lines and lines that are too short
        if not line or len(line) < 1:
            continue
        # CRITICAL: Skip code fences and markdown markers
        if line.startswith('```') or line.startswith('~~~'):
            continue
        # Skip lines that look like explanations
        if line.lower().startswith(('the ', 'this ', 'here', 'example', 'note')):
            continue
        if line.endswith(':'):
            continue
        # This line might be our answer - check if it has word-like content
        if line and not line.startswith('{') and not line.startswith('['):
            # Found something that looks like a result
            return line

    # If we didn't find anything obvious, return the last non-empty line
    # But still skip code fences
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith('```') and not line.startswith('~~~'):
            return line

    return None


def _parse_output(output: str) -> Tuple[Union[float, str], str]:
    """
    Parse execution output to extract value and unit.
    Handles both numeric and symbolic outputs.

    Expected formats:
    - Numeric: "10.0 m/s" → (10.0, "m/s")
    - Symbolic: "M*u/t m/s" → ("M*u/t", "m/s")

    Raises:
        ValueError: If output is malformed or contains code markers
    """

    output = output.strip()

    # CRITICAL: Reject malformed outputs containing code fences
    if output.startswith('```') or output.startswith('~~~'):
        raise ValueError(f"Output is malformed code block: {output[:50]}")

    if not output:
        raise ValueError("Output is empty")

    parts = output.split(None, 1)  # Split on first whitespace

    if len(parts) < 2:
        # Try to parse as just a number
        try:
            value = float(parts[0])
            return value, "dimensionless"
        except ValueError:
            # Not a number, treat as symbolic
            # But reject if it looks like a language tag
            if parts[0] in ('python', 'java', 'c', 'cpp', 'rust', 'go', 'javascript'):
                raise ValueError(f"Output appears to be code language tag: {parts[0]}")
            return parts[0], "dimensionless"

    first_token = parts[0]
    unit = parts[1].strip()

    # Validate unit is not a language marker or malformed
    if unit in ('python', 'java', 'c', 'cpp', 'rust') or unit.startswith('```'):
        raise ValueError(f"Malformed unit: {unit}")

    try:
        # Try to parse as numeric
        value = float(first_token)
        return value, unit
    except ValueError:
        # Not a number, treat as symbolic expression
        # Validate it's not malformed
        if first_token.startswith('```') or first_token in ('python', 'import', 'def'):
            raise ValueError(f"Output appears to be code, not expression: {first_token}")
        return first_token, unit


def _validate_result(value: Union[float, str], unit: str, step: Step, state: StateObject) -> Tuple[bool, Optional[str]]:
    """
    Validate that a result meets ESSENTIAL semantic requirements.

    We validate minimally - just the critical failures. Let the math layer
    catch dimensional and semantic errors. Trust the tool mechanism.

    Args:
        value: The computed value (float or symbolic expression)
        unit: The unit of the value
        step: The step that was executed
        state: Current state

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if result is acceptable
        - error_message: Description of critical error (None if valid)
    """

    # ESSENTIAL CHECK 1: Value should not be None
    if value is None:
        return False, "Result is None (extraction failed)"

    # ESSENTIAL CHECK 2: Value should not be empty string
    if isinstance(value, str) and not value.strip():
        return False, "Result is empty string"

    # ESSENTIAL CHECK 3: Reject obvious code artifacts
    if isinstance(value, str):
        if value.startswith(('```', '~~~', 'def ', 'import ', 'class ', 'print')):
            return False, f"Result is code, not expression: {value[:40]}"

    # ESSENTIAL CHECK 4: Check unit is not obviously malformed
    if not unit or unit.lower() in ('none', 'null', ''):
        return False, "Unit is missing or null"

    # Everything else is OK - let the downstream math/validation layers catch issues
    # A unit mismatch doesn't mean the result is wrong - it might just need conversion
    # A dimensionless result might be intentional - let the calculation layer validate

    return True, None


def _parse_multi_output(output: str, var_names: List[str]) -> Tuple[dict, dict]:
    """
    Parse execution output for multiple variables.
    Handles batch extraction output with multiple lines.

    Expected format (multiple lines):
    var1 value1 unit1
    var2 value2 unit2
    var3 value3 unit3

    Returns:
        Tuple of (values_dict, units_dict)
        - values_dict: {var_name: value, ...} where value is float or str
        - units_dict: {var_name: unit, ...}
    """
    values_dict = {}
    units_dict = {}

    lines = output.strip().split('\n')

    # Track what we actually parsed for debugging
    parsed_vars = set()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip code markers
        if line.startswith('```') or line.startswith('~~~'):
            continue

        # Parse line: "var_name value unit"
        parts = line.split(None, 2)  # Split into max 3 parts

        if len(parts) < 2:
            # Line doesn't have enough parts - skip it
            continue

        var_name = parts[0]

        # Check if this variable is in our expected list
        if var_name not in var_names:
            # This line is for a variable we don't care about, skip it
            continue

        parsed_vars.add(var_name)

        if len(parts) == 2:
            # No unit provided
            value_str = parts[1]
            unit = "dimensionless"
        else:
            # Unit is provided
            value_str = parts[1]
            unit = parts[2]

        # Try to parse value as float, otherwise treat as string (symbolic)
        try:
            value = float(value_str)
        except ValueError:
            # Not a number, treat as symbolic expression
            value = value_str

        values_dict[var_name] = value
        units_dict[var_name] = unit

    # Ensure all expected variables are in the dicts
    for var_name in var_names:
        if var_name not in values_dict:
            values_dict[var_name] = None
        if var_name not in units_dict:
            units_dict[var_name] = "unknown"

    return values_dict, units_dict


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