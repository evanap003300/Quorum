"""LLM execution functions for step processing."""

import asyncio
import importlib.util
import json
import os
import sys
from typing import List, Optional, Tuple, Union

from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from planner.schema import Step, StateObject
from prompts.solver import (
    SOLVER_SYSTEM_MESSAGE,
    SOLVER_MULTI_OUTPUT_SYSTEM_MESSAGE,
    build_extract_prompt,
    build_calculate_prompt,
    build_convert_prompt,
)
from solver.parsing import (
    calculate_cost,
    extract_result_from_text,
    parse_output,
    parse_multi_output,
    parse_json_output,
)

try:
    from e2b_code_interpreter import Sandbox
except ImportError:
    Sandbox = None

# Import python interpreter from tools
from tools.evaluation.code_interpreter import run as _run_python_impl

load_dotenv()

# Prefer direct OpenAI API for better latency, fall back to OpenRouter
if os.getenv("OPENAI_API_KEY"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    _using_direct_api = True
else:
    # Fallback to OpenRouter
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_KEY")
    )
    async_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_KEY")
    )
    _using_direct_api = False

# Python interpreter tool definition
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


def execute_with_llm(prompt: str, sandbox: Optional["Sandbox"] = None) -> Tuple[Union[float, str], str, str, float]:
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
            "content": SOLVER_SYSTEM_MESSAGE
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
        total_cost += calculate_cost(completion, model)

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
                extracted = extract_result_from_text(assistant_message.content)
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
    value, unit = parse_output(output)

    return value, unit, code, total_cost


def execute_with_llm_multi_output(prompt: str, outputs: List[str], step: Step, state: StateObject, sandbox: Optional["Sandbox"] = None) -> Tuple[dict, dict, str, float]:
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
            "content": SOLVER_MULTI_OUTPUT_SYSTEM_MESSAGE
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
        total_cost += calculate_cost(completion, model)

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
                extracted = extract_result_from_text(assistant_message.content)
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

    # Parse output: try JSON first, then fall back to multi-line format
    values_dict = None
    units_dict = None
    parse_error = None

    # Try JSON parsing first (for new multi-output calculations)
    if output.strip().startswith('{') and output.strip().endswith('}'):
        try:
            values_dict, units_dict = parse_json_output(output)
        except Exception as e:
            parse_error = f"JSON parsing failed: {str(e)}"

    # Fall back to multi-line parsing if JSON failed or wasn't JSON
    if values_dict is None:
        try:
            values_dict, units_dict = parse_multi_output(output, outputs)
        except Exception as e:
            if parse_error:
                parse_error += f"; Multi-line parsing also failed: {str(e)}"
            else:
                parse_error = f"Multi-line parsing failed: {str(e)}"

    if values_dict is None:
        raise ValueError(f"Could not parse output. {parse_error}\nRaw output:\n{output}")

    # Validate that all expected variables were parsed
    missing_vars = [v for v in outputs if values_dict.get(v) is None]
    if missing_vars:
        raise ValueError(f"Failed to extract: {missing_vars}. Output:\n{output}")

    return values_dict, units_dict, code, total_cost


# ============================================================================
# ASYNC VERSIONS FOR PARALLEL EXECUTION (K-Ahead Swarm)
# ============================================================================


async def execute_with_llm_async(prompt: str, sandbox: Optional["Sandbox"] = None) -> Tuple[Union[float, str], str, str, float]:
    """
    Async version of execute_with_llm for parallel execution in K-Ahead Swarm.

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
            "content": SOLVER_SYSTEM_MESSAGE
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
        completion = await async_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            temperature=0.1  # Slight randomness helps avoid degenerate code generation
        )

        # Track cost
        total_cost += calculate_cost(completion, model)

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
                    output = await run_python(code, sandbox)

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
                extracted = extract_result_from_text(assistant_message.content)
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
    value, unit = parse_output(output)

    return value, unit, code, total_cost


async def execute_with_llm_multi_output_async(prompt: str, outputs: List[str], step: Step, state: StateObject, sandbox: Optional["Sandbox"] = None) -> Tuple[dict, dict, str, float]:
    """
    Async version of execute_with_llm_multi_output for parallel execution in K-Ahead Swarm.

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
            "content": SOLVER_MULTI_OUTPUT_SYSTEM_MESSAGE
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
        completion = await async_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            temperature=0.1  # Slight randomness helps avoid degenerate code generation
        )

        # Track cost
        total_cost += calculate_cost(completion, model)

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
                    output = await run_python(code, sandbox)

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
                extracted = extract_result_from_text(assistant_message.content)
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

    # Parse output: try JSON first, then fall back to multi-line format
    values_dict = None
    units_dict = None
    parse_error = None

    # Try JSON parsing first (for new multi-output calculations)
    if output.strip().startswith('{') and output.strip().endswith('}'):
        try:
            values_dict, units_dict = parse_json_output(output)
        except Exception as e:
            parse_error = f"JSON parsing failed: {str(e)}"

    # Fall back to multi-line parsing if JSON failed or wasn't JSON
    if values_dict is None:
        try:
            values_dict, units_dict = parse_multi_output(output, outputs)
        except Exception as e:
            if parse_error:
                parse_error += f"; Multi-line parsing also failed: {str(e)}"
            else:
                parse_error = f"Multi-line parsing failed: {str(e)}"

    if values_dict is None:
        raise ValueError(f"Could not parse output. {parse_error}\nRaw output:\n{output}")

    # Validate that all expected variables were parsed
    missing_vars = [v for v in outputs if values_dict.get(v) is None]
    if missing_vars:
        raise ValueError(f"Failed to extract: {missing_vars}. Output:\n{output}")

    return values_dict, units_dict, code, total_cost
