"""Main solver entry point for executing atomic problem-solving steps."""

import os
import sys
from typing import List, Optional, Tuple, Union

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from planner.schema import Step, StateObject
from prompts.solver import (
    build_extract_prompt,
    build_calculate_prompt,
    build_convert_prompt,
    build_observe_prompt,
)
from solver.execution import (
    execute_with_llm,
    execute_with_llm_multi_output,
    execute_with_llm_async,
    execute_with_llm_multi_output_async,
)
from solver.parsing import validate_result
from vision import analyze_observation, analyze_observation_multi

try:
    from e2b_code_interpreter import Sandbox
except ImportError:
    Sandbox = None


async def solve_step(step: Step, state: StateObject, sandbox: Optional["Sandbox"] = None, error_context: Optional[str] = None) -> Tuple[bool, Optional[Union[float, str, dict]], Optional[Union[str, dict]], Optional[str], float]:
    """
    Execute a single atomic step.

    Args:
        step: The step to execute
        state: Current problem state with variable values
        sandbox: Optional existing sandbox to reuse. If None, creates new sandbox per execution.
        error_context: Optional error message from a previous failed attempt. If provided, will be prepended to the prompt to help agent correct the error.

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
            prompt = build_extract_prompt(step, state)
        elif step.operation == "calculate":
            prompt = build_calculate_prompt(step, state)
        elif step.operation == "convert":
            prompt = build_convert_prompt(step, state)
        elif step.operation == "observe":
            prompt = build_observe_prompt(step, state)
        else:
            return False, None, None, f"Unknown operation: {step.operation}", 0.0

        # If retrying after an error, prepend error context to the prompt
        if error_context:
            prompt = f"PREVIOUS ATTEMPT FAILED WITH THIS ERROR:\n{error_context}\n\nPlease fix the error and try again:\n\n{prompt}"

        # Check if this is a multi-output extraction step
        outputs = step.get_outputs()
        is_multi_output = len(outputs) > 1

        # SPECIAL HANDLING FOR OBSERVE OPERATIONS
        if step.operation == "observe":
            # OBSERVE operations use vision model, not code interpreter
            # Need image_path from context (passed from orchestrator)
            image_path = None
            if hasattr(state, "problem_context") and state.problem_context:
                image_path = state.problem_context.get("image_path")

            if not image_path:
                return False, None, None, "OBSERVE operation requires image_path in state.problem_context", 0.0

            try:
                if is_multi_output:
                    values, units, cost = analyze_observation_multi(image_path, prompt, outputs, step, state)
                    # Debug: check if we got valid dicts back
                    print(f"DEBUG OBSERVE multi-output: values type={type(values)}, units type={type(units)}")
                    print(f"DEBUG OBSERVE multi-output: values={values}, units={units}")
                else:
                    value, unit, cost = analyze_observation(image_path, prompt)
                    values, units = value, unit

                return True, values, units, None, cost
            except Exception as e:
                import traceback
                print(f"DEBUG OBSERVE exception: {type(e).__name__}: {e}")
                traceback.print_exc()
                return False, None, None, f"OBSERVE operation failed: {type(e).__name__}: {str(e)}", 0.0

        # NORMAL EXECUTION FOR EXTRACT/CALCULATE/CONVERT
        # Batch multi-output for extraction, calculation, and convert steps
        if is_multi_output and step.operation in ["extract", "calculate", "convert"]:
            values, units, code, cost = await execute_with_llm_multi_output_async(prompt, outputs, step, state, sandbox)
        else:
            value, unit, code, cost = await execute_with_llm_async(prompt, sandbox)
            values, units = value, unit

        return True, values, units, None, cost

    except Exception as e:
        return False, None, None, str(e), 0.0
