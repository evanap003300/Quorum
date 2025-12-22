"""Solver module prompts for step execution."""

from planner.schema import Step, StateObject

# System messages for LLM executors
SOLVER_SYSTEM_MESSAGE = "You are a code generator for physics calculations. Generate clean Python code and use the python_interpreter tool to execute it."

SOLVER_MULTI_OUTPUT_SYSTEM_MESSAGE = "You are a code generator for physics calculations. Generate clean Python code that extracts multiple values and prints them in the required format. Use the python_interpreter tool to execute it."


def build_extract_prompt(step: Step, state: StateObject) -> str:
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
            return base_prompt
        else:
            return base_prompt


def build_calculate_prompt(step: Step, state: StateObject) -> str:
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


def build_convert_prompt(step: Step, state: StateObject) -> str:
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
