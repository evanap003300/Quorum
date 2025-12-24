"""Solver module prompts for step execution."""

from planner.schema import Step, StateObject

# System messages for LLM executors
SOLVER_SYSTEM_MESSAGE = "You are a code generator for physics calculations. Generate clean Python code and use the python_interpreter tool to execute it."

SOLVER_MULTI_OUTPUT_SYSTEM_MESSAGE = "You are a code generator for physics calculations. Generate clean Python code and use the python_interpreter tool to execute it."


def build_extract_prompt(step: Step, state: StateObject) -> str:
    """Build prompt for extract operations"""
    outputs = step.get_outputs()

    # Handle multi-output extraction
    if len(outputs) > 1:
        # Build prompt for extracting multiple variables
        variables_section = ""
        for var_name in outputs:
            var = state.variables[var_name]
            unit = step.get_unit(var_name)
            # Include extraction hint if available
            hint_text = f" - LOCATION: {var.extraction_hint}" if var.extraction_hint else ""
            variables_section += f"- {var_name}: {var.description} (unit: {unit}){hint_text}\n"

        base_prompt = f"""EXTRACT ALL THESE VARIABLES:
{variables_section}

PROBLEM STATEMENT:
{state.problem_text}

WRITE PYTHON CODE that:
1. Reads the problem above carefully
2. For EACH variable, find its ACTUAL VALUE in the problem text (not the variable name)
3. Use the LOCATION hint if provided to find the exact value
4. PRINTS each variable on a separate line in EXACTLY this format:

   VAR_NAME VALUE UNIT

CRITICAL RULES:
✗ DO NOT use the variable name as the VALUE (e.g., "T_ref T_ref C" is WRONG)
✓ DO extract the actual number from the problem (e.g., "T_ref 72 C" is correct)

You must extract ALL {len(outputs)} variables: {", ".join(outputs)}

OUTPUT EXAMPLES (for numeric values from a table):
T_ref 72 C
T_x_plus 74 C
T_x_minus 70 C

OUTPUT EXAMPLES (for values directly stated in text):
m 5.0 kg
g 9.81 m/s**2

RULES:
- Extract NUMERIC VALUES from tables, grids, or stated values
- Use LOCATION hints to find the correct row/column in tables
- For symbolic variables, use the variable name as the VALUE
- Print EVERY variable, even if they appear later in the problem
- Each variable must be on its own line
- NO variable should have its name as both VAR_NAME and VALUE"""

        if step.is_symbolic:
            symbolic_section = f"""
THESE ARE SYMBOLIC VARIABLES:
- Extract the variable symbol (e.g., m, M, R, F) from the problem text
- The VALUE should be the symbol name itself
- Include the unit from the problem context"""
            return base_prompt + symbolic_section
        else:
            return base_prompt

    # Single variable extraction (original logic)
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
- Format: print(f"{str(sp.simplify(result))} {target_unit}")
"""
        return base_prompt + symbolic_section
    else:
        return base_prompt


def build_observe_prompt(step: Step, state: StateObject) -> str:
    """Build prompt for observe operations - reads specific values from diagram"""

    outputs = step.get_outputs()

    # Handle multi-output observation
    if len(outputs) > 1:
        variables_section = ""
        for var_name in outputs:
            var = state.variables[var_name]
            unit = step.get_unit(var_name)
            variables_section += f"  - {var_name}: {var.description} (unit: {unit})\n"

        return f"""READ THESE VALUES FROM THE DIAGRAM:

{variables_section}

TASK:
{step.description}

INSTRUCTIONS:
1. Use computer vision tools if needed (crop regions, enhance clarity, apply grid for reference)
2. Locate each variable in the diagram (table, graph, label, etc.)
3. Read the exact value and unit
4. Return the numeric or symbolic value you find

RESPONSE FORMAT (STRICT JSON):
Return ONLY valid JSON with this exact structure:

{{
  "var_name_1": "value_with_unit",
  "var_name_2": "value_with_unit"
}}

CRITICAL:
- Return ONLY the JSON object, no other text
- Include units in the value: "15 m/s", "3.2", "M kg", etc.
- If symbolic (variable letter), return the letter: "M", "L", "θ"
- All quotes and JSON must be valid

Example response:
{{
  "T_ref": "72 C",
  "h": "5 m"
}}
"""

    # Single variable observation
    var = state.variables[outputs[0]]
    unit = step.get_unit(outputs[0])

    return f"""READ THIS VALUE FROM THE DIAGRAM:

{var.description} (unit: {unit})

TASK:
{step.description}

INSTRUCTIONS:
1. Use computer vision tools if needed (crop regions, enhance clarity, apply grid for reference)
2. Locate the value in the diagram (table, graph, label, measurement, etc.)
3. Read the exact value and unit
4. Return what you find

RESPONSE FORMAT (STRICT JSON):
Return ONLY valid JSON with this exact structure:

{{
  "value": "measurement_with_unit"
}}

CRITICAL:
- Return ONLY the JSON object, no other text
- Include units: "15 m/s", "72 C", "3.2 kg", "M", "θ", etc.
- If it's a symbolic variable (single letter), just return the letter
- If it's a number, include the unit
- All quotes and JSON must be valid

Example responses:
{{
  "value": "15 m/s"
}}

{{
  "value": "M"
}}
"""
