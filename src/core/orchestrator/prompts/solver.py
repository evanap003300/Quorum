"""Solver module prompts for step execution."""

from planner.schema import Step, StateObject

# System messages for LLM executors
SOLVER_SYSTEM_MESSAGE = """You are a code generator for physics calculations. Generate clean Python code and use the python_interpreter tool to execute it.

IMPORTANT RULES:
1. Output results in the format: <value> <unit>
2. For angles, ALWAYS output in DEGREES unless the problem explicitly specifies radians or another unit.
3. Convert radian results to degrees before outputting: degrees = radians * (180 / π)
"""

SOLVER_MULTI_OUTPUT_SYSTEM_MESSAGE = """You are a code generator for physics calculations. Generate clean Python code and use the python_interpreter tool to execute it.

IMPORTANT RULES:
1. Output results in the format: <value> <unit>
2. For angles, ALWAYS output in DEGREES unless the problem explicitly specifies radians or another unit.
3. Convert radian results to degrees before outputting: degrees = radians * (180 / π)
"""


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

PYTHON CODE TEMPLATE:
```python
# Extract each variable and print in VAR_NAME VALUE UNIT format
# Use regex or text parsing to find values from problem statement

print("{outputs[0]} VALUE1 UNIT1")
print("{outputs[1]} VALUE2 UNIT2")
# ... (one print per variable, all on separate lines)
```

CRITICAL RULES:
1. Extract ACTUAL VALUES from the problem text, not variable names
2. Print each variable on its own line
3. Format: VAR_NAME VALUE UNIT (space-separated)
4. Extract ALL {len(outputs)} variables: {", ".join(outputs)}
5. Do NOT print explanations, only variable lines

EXAMPLES:
P 1.0 atm
T 298.0 K

OR:

theta_deg 104.5 degree
d_pm 95.7 pm

IMPORTANT:
- Use LOCATION hints if provided to find exact values
- For symbolic variables, the VALUE is the symbol name itself
- Every variable must produce exactly one line of output
- Output format is critical: VAR_NAME VALUE UNIT"""

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

    outputs = step.get_outputs()

    # Handle multi-output calculations
    if len(outputs) > 1:
        outputs_section = ""
        for var_name in outputs:
            var = state.variables[var_name]
            unit = step.get_unit(var_name)
            outputs_section += f"  - {var_name}: {var.description} (unit: {unit})\n"

        base_prompt = f"""Calculate {step.description}:

Inputs:
{inputs_text}

Formula: {step.formula}

CALCULATE ALL {len(outputs)} OUTPUTS:
{outputs_section}

CRITICAL: Output VALID JSON using json.dumps() - NOT Python dict notation.

PYTHON CODE TEMPLATE:
```python
import json
from scipy import constants as sci_constants

# Your calculation code here
result = {{
    "var1": "value1 unit1",
    "var2": "value2 unit2"
}}
print(json.dumps(result))
```

INSTRUCTIONS:
1. Define a Python dict with all {len(outputs)} outputs
2. Each value must be formatted as "number unit" (include unit with value)
3. Use json.dumps(result) to convert to valid JSON
4. Print ONLY the JSON output (one line)

EXAMPLE:
result = {{
    "m_empty": "8.0 kg",
    "m_after": "98.0 kg"
}}
print(json.dumps(result))

RULES:
- Define result as a dict with string values
- Use json.dumps() to ensure valid JSON output
- Each value includes: number + space + unit
- Use scientific notation for very large/small numbers
- Use sci_constants for physical constants
- Do NOT print variable assignments, only the JSON result"""

        if step.is_symbolic:
            symbolic_section = f"""

SYMBOLIC CALCULATION:
- Define symbolic variables: sp.symbols('L V u ...', positive=True, real=True)
- Apply formula and simplify: sp.simplify(result)
- CRITICAL: Preserve units! Each result must have appropriate units
- Print JSON with symbolic expressions: {{"var1": "expr1 unit1", ...}}
"""
            return base_prompt + symbolic_section
        else:
            return base_prompt

    # Single variable calculation (original logic)
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

        return f"""EXTRACT THESE VALUES FROM THE DIAGRAM:

TASK:
{step.description}

VARIABLES TO EXTRACT:
{variables_section}

CRITICAL INSTRUCTIONS:
1. Use computer vision tools to locate the correct regions in the diagram
   - Use grid overlay if helpful to identify table cells
   - Crop and enhance regions to read values clearly
   - If values are hard to read, use binarize_image or enhance_clarity
2. Find EACH variable in the table/diagram exactly as described
3. Return ALL {len(outputs)} variables in the response

RESPONSE FORMAT (PLAIN TEXT - one variable per line):
For each variable, respond with EXACTLY this format on a separate line:

VAR_NAME VALUE UNIT

Examples:
x0 4 dimensionless
y0 3 dimensionless
T0 72 C
T_x1 74 C
T_y1 70 C

CRITICAL RULES:
- Respond with ONLY plain text, no JSON or markdown
- One variable per line in format: VAR_NAME VALUE UNIT
- Separate VAR_NAME, VALUE, and UNIT with single spaces
- Include the unit: "4 dimensionless", "72 C", "74 deg", "M kg", etc.
- Every variable must be included, in the same order as listed above
- If you cannot find a value, use "unknown dimensionless"
- Do not skip any variables
- Do not add extra explanations
"""

    # Single variable observation
    var = state.variables[outputs[0]]
    unit = step.get_unit(outputs[0])

    return f"""READ THIS VALUE FROM THE DIAGRAM:

Variable: {outputs[0]}
Description: {var.description}
Expected unit: {unit}

TASK:
{step.description}

INSTRUCTIONS:
1. Use computer vision tools if needed (crop regions, enhance clarity, apply grid for reference)
2. Locate the value in the diagram (table, graph, label, measurement, etc.)
3. Read the exact value and unit
4. Return what you find

RESPONSE FORMAT (PLAIN TEXT):
Return ONLY the value with unit on a single line:

VALUE UNIT

Examples:
15 m/s
72 C
3.2 kg
M dimensionless
θ dimensionless

CRITICAL:
- Respond with ONLY plain text, no JSON or markdown
- Format: VALUE UNIT (separated by a single space)
- Include the unit: "15 m/s", "72 C", "3.2 kg", "M dimensionless", etc.
- If it's a symbolic variable (single letter), return: LETTER dimensionless
- Do not add explanations or extra text
"""
