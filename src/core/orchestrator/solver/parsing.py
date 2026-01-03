"""Output parsing, validation, and cost calculation functions."""

from typing import List, Optional, Tuple, Union

from planner.schema import Step, StateObject
from config.pricing import MODEL_PRICING


def _safe_parse_float(text: str) -> Optional[float]:
    """Safely parse float, returning None for invalid inputs.

    Handles edge cases like '.', '-', '+', 'None', empty strings that
    would cause float() to raise ValueError.

    Args:
        text: String to parse as float

    Returns:
        Float value if parsing succeeds, None otherwise
    """
    if not text or text.strip() in ('.', '-', '+', 'None', 'null', '', 'nan', 'NaN'):
        return None
    try:
        return float(text.strip())
    except (ValueError, TypeError):
        return None


def calculate_cost(completion, model: str) -> float:
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


def extract_result_from_text(text: str) -> Optional[str]:
    """
    Extract a result from plain text response.
    Looks for patterns like "M kg", "10.0 m/s", JSON objects, etc.
    Assumes the first meaningful line or last line contains the result.

    Returns:
        Extracted result string, or None if no result found
    """
    lines = text.strip().split('\n')

    # First, look for JSON objects (for multi-output calculations)
    for line in lines:
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            # This looks like a JSON object
            return line

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


def is_execution_error(output: str) -> bool:
    """
    Check if output is an execution error message rather than a valid result.

    Args:
        output: The output string to check

    Returns:
        True if output appears to be an error message
    """
    if not output:
        return False

    output_lower = output.lower()

    # Python exception patterns
    error_patterns = [
        'syntaxerror',
        'indentationerror',
        'nameerror',
        'typeerror',
        'valueerror',
        'keyerror',
        'indexerror',
        'attributeerror',
        'zerodivisionerror',
        'runtimeerror',
        'error:',
        'traceback (most recent call last)',
        'file "<string>"',
        'problematic line:',
    ]

    for pattern in error_patterns:
        if pattern in output_lower:
            return True

    return False


def parse_output(output: str) -> Tuple[Union[float, str], str]:
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

    # CRITICAL: Check if output is an error message
    if is_execution_error(output):
        raise ValueError(f"Code execution error: {output[:200]}")

    # CRITICAL: Reject malformed outputs containing code fences
    if output.startswith('```') or output.startswith('~~~'):
        raise ValueError(f"Output is malformed code block: {output[:50]}")

    if not output:
        raise ValueError("Output is empty")

    parts = output.split(None, 1)  # Split on first whitespace

    if len(parts) < 2:
        # Try to parse as just a number
        value = _safe_parse_float(parts[0])
        if value is not None:
            return value, "dimensionless"
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

    # Try to parse as numeric using safe parser
    value = _safe_parse_float(first_token)
    if value is not None:
        return value, unit
    # Not a number, treat as symbolic expression
    # Validate it's not malformed
    if first_token.startswith('```') or first_token in ('python', 'import', 'def'):
        raise ValueError(f"Output appears to be code, not expression: {first_token}")
    return first_token, unit


def validate_result(value: Union[float, str], unit: str, step: Step, state: StateObject) -> Tuple[bool, Optional[str]]:
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
    # A value that matches variable name might be intentional (symbolic/given as name)

    return True, None


def parse_json_output(output: str) -> Tuple[dict, dict]:
    """
    Parse JSON output from multi-output calculations.

    Expected format:
    {"var1": "10.0 m/s", "var2": "20.0 kg", ...}

    Also handles Python dict format (single quotes) as fallback.

    Returns:
        Tuple of (values_dict, units_dict)
        - values_dict: {var_name: value, ...} where value is float or str
        - units_dict: {var_name: unit, ...}

    Raises:
        ValueError: If output is not valid JSON or is an execution error
    """
    import json

    output = output.strip()

    # CRITICAL: Check if output is an error message
    if is_execution_error(output):
        raise ValueError(f"Code execution error: {output[:200]}")

    # Parse JSON
    try:
        data = json.loads(output)
    except json.JSONDecodeError as e:
        # Fallback: try to convert Python dict format (single quotes) to JSON (double quotes)
        try:
            # Replace single quotes with double quotes for Python dict -> JSON conversion
            json_str = output.replace("'", '"')
            data = json.loads(json_str)
        except json.JSONDecodeError:
            raise ValueError(f"Output is not valid JSON: {str(e)}")

    if not isinstance(data, dict):
        raise ValueError(f"JSON output must be an object, got {type(data).__name__}")

    values_dict = {}
    units_dict = {}

    for var_name, value_str in data.items():
        # Parse "value unit" format from each entry
        value_str = str(value_str).strip()

        # Split on last space to separate value and unit
        parts = value_str.rsplit(None, 1)

        if len(parts) == 2:
            value_text, unit = parts
        else:
            value_text = parts[0] if parts else ""
            unit = "dimensionless"

        # Try to parse value as float using safe parser, otherwise treat as symbolic
        value = _safe_parse_float(value_text)
        if value is None:
            value = value_text  # Keep as string for symbolic values

        values_dict[var_name] = value
        units_dict[var_name] = unit

    return values_dict, units_dict


def parse_multi_output(output: str, var_names: List[str]) -> Tuple[dict, dict]:
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

    Raises:
        ValueError: If output is an execution error
    """
    # CRITICAL: Check if output is an error message
    if is_execution_error(output):
        raise ValueError(f"Code execution error: {output[:200]}")

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

        # Try to parse value as float using safe parser, otherwise treat as string (symbolic)
        value = _safe_parse_float(value_str)
        if value is None:
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
