"""Answer comparison logic for benchmark evaluation."""

from pydantic import BaseModel
from typing import Union, Optional, Tuple
from .unit_converter import UnitConverter
import math
import re

try:
    import sympy
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


def extract_number(answer: Union[float, str]) -> Tuple[Optional[float], str]:
    """
    Robustly extract a numeric value from various answer formats.

    Uses cascading strategies to handle different formats and returns both the
    extracted number and the method used for extraction (useful for debugging).

    Args:
        answer: The answer to extract a number from (float, int, or string)

    Returns:
        Tuple of (numeric_value, extraction_method)
        - numeric_value: float or None if all extraction methods failed
        - extraction_method: string describing which method succeeded
    """
    attempts = []

    # Strategy 1: Direct numeric check
    if isinstance(answer, (int, float)):
        if math.isnan(answer) or math.isinf(answer):
            return None, "invalid_numeric_value"
        return float(answer), "direct_numeric"

    if not isinstance(answer, str):
        return None, "not_convertible"

    text = answer.strip()

    if not text:
        return None, "empty_string"

    # Strategy 2: Sympy evaluation (catches symbolic math expressions)
    if SYMPY_AVAILABLE:
        try:
            expr = sympy.sympify(text)
            result = float(expr.evalf())
            if not (math.isnan(result) or math.isinf(result)):
                return result, "sympy_evaluation"
        except Exception:
            attempts.append("sympy_failed")

    # Strategy 3: Standard float parsing (handles scientific notation)
    try:
        result = float(text)
        if not (math.isnan(result) or math.isinf(result)):
            return result, "standard_float"
    except ValueError:
        attempts.append("standard_float_failed")

    # Strategy 4: Remove commas and underscores, then parse
    cleaned = text.replace(",", "").replace("_", "")
    if cleaned != text:  # Only try if we actually removed something
        try:
            result = float(cleaned)
            if not (math.isnan(result) or math.isinf(result)):
                return result, "cleaned_numeric"
        except ValueError:
            attempts.append("cleaned_numeric_failed")

    # Strategy 5: Percentage handling (e.g., "15%" -> 0.15) - BEFORE regex to catch percentages
    if "%" in text:
        try:
            percent_text = text.replace("%", "").strip()
            # Try to find a number in the percentage text
            percent_match = re.search(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', percent_text)
            if percent_match:
                result = float(percent_match.group()) / 100.0
                if not (math.isnan(result) or math.isinf(result)):
                    return result, "percentage"
        except Exception:
            attempts.append("percentage_failed")

    # Strategy 5.5: Scientific notation with LaTeX/Unicode exponents
    # Handles formats like: "1.91 $10^{-47}$", "1.91 × 10^-47", "1.91 \times 10^{-47}", "7.9 x 10^34"
    latex_sci_notation_patterns = [
        # LaTeX format with optional content after exponent: "1.91 $10^{-47}$" or "1.91 $10^{-47} \mathrm{...}$"
        r'(-?\d+\.?\d*)\s*\$10\^\{?(-?\d+)\}?',
        # Multiplication sign variants: "1.91 × 10^-47", "7.9 x 10^{34}", "1.91 * 10^34"
        # Note: handles optional braces around exponent
        r'(-?\d+\.?\d*)\s*[×\*xX]\s*10\^\s*\{?(-?\d+)\}?',
        # LaTeX \times: "1.91 \times 10^-47" or "1.91 \times 10^{-47}"
        # Note: handles optional braces around exponent
        r'(-?\d+\.?\d*)\s*\\times\s+10\^\s*\{?(-?\d+)\}?',
        # Standard format with power: "1.91 * 10**-47" or "1.91e-47" (already handled)
    ]

    for pattern in latex_sci_notation_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                base = float(match.group(1))
                exponent = int(match.group(2))
                result = base * (10 ** exponent)
                if not (math.isnan(result) or math.isinf(result)):
                    return result, "latex_scientific_notation"
            except (ValueError, AttributeError):
                attempts.append("latex_sci_notation_failed")

    # Strategy 5.6: Try first token extraction (handles "10 m/s" -> extract "10", not the "/" part)
    # Split by whitespace and try to parse the first token
    tokens = text.split()
    if tokens:
        first_token = tokens[0]
        try:
            result = float(first_token)
            if not (math.isnan(result) or math.isinf(result)):
                return result, "first_token_extraction"
        except ValueError:
            attempts.append("first_token_extraction_failed")

    # Strategy 6: Regex-based extraction (find all numbers in first token or full text)
    # Pattern matches: integers, decimals, and scientific notation
    number_pattern = r'-?\d+\.?\d*(?:[eE][+-]?\d+)?'

    # First try to find all numbers in just the first token (avoids picking up unit numbers)
    if tokens:
        first_token_matches = re.findall(number_pattern, tokens[0])
        if first_token_matches:
            try:
                # Take the last number in the first token (for cases like "-5.5e-10")
                result = float(first_token_matches[-1])
                if not (math.isnan(result) or math.isinf(result)):
                    return result, "regex_first_token"
            except ValueError:
                attempts.append("regex_first_token_failed")

    # Fall back to finding all numbers in the entire text
    matches = re.findall(number_pattern, text)
    if matches:
        try:
            # Take the first match (more likely to be the answer than the last in complex units)
            result = float(matches[0])
            if not (math.isnan(result) or math.isinf(result)):
                return result, "regex_extraction"
        except ValueError:
            attempts.append("regex_extraction_failed")

    # Strategy 7: Fraction evaluation (e.g., "1/2" -> 0.5)
    if SYMPY_AVAILABLE and "/" in text:
        try:
            # Check if this looks like a fraction
            parts = text.split()
            for part in parts:
                if "/" in part and not any(c in part for c in "()[]{}"):
                    try:
                        expr = sympy.sympify(part)
                        result = float(expr.evalf())
                        if not (math.isnan(result) or math.isinf(result)):
                            return result, "fraction_evaluation"
                    except Exception:
                        pass
        except Exception:
            attempts.append("fraction_evaluation_failed")

    # All extraction methods failed
    return None, f"all_failed: {'; '.join(attempts)}"


def evaluate_symbolic_expression(expr_str: str) -> Optional[float]:
    """Try to evaluate a symbolic expression to a numeric value.

    Args:
        expr_str: String representation of expression (e.g., "acos(2/3)")

    Returns:
        Numeric value if evaluation succeeds, None otherwise
    """
    if not SYMPY_AVAILABLE:
        return None

    try:
        # Parse and evaluate the expression
        expr = sympy.sympify(expr_str)
        # Evaluate to a floating point number
        result = float(expr.evalf())
        return result
    except Exception:
        return None


class ComparisonResult(BaseModel):
    """Result of comparing predicted vs ground truth answer."""

    verdict: str  # "CORRECT", "INCORRECT", "ERROR"
    confidence: float  # 0.0 to 1.0
    reason: str  # Human-readable explanation
    predicted_value: Optional[float] = None
    ground_truth_value: Optional[float] = None
    relative_error: Optional[float] = None

    # NEW: Extraction method tracking for debugging
    predicted_extraction_method: Optional[str] = None
    ground_truth_extraction_method: Optional[str] = None


class AnswerComparator:
    """Compare predicted and ground truth answers with tolerance."""

    def __init__(self, tolerance: float = 0.015, allow_unit_conversion: bool = True):
        """Initialize comparator.

        Args:
            tolerance: Relative tolerance for numeric comparison (0.01 = 1%)
            allow_unit_conversion: Allow unit conversion when comparing
        """
        self.tolerance = tolerance
        self.allow_unit_conversion = allow_unit_conversion
        self.unit_converter = UnitConverter()

    def compare(
        self,
        predicted: Union[float, str],
        predicted_unit: str,  # IGNORED - using numeric-only comparison
        ground_truth: Union[float, str],
        ground_truth_unit: str,  # IGNORED - using numeric-only comparison
    ) -> ComparisonResult:
        """Compare predicted answer with ground truth (NUMERIC ONLY - units ignored).

        This method completely ignores units and focuses on robust numeric extraction
        and comparison. It uses cascading extraction strategies to handle various
        answer formats.

        Args:
            predicted: Predicted numeric answer or string
            predicted_unit: Unit of predicted answer (IGNORED)
            ground_truth: Ground truth answer
            ground_truth_unit: Unit of ground truth answer (IGNORED)

        Returns:
            ComparisonResult with verdict, extraction methods, and explanation
        """
        try:
            # Combine answer and unit for extraction (handles "8.44" + "$10^{34}$")
            # This is critical when exponent is embedded in the unit field
            combined_pred = f"{predicted} {predicted_unit}".strip() if predicted_unit else str(predicted)
            combined_truth = f"{ground_truth} {ground_truth_unit}".strip() if ground_truth_unit else str(ground_truth)

            # Extract numeric values using robust extraction function
            pred_value, pred_method = extract_number(combined_pred)
            truth_value, truth_method = extract_number(combined_truth)

            # Check for extraction failures
            if pred_value is None or truth_value is None:
                reason = (
                    f"Failed to extract numeric values. "
                    f"Predicted: {predicted} (method: {pred_method}), "
                    f"Ground truth: {ground_truth} (method: {truth_method})"
                )
                return ComparisonResult(
                    verdict="ERROR",
                    confidence=0.0,
                    reason=reason,
                    predicted_value=pred_value,
                    ground_truth_value=truth_value,
                    predicted_extraction_method=pred_method,
                    ground_truth_extraction_method=truth_method,
                )

            # Numeric comparison with tolerance (units already ignored)
            is_correct, relative_error = self._numeric_compare(pred_value, truth_value)

            # If comparison fails, try rad↔deg conversion as fallback
            if not is_correct:
                converted_pred, conversion_applied = self._try_angular_conversion(
                    pred_value, predicted_unit, truth_value, ground_truth_unit
                )
                if conversion_applied:
                    is_correct, relative_error = self._numeric_compare(converted_pred, truth_value)
                    if is_correct:
                        pred_value = converted_pred  # Use converted value for reporting

            # If still not correct, try metric prefix unit conversion (e.g., m² ↔ nm²)
            if not is_correct and self.allow_unit_conversion:
                try:
                    # Try converting predicted value to ground truth units
                    converted_pred = self.unit_converter.normalize(
                        pred_value, predicted_unit, ground_truth_unit
                    )
                    if converted_pred is not None and converted_pred != pred_value:
                        is_correct, relative_error = self._numeric_compare(converted_pred, truth_value)
                        if is_correct:
                            pred_value = converted_pred  # Use converted value for reporting
                            pred_method = f"{pred_method}+unit_conversion"
                except Exception:
                    pass  # Fall back to original comparison

            if is_correct:
                reason = (
                    f"✓ CORRECT: Numeric match within {self.tolerance*100:.1f}% tolerance. "
                    f"Predicted: {pred_value:.6e} (via {pred_method}), "
                    f"Ground truth: {truth_value:.6e} (via {truth_method}), "
                    f"Error: {relative_error*100:.4f}%"
                )
                return ComparisonResult(
                    verdict="CORRECT",
                    confidence=1.0 - min(relative_error / self.tolerance, 1.0),
                    reason=reason,
                    predicted_value=pred_value,
                    ground_truth_value=truth_value,
                    relative_error=relative_error,
                    predicted_extraction_method=pred_method,
                    ground_truth_extraction_method=truth_method,
                )
            else:
                reason = (
                    f"✗ INCORRECT: Numeric mismatch. "
                    f"Predicted: {pred_value:.6e} (via {pred_method}), "
                    f"Ground truth: {truth_value:.6e} (via {truth_method}), "
                    f"Error: {relative_error*100:.4f}% (tolerance: {self.tolerance*100:.1f}%)"
                )
                return ComparisonResult(
                    verdict="INCORRECT",
                    confidence=0.0,
                    reason=reason,
                    predicted_value=pred_value,
                    ground_truth_value=truth_value,
                    relative_error=relative_error,
                    predicted_extraction_method=pred_method,
                    ground_truth_extraction_method=truth_method,
                )

        except Exception as e:
            reason = f"Comparison error: {str(e)}"
            return ComparisonResult(
                verdict="ERROR",
                confidence=0.0,
                reason=reason,
            )

    def _parse_answer(self, answer: Union[float, str]) -> Optional[float]:
        """Parse answer to numeric value.

        Args:
            answer: Numeric or string answer

        Returns:
            Parsed numeric value or None
        """
        if isinstance(answer, (int, float)):
            return float(answer)

        if isinstance(answer, str):
            # Remove whitespace
            text = answer.strip()

            # Handle scientific notation
            try:
                return float(text)
            except ValueError:
                pass

            # Try to extract numeric part (e.g., "1.5 m/s" -> 1.5)
            parts = text.split()
            if parts:
                try:
                    return float(parts[0])
                except ValueError:
                    pass

            # Try to evaluate as symbolic expression (e.g., "acos(2/3)")
            symbolic_result = evaluate_symbolic_expression(text)
            if symbolic_result is not None:
                return symbolic_result

        return None

    def _numeric_compare(self, pred: float, truth: float) -> tuple[bool, float]:
        """Compare two numeric values with relative tolerance.

        Args:
            pred: Predicted value
            truth: Ground truth value

        Returns:
            Tuple of (is_correct, relative_error)
        """
        # Handle edge cases
        if math.isnan(pred) or math.isnan(truth):
            return False, float('inf')

        if math.isinf(pred) or math.isinf(truth):
            return pred == truth, float('inf') if pred != truth else 0.0

        # Avoid division by zero
        if truth == 0:
            if pred == 0:
                return True, 0.0
            else:
                return False, float('inf')

        # Calculate relative error
        relative_error = abs(pred - truth) / abs(truth)

        # Compare with tolerance
        is_correct = relative_error <= self.tolerance

        return is_correct, relative_error

    def _try_angular_conversion(
        self,
        pred_value: float,
        pred_unit: str,
        truth_value: float,
        truth_unit: str
    ) -> tuple[float, bool]:
        """Try to convert between radians and degrees if units mismatch.

        Args:
            pred_value: Predicted numeric value
            pred_unit: Predicted unit string
            truth_value: Ground truth numeric value
            truth_unit: Ground truth unit string

        Returns:
            Tuple of (converted_value, conversion_was_applied)
        """
        # Normalize unit strings
        pred_unit_lower = (pred_unit or "").lower().strip()
        truth_unit_lower = (truth_unit or "").lower().strip()

        # Define unit patterns
        rad_patterns = ["rad", "radian", "radians"]
        deg_patterns = ["deg", "degree", "degrees", "°", "^\\circ", "circ"]

        pred_is_rad = any(p in pred_unit_lower for p in rad_patterns)
        pred_is_deg = any(p in pred_unit_lower for p in deg_patterns)
        truth_is_rad = any(p in truth_unit_lower for p in rad_patterns)
        truth_is_deg = any(p in truth_unit_lower for p in deg_patterns)

        # Only convert if there's a clear rad↔deg mismatch
        if pred_is_rad and truth_is_deg:
            # Convert predicted from radians to degrees
            return math.degrees(pred_value), True
        elif pred_is_deg and truth_is_rad:
            # Convert predicted from degrees to radians
            return math.radians(pred_value), True

        # No conversion needed or applicable
        return pred_value, False
