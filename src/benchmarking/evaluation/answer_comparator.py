"""Answer comparison logic for benchmark evaluation."""

from pydantic import BaseModel
from typing import Union, Optional
from .unit_converter import UnitConverter
import math

try:
    import sympy
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


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


class AnswerComparator:
    """Compare predicted and ground truth answers with tolerance."""

    def __init__(self, tolerance: float = 0.01, allow_unit_conversion: bool = True):
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
        predicted_unit: str,
        ground_truth: Union[float, str],
        ground_truth_unit: str,
    ) -> ComparisonResult:
        """Compare predicted answer with ground truth.

        Args:
            predicted: Predicted numeric answer or string
            predicted_unit: Unit of predicted answer
            ground_truth: Ground truth answer
            ground_truth_unit: Unit of ground truth answer

        Returns:
            ComparisonResult with verdict and explanation
        """
        try:
            # Parse answers to numeric values
            pred_value = self._parse_answer(predicted)
            truth_value = self._parse_answer(ground_truth)

            # Check for parsing failures
            if pred_value is None or truth_value is None:
                reason = (
                    f"Failed to parse answers as numeric values. "
                    f"Predicted: {predicted} ({predicted_unit}), "
                    f"Ground truth: {ground_truth} ({ground_truth_unit})"
                )
                return ComparisonResult(
                    verdict="ERROR",
                    confidence=0.0,
                    reason=reason,
                    predicted_value=pred_value,
                    ground_truth_value=truth_value,
                )

            # Handle unit conversion
            if predicted_unit != ground_truth_unit and self.allow_unit_conversion:
                try:
                    converted_pred = self.unit_converter.normalize(
                        pred_value, predicted_unit, ground_truth_unit
                    )
                    if converted_pred is None:
                        reason = (
                            f"Unit conversion failed: could not convert "
                            f"{pred_value} {predicted_unit} to {ground_truth_unit}. "
                            f"Ground truth: {truth_value} {ground_truth_unit}"
                        )
                        return ComparisonResult(
                            verdict="ERROR",
                            confidence=0.0,
                            reason=reason,
                            predicted_value=pred_value,
                            ground_truth_value=truth_value,
                        )
                    pred_value = converted_pred
                except Exception as e:
                    reason = f"Unit conversion error: {str(e)}"
                    return ComparisonResult(
                        verdict="ERROR",
                        confidence=0.0,
                        reason=reason,
                        predicted_value=pred_value,
                        ground_truth_value=truth_value,
                    )

            # Numeric comparison with tolerance
            is_correct, relative_error = self._numeric_compare(pred_value, truth_value)

            if is_correct:
                reason = (
                    f"✓ CORRECT: Numeric match within {self.tolerance*100:.1f}% tolerance. "
                    f"Predicted: {pred_value:.6e}, Ground truth: {truth_value:.6e}, "
                    f"Error: {relative_error*100:.4f}%"
                )
                return ComparisonResult(
                    verdict="CORRECT",
                    confidence=1.0 - min(relative_error / self.tolerance, 1.0),  # Confidence based on error
                    reason=reason,
                    predicted_value=pred_value,
                    ground_truth_value=truth_value,
                    relative_error=relative_error,
                )
            else:
                reason = (
                    f"✗ INCORRECT: Numeric mismatch. "
                    f"Predicted: {pred_value:.6e}, Ground truth: {truth_value:.6e}, "
                    f"Error: {relative_error*100:.4f}% (tolerance: {self.tolerance*100:.1f}%)"
                )
                return ComparisonResult(
                    verdict="INCORRECT",
                    confidence=0.0,
                    reason=reason,
                    predicted_value=pred_value,
                    ground_truth_value=truth_value,
                    relative_error=relative_error,
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
