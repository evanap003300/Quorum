"""Problem executor with timeout handling."""

from pydantic import BaseModel
from typing import Union, Optional
import time
import signal
from contextlib import contextmanager
import sys
import os

try:
    import sympy
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

# Add orchestrator to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../core/orchestrator'))

from src.benchmarking.dataset.scibench_loader import BenchmarkProblem


def evaluate_symbolic_answer(answer: Union[float, str]) -> Union[float, str]:
    """Try to evaluate a symbolic expression to a numeric value.

    Args:
        answer: Numeric or string answer

    Returns:
        Numeric value if it's a symbolic expression that can be evaluated,
        otherwise returns the original answer
    """
    if not isinstance(answer, str):
        return answer

    if not SYMPY_AVAILABLE:
        return answer

    try:
        # Parse and evaluate the expression
        expr = sympy.sympify(answer.strip())
        # Evaluate to a floating point number
        result = float(expr.evalf())
        return result
    except Exception:
        # If evaluation fails, return original answer
        return answer


class ProblemResult(BaseModel):
    """Result of executing a single problem."""

    problem_id: str
    success: bool
    predicted_answer: Optional[Union[float, str]]
    predicted_unit: Optional[str]
    ground_truth_answer: str
    ground_truth_unit: str

    total_time: float
    total_cost: float
    plan_time: Optional[float] = None
    execution_time: Optional[float] = None
    review_time: Optional[float] = None
    vision_time: Optional[float] = None

    verdict: str  # "CORRECT", "INCORRECT", "ERROR", "TIMEOUT"
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    num_steps: Optional[int] = None
    failed_at_step: Optional[int] = None
    comparison_reason: Optional[str] = None  # Why the answer matched/didn't match


class TimeoutError(Exception):
    """Raised when problem execution exceeds timeout."""
    pass


@contextmanager
def time_limit(seconds: int):
    """Context manager for Unix alarm-based timeout.

    Args:
        seconds: Timeout duration in seconds

    Raises:
        TimeoutError: If timeout is exceeded
    """
    def signal_handler(signum, frame):
        raise TimeoutError(f"Problem execution exceeded {seconds}s timeout")

    # Set signal handler and alarm
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)  # Disable alarm


class ProblemExecutor:
    """Execute individual benchmark problems with timeout handling."""

    def __init__(self, timeout_seconds: int = 300):
        """Initialize executor.

        Args:
            timeout_seconds: Timeout per problem in seconds
        """
        self.timeout_seconds = timeout_seconds

    def execute(self, problem: BenchmarkProblem) -> ProblemResult:
        """Execute a single problem.

        Args:
            problem: Benchmark problem to solve

        Returns:
            ProblemResult with outcome and metrics
        """
        start_time = time.time()

        try:
            # Import here to avoid circular imports
            from orchestrate import solve_problem

            # Execute with timeout
            result = self._execute_with_timeout(problem.problem_text)

            # Extract results
            end_time = time.time()
            total_time = end_time - start_time

            if result.get("success", False):
                # Extract plan steps safely - handle both dict and Plan object
                plan_obj = result.get("plan")
                num_steps = None
                if plan_obj:
                    try:
                        # Try as dict first
                        if isinstance(plan_obj, dict):
                            num_steps = len(plan_obj.get("steps", []))
                        else:
                            # Try as Pydantic model
                            num_steps = len(plan_obj.steps)
                    except Exception:
                        num_steps = None

                return ProblemResult(
                    problem_id=problem.problem_id,
                    success=True,
                    predicted_answer=evaluate_symbolic_answer(result.get("final_answer")),
                    predicted_unit=result.get("final_unit", ""),
                    ground_truth_answer=problem.ground_truth_answer,
                    ground_truth_unit=problem.ground_truth_unit,
                    total_time=total_time,
                    total_cost=result.get("total_cost", 0.0),
                    plan_time=result.get("plan_time"),
                    execution_time=result.get("execution_time"),
                    review_time=result.get("review_time"),
                    vision_time=result.get("vision_time"),
                    verdict="CORRECT",  # Will be set by comparator
                    error_message=None,
                    num_steps=num_steps,
                    failed_at_step=result.get("failed_at_step"),
                )
            else:
                # Extract plan steps safely - handle both dict and Plan object
                plan_obj = result.get("plan")
                num_steps = None
                if plan_obj:
                    try:
                        # Try as dict first
                        if isinstance(plan_obj, dict):
                            num_steps = len(plan_obj.get("steps", []))
                        else:
                            # Try as Pydantic model
                            num_steps = len(plan_obj.steps)
                    except Exception:
                        num_steps = None

                return ProblemResult(
                    problem_id=problem.problem_id,
                    success=False,
                    predicted_answer=evaluate_symbolic_answer(result.get("final_answer")),
                    predicted_unit=result.get("final_unit"),
                    ground_truth_answer=problem.ground_truth_answer,
                    ground_truth_unit=problem.ground_truth_unit,
                    total_time=total_time,
                    total_cost=result.get("total_cost", 0.0),
                    plan_time=result.get("plan_time"),
                    execution_time=result.get("execution_time"),
                    review_time=result.get("review_time"),
                    vision_time=result.get("vision_time"),
                    verdict="ERROR",
                    error_message=result.get("error"),
                    error_type=self._categorize_error(result.get("error", "")),
                    num_steps=num_steps,
                    failed_at_step=result.get("failed_at_step"),
                )

        except TimeoutError as e:
            end_time = time.time()
            return ProblemResult(
                problem_id=problem.problem_id,
                success=False,
                predicted_answer=None,
                predicted_unit=None,
                ground_truth_answer=problem.ground_truth_answer,
                ground_truth_unit=problem.ground_truth_unit,
                total_time=end_time - start_time,
                total_cost=0.0,
                verdict="TIMEOUT",
                error_message=str(e),
                error_type="TIMEOUT",
            )

        except Exception as e:
            end_time = time.time()
            return ProblemResult(
                problem_id=problem.problem_id,
                success=False,
                predicted_answer=None,
                predicted_unit=None,
                ground_truth_answer=problem.ground_truth_answer,
                ground_truth_unit=problem.ground_truth_unit,
                total_time=end_time - start_time,
                total_cost=0.0,
                verdict="ERROR",
                error_message=str(e),
                error_type="EXECUTION_ERROR",
            )

    def _execute_with_timeout(self, problem_text: str) -> dict:
        """Execute solve_problem with timeout handling.

        Args:
            problem_text: Problem text to solve

        Returns:
            Result dictionary from solve_problem
        """
        from orchestrate import solve_problem

        # On Unix systems, use signal.alarm for timeout
        if sys.platform != "win32":
            try:
                with time_limit(self.timeout_seconds):
                    result = solve_problem(problem_text)
                    return result
            except TimeoutError:
                raise
            except Exception as e:
                # Catch any other errors and return as error result
                return {
                    "success": False,
                    "error": f"Solver error: {str(e)[:200]}",
                    "final_answer": None,
                    "final_unit": None,
                    "total_time": 0,
                    "total_cost": 0.0,
                }
        else:
            # On Windows, just execute without timeout (signal.alarm not available)
            try:
                result = solve_problem(problem_text)
                return result
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Solver error: {str(e)[:200]}",
                    "final_answer": None,
                    "final_unit": None,
                    "total_time": 0,
                    "total_cost": 0.0,
                }

    def _categorize_error(self, error_msg: str) -> str:
        """Categorize error type based on error message.

        Args:
            error_msg: Error message from solver

        Returns:
            Error category string
        """
        error_lower = error_msg.lower()

        if "planning" in error_lower:
            return "PLANNING_ERROR"
        elif "physics" in error_lower or "validation" in error_lower:
            return "VALIDATION_ERROR"
        elif "execution" in error_lower or "calculate" in error_lower:
            return "EXECUTION_ERROR"
        elif "vision" in error_lower or "image" in error_lower:
            return "VISION_ERROR"
        else:
            return "UNKNOWN_ERROR"
