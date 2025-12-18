"""
Custom exceptions for the planning system.

Exception hierarchy:
- PlanningError (base)
  - SchemaValidationError
  - LLMError
  - AtomicityViolationError (also inherits from ValueError)
  - CritiqueError
"""


class PlanningError(Exception):
    """
    Base exception for all planning-related errors.

    This is the top-level exception that should be caught when planning fails.
    More specific exceptions inherit from this.
    """
    pass


class SchemaValidationError(PlanningError):
    """
    Raised when LLM output doesn't match the expected schema.

    This occurs when:
    - LLM output is not valid JSON
    - JSON doesn't conform to Pydantic models
    - Required fields are missing
    - Field values are of wrong type
    """

    def __init__(self, message: str, raw_output: str = None):
        """
        Initialize SchemaValidationError.

        Args:
            message: Human-readable error message
            raw_output: The LLM output that failed validation (for debugging)
        """
        super().__init__(message)
        self.raw_output = raw_output
        self.message = message

    def __str__(self) -> str:
        """Return detailed error message."""
        msg = f"Schema validation failed: {self.message}"
        if self.raw_output:
            msg += f"\n\nRaw LLM output (first 500 chars):\n{self.raw_output[:500]}"
        return msg


class LLMError(PlanningError):
    """
    Raised when an LLM API call fails.

    This occurs when:
    - Network connection fails
    - API rate limit exceeded
    - API returns an error response
    - LLM times out
    - Invalid API credentials
    """

    def __init__(self, message: str, api_error: Exception = None):
        """
        Initialize LLMError.

        Args:
            message: Human-readable error message
            api_error: The underlying exception from the OpenAI client
        """
        super().__init__(message)
        self.api_error = api_error
        self.message = message

    def __str__(self) -> str:
        """Return detailed error message."""
        msg = f"LLM API error: {self.message}"
        if self.api_error:
            msg += f"\nUnderlying error: {str(self.api_error)}"
        return msg


class AtomicityViolationError(PlanningError, ValueError):
    """
    Raised when a step violates the atomicity constraint.

    A step is atomic if it performs exactly one operation. Violations include:
    - Multiple action verbs (e.g., "solve and substitute")
    - Sequence indicators (e.g., "first..., then...")
    - Multiple operations separated by conjunctions

    This exception inherits from both PlanningError and ValueError for compatibility.
    """

    def __init__(self, message: str, step_id: str = None, description: str = None):
        """
        Initialize AtomicityViolationError.

        Args:
            message: Human-readable error message
            step_id: The ID of the step that violated atomicity
            description: The step description that violated the rule
        """
        super().__init__(message)
        self.step_id = step_id
        self.description = description
        self.message = message

    def __str__(self) -> str:
        """Return detailed error message."""
        msg = f"Atomicity violation: {self.message}"
        if self.step_id:
            msg += f"\nStep ID: {self.step_id}"
        if self.description:
            msg += f"\nDescription: {self.description}"
        msg += "\n\nReminder: Each step must be atomic (single operation). " \
               "Examples:\n" \
               "  ✓ 'Solve F = ma for a'\n" \
               "  ✓ 'Substitute v0 into equation'\n" \
               "  ✗ 'Solve for a then substitute into next equation' (compound)\n" \
               "  ✗ 'Differentiate and integrate' (multiple operations)"
        return msg


class CritiqueError(PlanningError):
    """
    Raised when critique-related operations fail.

    This occurs when:
    - A critic produces invalid output
    - Gatekeeper fails to evaluate critiques
    - Critique-refine loop hits max iterations
    - Critic returns non-JSON output
    """

    def __init__(self, message: str, critic_id: str = None):
        """
        Initialize CritiqueError.

        Args:
            message: Human-readable error message
            critic_id: The ID of the critic that failed (if applicable)
        """
        super().__init__(message)
        self.critic_id = critic_id
        self.message = message

    def __str__(self) -> str:
        """Return detailed error message."""
        msg = f"Critique error: {self.message}"
        if self.critic_id:
            msg += f"\nCritic: {self.critic_id}"
        return msg


class RefinementError(PlanningError):
    """
    Raised when plan refinement fails.

    This occurs when:
    - Refinement prompt produces invalid JSON
    - Refined plan has new violations
    - Max refinement iterations exceeded
    """

    def __init__(self, message: str, iteration: int = None, attempts: int = None):
        """
        Initialize RefinementError.

        Args:
            message: Human-readable error message
            iteration: The refinement iteration number
            attempts: Total number of attempts before failure
        """
        super().__init__(message)
        self.iteration = iteration
        self.attempts = attempts
        self.message = message

    def __str__(self) -> str:
        """Return detailed error message."""
        msg = f"Plan refinement error: {self.message}"
        if self.iteration:
            msg += f"\nRefinement iteration: {self.iteration}"
        if self.attempts:
            msg += f"\nAttempts before failure: {self.attempts}"
        return msg


# Export all exceptions
__all__ = [
    "PlanningError",
    "SchemaValidationError",
    "LLMError",
    "AtomicityViolationError",
    "CritiqueError",
    "RefinementError",
]
