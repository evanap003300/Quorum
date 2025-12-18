"""
Pydantic v2 schemas for the Quorum planning system.

This module defines all data models with strict validation to ensure type safety
and prevent silent data coercion. All models use Pydantic v2 with strict mode enabled.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Literal, Optional, Any
from datetime import datetime
from uuid import uuid4
import re


# Global strict configuration for all models
STRICT_CONFIG = ConfigDict(strict=True, extra='forbid')


class AtomicStep(BaseModel):
    """
    A single, atomic operation in a plan.

    Each step represents exactly one operation (algebraic, symbolic, matrix, ODE, or substitution).
    No compound operations are allowed. Steps are validated for atomicity at parse time.
    """
    model_config = STRICT_CONFIG

    step_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this step (UUID format)"
    )
    type: Literal["algebraic", "symbolic", "matrix", "ode", "substitute"] = Field(
        description="Operation type: algebraic (solve equations), symbolic (manipulate expressions), "
                    "matrix (linear algebra), ode (differential equations), substitute (variable substitution)"
    )
    description: str = Field(
        max_length=200,
        description="Concise, single-sentence description of what this step does"
    )
    inputs: list[str] = Field(
        min_length=1,
        description="List of input variable names or equation IDs required for this step"
    )
    output: str = Field(
        description="Output variable name or result identifier produced by this step"
    )
    expected_units: Optional[str] = Field(
        default=None,
        description="Physical units of the output (e.g., 'm/s^2', 'N', 'J'). None if dimensionless or symbolic."
    )
    tolerance: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Numerical tolerance for verification (e.g., 1e-8). None for exact symbolic results."
    )
    justification: str = Field(
        max_length=200,
        description="Short justification (≤50 tokens) explaining why this step is valid and which physical law/principle applies"
    )

    @field_validator('justification')
    @classmethod
    def validate_justification_length(cls, v: str) -> str:
        """
        Validate that justification is concise (≤50 tokens).

        Using rough heuristic: 4 characters ≈ 1 token. So 200 chars ≈ 50 tokens.
        Raises ValueError if justification appears too long.
        """
        estimated_tokens = len(v) / 4
        if estimated_tokens > 50:
            raise ValueError(
                f"Justification too long (~{estimated_tokens:.0f} tokens, max 50). "
                f"Keep justifications short and focused on the physics/math principle."
            )
        if len(v) < 10:
            raise ValueError(
                "Justification too short. Provide at least a brief explanation of which principle applies."
            )
        return v

    @field_validator('description')
    @classmethod
    def validate_description_no_compound_verbs(cls, v: str) -> str:
        """
        Preliminary check for compound operations in description.
        This is a heuristic; full atomicity validation happens in validators.py.
        """
        description_lower = v.lower()
        # Check for obvious multi-step indicators
        compound_patterns = [
            (r'\band\b', "Multiple operations indicated by 'and'"),
            (r'\bthen\b', "Sequence indicated by 'then'"),
            (r',\s*then\b', "Comma-then sequence"),
        ]

        for pattern, msg in compound_patterns:
            if re.search(pattern, description_lower):
                raise ValueError(
                    f"Step description appears to contain multiple operations: '{v}'. "
                    f"Each step must be atomic (single operation). {msg}. "
                    f"Break this into separate steps."
                )
        return v


class Plan(BaseModel):
    """
    A complete plan consisting of ordered, atomic steps with metadata.

    The plan is versioned and immutable once approved by the Gatekeeper.
    Each plan describes a symbolic decomposition of the problem (no numeric computation).
    """
    model_config = STRICT_CONFIG

    plan_version: int = Field(
        default=1,
        ge=1,
        description="Plan version number (incremented on each refinement)"
    )
    variables: dict[str, dict[str, str]] = Field(
        default_factory=dict,
        description="Mapping of variable names to their descriptions and units. "
                    "Example: {'v0': {'unit': 'm/s', 'desc': 'initial velocity'}, ...}"
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="List of explicit assumptions made in the plan "
                    "(e.g., 'no air resistance', 'constant acceleration', 'g:9.81')"
    )
    steps: list[AtomicStep] = Field(
        min_length=1,
        description="Ordered list of atomic steps. Each step is a single operation."
    )
    composition: dict[str, Any] = Field(
        default_factory=dict,
        description="Step dependency graph showing which steps depend on which. "
                    "Example: {'step2': ['step1'], 'step3': ['step1', 'step2']}"
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Plan metadata: planner_model, created_at timestamp, etc."
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="ISO8601 timestamp when this plan was created"
    )

    @field_validator('steps')
    @classmethod
    def validate_step_atomicity(cls, steps: list[AtomicStep]) -> list[AtomicStep]:
        """
        Validate that all steps are atomic (import happens at runtime to avoid circular imports).
        This is a placeholder; full validation happens in validators.py after import.
        """
        # Return steps as-is; full validation is delegated to validators.py
        # which is called separately after schema instantiation
        return steps


class StateObject(BaseModel):
    """
    The master state object tracking the entire problem lifecycle.

    This is the authoritative source of truth for the planning process.
    Only the Cleaner/Validator module may commit changes to 'knowns'.
    All other components read from the state or propose changes.
    """
    model_config = STRICT_CONFIG

    state_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this state object (UUID format)"
    )
    problem_text: str = Field(
        min_length=1,
        description="Original natural language problem description from the user"
    )
    domain: Literal["physics", "mathematics", "chemistry", "engineering", "biology", "other"] = Field(
        default="physics",
        description="Problem domain for contextual prompting"
    )
    goal: str = Field(
        default="",
        description="What we are solving for (e.g., 'final velocity', 'acceleration')"
    )
    knowns: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Known variables with values and metadata. "
                    "Example: {'v0': {'value': 5, 'unit': 'm/s', 'source': 'user'}}. "
                    "WRITE-PROTECTED: Only Cleaner/Validator may modify."
    )
    unknowns: list[str] = Field(
        default_factory=list,
        description="List of unknown variables to solve for"
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Explicit assumptions (e.g., 'constant acceleration', 'no friction')"
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Constraints on variables (e.g., 'v_f must be positive', 'mass > 0')"
    )
    plan: Optional['Plan'] = Field(
        default=None,
        description="Generated plan (None until planner creates it)"
    )
    audit_log: list[dict[str, str]] = Field(
        default_factory=list,
        description="Immutable audit trail of all state changes and events"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="ISO8601 timestamp when this state object was created"
    )

    def log_event(self, event_type: str, message: str) -> None:
        """
        Append an event to the audit log.

        Args:
            event_type: Type of event (e.g., 'planning_started', 'plan_generated', 'critique_round')
            message: Detailed message about what happened
        """
        self.audit_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": event_type,
            "message": message
        })


class CritiqueUnit(BaseModel):
    """
    Structured feedback from a critic about a plan.

    Critics return instances of this class to provide actionable, structured feedback
    that the Planner can use to refine the plan. Each critique is specific to a step.
    """
    model_config = STRICT_CONFIG

    critic_id: str = Field(
        description="Unique identifier of the critic that produced this critique "
                    "(e.g., 'physics_lawyer', 'dependency_checker', 'pre_mortem')"
    )
    step_id: str = Field(
        description="ID of the step this critique applies to"
    )
    issue: Literal[
        "unit_mismatch",
        "missing_dependency",
        "law_inapplicable",
        "assumption_violation",
        "compound_operation",
        "undefined_variable",
        "other"
    ] = Field(
        description="Categorization of the issue"
    )
    claim: str = Field(
        max_length=300,
        description="The critic's assertion/finding about what's wrong"
    )
    support: str = Field(
        max_length=500,
        description="Evidence or reasoning from state/assumptions that backs up the claim"
    )
    severity: Literal["blocking", "warning", "suggestion"] = Field(
        description="Severity level: 'blocking' must be fixed before plan approval, "
                    "'warning' should be addressed, 'suggestion' is optional"
    )
    suggestion: Optional[str] = Field(
        default=None,
        max_length=300,
        description="Recommended fix or refinement (optional)"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="ISO8601 timestamp when this critique was generated"
    )


# Export all schemas
__all__ = [
    "STRICT_CONFIG",
    "AtomicStep",
    "Plan",
    "StateObject",
    "CritiqueUnit",
]
