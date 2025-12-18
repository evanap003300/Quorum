"""
Abstract base classes and protocols for the planning system.

This module defines the interfaces that critics, gatekeepers, and solvers must implement.
These interfaces enable extensibility and allow users to implement custom critics/gatekeepers.
"""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable, Tuple, List
from .schemas import Plan, CritiqueUnit


class BaseCritic(ABC):
    """
    Abstract base class for all critics in the planning system.

    Critics analyze a plan and return structured feedback (CritiqueUnit objects).
    Each critic focuses on a specific aspect (physics, dependencies, edge cases, etc.).

    Example subclasses:
    - PhysicsLawyer: Checks physical laws, units, sign conventions
    - DependencyChecker: Verifies variable dependencies, detects cycles
    - PreMortem: Identifies most likely failure points

    Usage:
        critic = MyCustomCritic()
        critiques = critic.critique(plan)
        for critique in critiques:
            print(f"Issue: {critique.issue}, Severity: {critique.severity}")
    """

    @property
    @abstractmethod
    def critic_id(self) -> str:
        """
        Unique identifier for this critic.

        Returns:
            String ID (e.g., 'physics_lawyer', 'dependency_checker', 'pre_mortem')
        """
        pass

    @abstractmethod
    def critique(self, plan: Plan) -> List[CritiqueUnit]:
        """
        Analyze a plan and return list of critiques.

        Implement this method to analyze the plan and produce structured feedback.
        Return an empty list if no issues are found.

        Args:
            plan: The Plan to critique

        Returns:
            List of CritiqueUnit objects (empty if no issues found).
            Each CritiqueUnit should:
            - Identify specific step(s) with issues
            - Provide evidence from the plan
            - Suggest concrete fixes
            - Be focused on one issue per unit

        Raises:
            Exception: Only if the critique process itself fails (not for plan issues)
        """
        pass

    def __str__(self) -> str:
        """Return a string representation."""
        return f"{self.__class__.__name__}(id='{self.critic_id}')"


class BaseGatekeeper(ABC):
    """
    Abstract base class for the gatekeeper (plan judge).

    The gatekeeper reviews all critiques from all critics and decides:
    1. Whether the plan should be refined
    2. Which critiques are blocking (must be fixed) vs warnings/suggestions

    The gatekeeper implements the decision logic for the critique-refine loop.
    Max 2 critique-refine cycles are enforced by the Planner class.

    Example implementation:
        class ThresholdGatekeeper(BaseGatekeeper):
            def should_refine(self, critiques):
                blocking = [c for c in critiques if c.severity == "blocking"]
                return (len(blocking) > 0, blocking)

    Usage:
        gatekeeper = MyGatekeepImpl()
        should_refine, blocking_issues = gatekeeper.should_refine(critiques)
        if should_refine:
            # Send to planner for refinement
            pass
        else:
            # Plan is approved
            pass
    """

    @abstractmethod
    def should_refine(self, critiques: List[CritiqueUnit]) -> Tuple[bool, List[CritiqueUnit]]:
        """
        Decide if plan needs refinement based on critiques.

        Implement this method to define the decision logic. Common strategies:
        - Majority rule: approve if most critiques are suggestions
        - Threshold: approve if <N blocking critiques
        - Consensus: approve only if all critics agree

        Args:
            critiques: List of all CritiqueUnit objects from all critics

        Returns:
            Tuple of (should_refine: bool, blocking_issues: list[CritiqueUnit])
            - should_refine: True if plan needs refinement, False if approved
            - blocking_issues: List of critiques that must be fixed (empty if none)

        Examples:
            # Strict: any blocking issue requires refinement
            blocking = [c for c in critiques if c.severity == "blocking"]
            return (len(blocking) > 0, blocking)

            # Permissive: only refine if multiple blocking issues
            blocking = [c for c in critiques if c.severity == "blocking"]
            return (len(blocking) > 2, blocking)

            # Custom: apply weighted scoring
            score = sum(1 if c.severity == "blocking" else 0.5 for c in critiques)
            return (score > threshold, blocking)
        """
        pass

    def __str__(self) -> str:
        """Return a string representation."""
        return f"{self.__class__.__name__}()"


@runtime_checkable
class SolverProtocol(Protocol):
    """
    Protocol for MDAP solver engine.

    This defines the interface that the solver must implement for the Planner
    to pass executed plans to it. This is used for type checking and documentation.

    The solver receives an atomic step, executes it via multiple agents,
    uses voting to select the best result, and returns it.

    Future integration point (not yet implemented in Quorum):
        planner.create_plan() -> StateObject with Plan
        solver.execute_plan(state.plan, state) -> Results
    """

    def execute_step(self, step: "AtomicStep", state: "StateObject") -> dict:
        """
        Execute a single atomic step.

        Args:
            step: The AtomicStep to execute
            state: The current StateObject with context

        Returns:
            Dictionary with:
            - 'output': The computed value
            - 'status': 'success' or 'error'
            - 'error_message': If failed
            - 'solver_votes': Voting record (for k-group voting)
        """
        ...

    def execute_plan(self, plan: Plan, state: "StateObject") -> dict:
        """
        Execute all steps in a plan.

        Args:
            plan: The complete Plan to execute
            state: The current StateObject with context

        Returns:
            Dictionary with:
            - 'final_result': Computed answer
            - 'steps_executed': Number of steps completed
            - 'errors': Any execution errors
            - 'final_state': Updated StateObject
        """
        ...


# Import for type hints (avoid circular imports)
try:
    from .schemas import AtomicStep, StateObject
except ImportError:
    # These might not be available during initial module load
    pass


# Export all interfaces
__all__ = [
    "BaseCritic",
    "BaseGatekeeper",
    "SolverProtocol",
]
