"""
Validators for the planning system.

This module enforces critical constraints:
1. Atomicity: Each step must be a single operation (no compound operations)
2. Structure: Plans must have valid dependencies and no circular references
3. Variables: All input variables must be defined before use
"""

import re
from typing import Set, Tuple
from .schemas import AtomicStep, Plan
from .exceptions import AtomicityViolationError


# Patterns that indicate compound operations
COMPOUND_OPERATION_PATTERNS = [
    (r'\band\b(?!\s+(?:constant|force|velocity|acceleration|mass|displacement))', "Multiple operations indicated by 'and'"),
    (r'\bthen\b', "Sequence indicated by 'then'"),
    (r';\s*', "Semicolon indicates multiple statements"),
    (r',\s*(?:then|also|next)\b', "Comma-then or comma-also sequence"),
    (r'\bfirst\b.*\b(?:then|second|next)\b', "Explicit multi-step sequence"),
    (r';\s*(?:then\s+)?substitute', "Solve then substitute pattern"),
    (r'integrate.*differentiate|differentiate.*integrate', "Cross-operation (differentiate/integrate)"),
]

# Action verbs that suggest operations
ACTION_VERBS = [
    'solve', 'substitute', 'differentiate', 'integrate', 'simplify',
    'factor', 'expand', 'compute', 'calculate', 'evaluate', 'derive',
    'combine', 'manipulate', 'transform', 'isolate', 'rearrange',
    'solve_for', 'plug_in', 'apply',
]

# Maximum number of verbs allowed in a single step (strict: 1)
MAX_VERBS_PER_STEP = 1


class AtomicityViolationError(ValueError):
    """Raised when a step violates the atomicity constraint."""
    pass


def validate_step_atomicity(step: AtomicStep) -> None:
    """
    Validate that a step is truly atomic (single operation).

    A step violates atomicity if:
    1. Description contains patterns like "and", "then", ";", "first...second"
    2. Description contains multiple action verbs
    3. Description suggests compound operations

    Args:
        step: The AtomicStep to validate

    Raises:
        AtomicityViolationError: If step contains compound operations
    """
    description_lower = step.description.lower()

    # Check 1: Pattern-based detection
    for pattern, reason in COMPOUND_OPERATION_PATTERNS:
        if re.search(pattern, description_lower):
            raise AtomicityViolationError(
                f"Step '{step.step_id}' violates atomicity constraint. "
                f"Description: '{step.description}'\n"
                f"Reason: {reason}\n"
                f"Each step must represent exactly one operation. "
                f"Break this into separate steps."
            )

    # Check 2: Count action verbs
    verb_count = 0
    matched_verbs = []
    for verb in ACTION_VERBS:
        # Use word boundary to avoid false positives (e.g., 'solve' in 'solution')
        if re.search(r'\b' + verb + r'\b', description_lower):
            verb_count += 1
            matched_verbs.append(verb)

    if verb_count > MAX_VERBS_PER_STEP:
        raise AtomicityViolationError(
            f"Step '{step.step_id}' violates atomicity constraint. "
            f"Description: '{step.description}'\n"
            f"Found {verb_count} action verbs: {matched_verbs}\n"
            f"Each step should contain exactly one action verb. "
            f"Break this into separate steps."
        )

    # Check 3: Heuristic - check for conjunction indicators with verbs
    if verb_count == 1:
        # Check if there are coordination patterns that suggest additional operations
        coord_patterns = [
            (r'and\s+then\b', "and then"),
            (r'and\s+(?:also\s+)?(?:use|apply|solve|substitute)', "and use/apply/solve"),
            (r',\s*then\s+(?:use|substitute|apply)', ", then use/substitute"),
        ]

        for pattern, example in coord_patterns:
            if re.search(pattern, description_lower):
                raise AtomicityViolationError(
                    f"Step '{step.step_id}' violates atomicity constraint. "
                    f"Description: '{step.description}'\n"
                    f"Reason: Contains conjunction pattern '{example}' suggesting multiple operations\n"
                    f"Decompose this into atomic steps."
                )


def validate_plan_structure(plan: Plan) -> None:
    """
    Validate the overall structure and dependencies in a plan.

    Checks:
    1. Step IDs are unique
    2. Output variables are unique
    3. All step inputs reference defined variables
    4. No circular dependencies
    5. All variables used are defined somewhere

    Args:
        plan: The Plan to validate

    Raises:
        ValueError: If plan structure is invalid
    """
    # Check 1: Unique step IDs
    step_ids = [step.step_id for step in plan.steps]
    if len(step_ids) != len(set(step_ids)):
        duplicates = [sid for sid in set(step_ids) if step_ids.count(sid) > 1]
        raise ValueError(
            f"Plan contains duplicate step IDs: {duplicates}\n"
            f"Each step must have a unique ID."
        )

    # Check 2: Unique output variables
    output_vars = [step.output for step in plan.steps]
    if len(output_vars) != len(set(output_vars)):
        duplicates = [var for var in set(output_vars) if output_vars.count(var) > 1]
        raise ValueError(
            f"Plan contains steps with duplicate output variables: {duplicates}\n"
            f"Each step must produce a unique output variable."
        )

    # Collect all available variables (from plan.variables + step outputs)
    all_available_vars = set(plan.variables.keys())
    all_available_vars.update(output_vars)

    # Build step index for easier lookup
    step_by_id = {step.step_id: step for step in plan.steps}

    # Check 3: All inputs reference valid variables or previous step outputs
    for i, step in enumerate(plan.steps):
        for input_var in step.inputs:
            # Check if it's a known variable
            if input_var not in all_available_vars:
                raise ValueError(
                    f"Step '{step.step_id}' references undefined variable '{input_var}'. "
                    f"Available variables: {all_available_vars}\n"
                    f"Either: (1) add '{input_var}' to plan.variables, "
                    f"(2) ensure a previous step outputs '{input_var}', or "
                    f"(3) fix the input variable name."
                )

    # Check 4: Detect circular dependencies
    _check_circular_dependencies(plan, step_by_id)

    # Check 5: Check composition/dependency graph is consistent
    if plan.composition:
        _validate_composition_graph(plan, step_by_id)


def _check_circular_dependencies(plan: Plan, step_by_id: dict) -> None:
    """
    Detect circular dependencies in the plan.

    A circular dependency occurs if step A depends on step B which (transitively) depends on A.

    Args:
        plan: The Plan to check
        step_by_id: Mapping of step_id -> step

    Raises:
        ValueError: If circular dependencies are found
    """
    def find_dependencies(step_id: str, visited: Set[str], path: list) -> None:
        """Recursively find all dependencies."""
        if step_id in visited:
            # Found a cycle
            cycle_path = path[path.index(step_id):] + [step_id]
            raise ValueError(
                f"Circular dependency detected: {' -> '.join(cycle_path)}\n"
                f"Steps cannot depend on themselves (directly or transitively)."
            )

        if step_id not in step_by_id:
            return

        visited.add(step_id)
        path.append(step_id)

        step = step_by_id[step_id]

        # Find what variables this step produces
        produced_vars = {step.output}

        # For each subsequent step, check if any earlier step is needed
        for other_step in plan.steps:
            if other_step.step_id == step_id:
                break  # Stop at current step

            if any(var in other_step.inputs for var in produced_vars):
                # This step depends on a later step (reverse dependency - potential issue but not a cycle yet)
                pass

        # Now check what this step depends on and trace back
        for input_var in step.inputs:
            # Find which step(s) produce this variable
            for prev_step in plan.steps:
                if prev_step.output == input_var:
                    find_dependencies(prev_step.step_id, visited.copy(), path.copy())

    for step_id in step_by_id:
        try:
            find_dependencies(step_id, set(), [])
        except ValueError:
            raise


def _validate_composition_graph(plan: Plan, step_by_id: dict) -> None:
    """
    Validate that the composition dictionary accurately represents dependencies.

    The composition dict maps step IDs to lists of steps they depend on.
    If provided, it must match the actual dependencies inferred from inputs/outputs.

    Args:
        plan: The Plan to validate
        step_by_id: Mapping of step_id -> step

    Raises:
        ValueError: If composition graph is inconsistent with plan steps
    """
    # Infer dependencies from step inputs/outputs
    inferred_composition = {}
    for i, step in enumerate(plan.steps):
        dependencies = []
        for j, prev_step in enumerate(plan.steps):
            if j < i and prev_step.output in step.inputs:
                dependencies.append(prev_step.step_id)
        if dependencies:
            inferred_composition[step.step_id] = dependencies

    # Compare with provided composition (if non-empty)
    if plan.composition:
        for step_id, declared_deps in plan.composition.items():
            if step_id not in step_by_id:
                raise ValueError(f"Composition references undefined step ID: {step_id}")

            inferred_deps = set(inferred_composition.get(step_id, []))
            declared_deps_set = set(declared_deps) if isinstance(declared_deps, list) else set()

            if inferred_deps != declared_deps_set:
                raise ValueError(
                    f"Composition graph for step '{step_id}' is inconsistent. "
                    f"Declared: {declared_deps_set}, but inferred: {inferred_deps} "
                    f"based on inputs/outputs."
                )


def validate_all_steps_atomic(plan: Plan) -> None:
    """
    Validate atomicity for all steps in a plan.

    This is called after Plan schema validation to ensure all steps pass atomicity checks.

    Args:
        plan: The Plan to validate

    Raises:
        AtomicityViolationError: If any step is not atomic
    """
    for step in plan.steps:
        validate_step_atomicity(step)


# Export all validators
__all__ = [
    "validate_step_atomicity",
    "validate_plan_structure",
    "validate_all_steps_atomic",
    "AtomicityViolationError",
]
