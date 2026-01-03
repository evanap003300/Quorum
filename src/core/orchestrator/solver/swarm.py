"""K-Ahead Swarm execution with majority voting for parallel step execution."""

import asyncio
from typing import List, Tuple, Union, Optional
from copy import deepcopy
import re

from planner.schema import Step, StateObject

try:
    from e2b_code_interpreter import Sandbox
except ImportError:
    Sandbox = None


async def solve_step_with_swarm(
    step: Step,
    state: StateObject,
    sandbox: Optional["Sandbox"] = None,
    k: int = 3,
    previous_error: Optional[str] = None,
) -> Tuple[bool, Optional[Union[float, str, dict]], Optional[Union[str, dict]], Optional[str], float]:
    """
    Execute step with K parallel attempts and majority voting.

    Runs k independent solver instances in parallel, each with an isolated state copy.
    Results are voted on using numeric proximity (1% tolerance) to select the most
    common result. This provides robustness against syntax errors and random hallucinations.

    Args:
        step: The step to execute
        state: Current state (will be cloned for each execution)
        sandbox: Optional sandbox to reuse
        k: Number of parallel executions (default 3)
        previous_error: Error from previous failed attempt (enables learning from failures)

    Returns:
        Tuple of (success, value, unit, error, cost)
        Same format as solve_step: (bool, Optional[value], Optional[unit], Optional[str], float)
    """
    # Import here to avoid circular imports
    from solver.solver import solve_step

    print(f"  🐝 Launching Swarm (k={k})...")

    # 1. Create k independent executions with isolated state copies
    tasks = []
    for i in range(k):
        state_copy = deepcopy(state)  # Isolate state for this execution
        # Pass error context from previous failed attempts to help agents learn
        task = solve_step(step, state_copy, sandbox, error_context=previous_error)
        tasks.append(task)

    # 2. Run all k executions in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 3. Filter valid results (success=True, no exceptions)
    valid_results = []
    total_cost = 0.0

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"    Agent {i+1}/{k}: ❌ Exception - {type(result).__name__}: {str(result)[:50]}")
            continue

        success, value, unit, error, cost = result
        total_cost += cost

        if success and value is not None:
            valid_results.append((value, unit, cost))
            print(f"    Agent {i+1}/{k}: ✓ Success - {value} {unit}")
        else:
            print(f"    Agent {i+1}/{k}: ❌ Failed - {error}")

    # 4. Check if we have any valid results
    if not valid_results:
        return False, None, None, "Swarm failed: All agents crashed or failed", total_cost

    # 5. Perform majority voting (only if k > 1)
    if k == 1 or len(valid_results) == 1:
        # Single execution or single valid result - just return it
        winning_value, winning_unit, _ = valid_results[0]
        return True, winning_value, winning_unit, None, total_cost

    winning_value, winning_unit, vote_count = _majority_vote(valid_results, len(valid_results))

    print(f"  🗳️  Swarm Consensus: {vote_count}/{len(valid_results)} agents agreed on {winning_value} {winning_unit}")

    return True, winning_value, winning_unit, None, total_cost


def _majority_vote(
    results: List[Tuple[Union[float, str, dict], Union[str, dict], float]],
    total_agents: int,
) -> Tuple[Union[float, str, dict], Union[str, dict], int]:
    """
    Perform majority voting on results.

    Groups numeric values within 1% tolerance and picks the most common.
    For multi-output (dict) results, returns the first valid result.

    Args:
        results: List of (value, unit, cost) tuples from successful executions
        total_agents: Total number of agents (for reporting)

    Returns:
        Tuple of (winning_value, winning_unit, vote_count)
    """
    # Handle multi-output case (dicts) - TODO: implement per-variable voting
    if results and isinstance(results[0][0], dict):
        # For multi-output, just return first result for now
        # This could be improved with per-variable voting
        return results[0][0], results[0][1], 1

    # Extract numeric values for voting
    numeric_results = []
    for value, unit, cost in results:
        try:
            num = _extract_number(value)
            if num is not None:
                numeric_results.append((num, value, unit))
        except Exception:
            continue

    if not numeric_results:
        # Fall back to first result if no numeric extraction possible
        return results[0][0], results[0][1], 1

    # Group by proximity (1% tolerance)
    groups = _group_by_proximity(numeric_results, tolerance=0.01)

    # Find largest group (majority)
    largest_group = max(groups, key=len)
    vote_count = len(largest_group)

    # Return the first result from the winning group
    _, winning_value, winning_unit = largest_group[0]

    return winning_value, winning_unit, vote_count


def _extract_number(value: Union[float, str]) -> Optional[float]:
    """
    Extract numeric value from various formats for voting.

    Handles floats, strings with numbers, and simple expressions.

    Args:
        value: Value to extract number from

    Returns:
        Float value if extraction succeeds, None otherwise
    """
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        # Try direct parsing first
        try:
            return float(value)
        except ValueError:
            pass

        # Try regex extraction - look for first number pattern
        match = re.search(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', value)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass

    return None


def _group_by_proximity(
    results: List[Tuple[float, Union[float, str], Union[str, dict]]],
    tolerance: float = 0.01,
) -> List[List[Tuple[float, Union[float, str], Union[str, dict]]]]:
    """
    Group numeric results by proximity (1% relative tolerance).

    Values that differ by less than 'tolerance' (relative) are grouped together.
    Handles edge case of values near zero.

    Args:
        results: List of (numeric_value, original_value, unit) tuples
        tolerance: Relative tolerance for grouping (0.01 = 1%)

    Returns:
        List of groups, each group is a list of similar results
    """
    groups = []

    for num, value, unit in results:
        # Try to find existing group within tolerance
        placed = False

        for group in groups:
            group_num = group[0][0]  # Representative value from group

            # Check relative difference
            if abs(group_num) < 1e-10:  # Handle near-zero values
                if abs(num) < 1e-10:  # Both near zero - same group
                    group.append((num, value, unit))
                    placed = True
                    break
            else:
                # Calculate relative difference
                rel_diff = abs(num - group_num) / abs(group_num)
                if rel_diff <= tolerance:
                    group.append((num, value, unit))
                    placed = True
                    break

        if not placed:
            # Create new group for this value
            groups.append([(num, value, unit)])

    return groups
