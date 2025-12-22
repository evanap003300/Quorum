import os
import sys
import time
from typing import Dict, Any, Optional

# Add orchestrator directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from planner.planner import plan
from solver.solver import solve_step, _validate_result
from planner.schema import StateObject, Plan


def solve_problem(problem: str) -> Dict[str, Any]:
    print(f"Problem: {problem}\n")

    problem_start_time = time.time()

    # Step 1: Plan the problem
    print("="*60)
    print("PLANNING")
    print("="*60)

    plan_start_time = time.time()
    try:
        state, plan_obj, plan_cost = plan(problem)
    except Exception as e:
        return {
            "success": False,
            "error": f"Planning failed: {e}",
            "final_answer": None,
            "final_unit": None,
            "state": None,
            "plan": None,
            "total_time": time.time() - problem_start_time,
            "plan_time": 0,
            "execution_time": 0,
            "plan_cost": 0,
            "total_cost": 0
        }

    plan_time = time.time() - plan_start_time

    print(f"✓ Created plan with {len(plan_obj.steps)} steps")
    print(f"  Domain: {state.domain}")
    print(f"  Approach: {plan_obj.approach}")
    print(f"  Target: {plan_obj.final_output}")
    print(f"  Planning time: {plan_time:.2f}s | Cost: ${plan_cost:.4f}\n")

    # Step 2: Execute each step
    print("="*60)
    print("EXECUTION")
    print("="*60)

    execution_start_time = time.time()
    total_cost = 0.0

    for i, step in enumerate(plan_obj.steps, 1):
        outputs = step.get_outputs()
        output_info = ", ".join(outputs) if len(outputs) > 1 else step.output
        expected_unit_info = str(step.expected_units) if len(outputs) > 1 else step.expected_unit

        print(f"\nStep {step.step_id}/{len(plan_obj.steps)}: {step.description}")
        print(f"  Operation: {step.operation}")
        print(f"  Output: {output_info} ({expected_unit_info})")

        step_start_time = time.time()
        # Solve this step
        success, value, unit, error, cost = solve_step(step, state)
        step_time = time.time() - step_start_time
        total_cost += cost

        if not success:
            print(f"  ✗ FAILED: {error}")
            return {
                "success": False,
                "error": f"Step {step.step_id} failed: {error}",
                "final_answer": None,
                "final_unit": None,
                "state": state,
                "plan": plan_obj,
                "failed_at_step": step.step_id,
                "total_time": time.time() - problem_start_time,
                "plan_time": plan_time,
                "execution_time": time.time() - execution_start_time,
                "plan_cost": plan_cost,
                "execution_cost": total_cost,
                "total_cost": plan_cost + total_cost
            }

        # Update state - handle both single and multiple outputs
        if len(outputs) > 1:
            # Multi-output step
            assert isinstance(value, dict) and isinstance(unit, dict), "Multi-output step should return dicts"
            for var_name in outputs:
                if var_name in value and var_name in unit:
                    # Validate result before updating state
                    is_valid, error_msg = _validate_result(value[var_name], unit[var_name], step, state)
                    if not is_valid:
                        print(f"  ✗ VALIDATION FAILED: {error_msg}")
                        return {
                            "success": False,
                            "error": f"Step {step.step_id} failed validation: {error_msg}",
                            "final_answer": None,
                            "final_unit": None,
                            "state": state,
                            "plan": plan_obj,
                            "failed_at_step": step.step_id,
                            "total_time": time.time() - problem_start_time,
                            "plan_time": plan_time,
                            "execution_time": time.time() - execution_start_time,
                            "plan_cost": plan_cost,
                            "execution_cost": total_cost,
                            "total_cost": plan_cost + total_cost
                        }

                    state.variables[var_name].value = value[var_name]
                    state.variables[var_name].unit = unit[var_name]
                    state.variables[var_name].source_step = step.step_id
                    print(f"  ✓ {var_name} = {value[var_name]} {unit[var_name]}")

                    # Check for unit mismatch
                    expected_unit = step.get_unit(var_name)
                    if unit[var_name] != expected_unit:
                        print(f"    ⚠ WARNING: Expected unit {expected_unit}, got {unit[var_name]}")
        else:
            # Single-output step
            var_name = outputs[0]

            # Validate result before updating state
            is_valid, error_msg = _validate_result(value, unit, step, state)
            if not is_valid:
                print(f"  ✗ VALIDATION FAILED: {error_msg}")
                return {
                    "success": False,
                    "error": f"Step {step.step_id} failed validation: {error_msg}",
                    "final_answer": None,
                    "final_unit": None,
                    "state": state,
                    "plan": plan_obj,
                    "failed_at_step": step.step_id,
                    "total_time": time.time() - problem_start_time,
                    "plan_time": plan_time,
                    "execution_time": time.time() - execution_start_time,
                    "plan_cost": plan_cost,
                    "execution_cost": total_cost,
                    "total_cost": plan_cost + total_cost
                }

            state.variables[var_name].value = value
            state.variables[var_name].unit = unit
            state.variables[var_name].source_step = step.step_id
            print(f"  ✓ {var_name} = {value} {unit}")

            # Check for unit mismatch
            if unit != step.expected_unit:
                print(f"    ⚠ WARNING: Expected unit {step.expected_unit}, got {unit}")

        print(f"    Time: {step_time:.2f}s | Cost: ${cost:.4f}")

    execution_time = time.time() - execution_start_time

    # Step 3: Extract final answer
    print("\n" + "="*60)
    print("RESULT")
    print("="*60)

    final_var = state.variables[plan_obj.final_output]

    total_time = time.time() - problem_start_time

    print(f"\nFinal answer: {plan_obj.final_output} = {final_var.value} {final_var.unit}")
    print(f"\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    print(f"Planning time:    {plan_time:.2f}s     | Cost: ${plan_cost:.4f}")
    print(f"Execution time:   {execution_time:.2f}s    | Cost: ${total_cost:.4f}")
    print(f"Total time:       {total_time:.2f}s")
    print(f"Total cost:       ${plan_cost + total_cost:.4f}")
    print(f"Cost per step:    ${total_cost / len(plan_obj.steps):.4f}")

    return {
        "success": True,
        "final_answer": final_var.value,
        "final_unit": final_var.unit,
        "state": state,
        "plan": plan_obj,
        "error": None,
        "total_time": total_time,
        "plan_time": plan_time,
        "execution_time": execution_time,
        "plan_cost": plan_cost,
        "execution_cost": total_cost,
        "total_cost": plan_cost + total_cost
    }


def test_orchestrator():
    """Test the orchestrator on sample problems"""
    
    problems = [
        """A ball is thrown at speed v from zero height on level ground. At what angle should
it be thrown so that the distance traveled through the air is maximum. (You will
have to solve something numerically.)"""
    ]
    
    for i, problem in enumerate(problems, 1):
        print("\n" + "="*70)
        print(f"PROBLEM {i}")
        print("="*70)
        
        result = solve_problem(problem)
        
        if result["success"]:
            print(f"\n✓ SUCCESS: {result['final_answer']} {result['final_unit']}")
        else:
            print(f"\n✗ FAILED: {result['error']}")
        
        print("\n")


if __name__ == "__main__":
    test_orchestrator()