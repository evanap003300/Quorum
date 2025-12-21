import os
import sys
from typing import Dict, Any, Optional

# Add orchestrator directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from planner.planner import plan
from solver.solver import solve_step
from planner.schema import StateObject, Plan


def solve_problem(problem: str) -> Dict[str, Any]:
    print(f"Problem: {problem}\n")
    
    # Step 1: Plan the problem
    print("="*60)
    print("PLANNING")
    print("="*60)
    
    try:
        state, plan_obj = plan(problem)
    except Exception as e:
        return {
            "success": False,
            "error": f"Planning failed: {e}",
            "final_answer": None,
            "final_unit": None,
            "state": None,
            "plan": None
        }
    
    print(f"✓ Created plan with {len(plan_obj.steps)} steps")
    print(f"  Domain: {state.domain}")
    print(f"  Approach: {plan_obj.approach}")
    print(f"  Target: {plan_obj.final_output}\n")
    
    # Step 2: Execute each step
    print("="*60)
    print("EXECUTION")
    print("="*60)
    
    for i, step in enumerate(plan_obj.steps, 1):
        print(f"\nStep {step.step_id}/{len(plan_obj.steps)}: {step.description}")
        print(f"  Operation: {step.operation}")
        print(f"  Output: {step.output} ({step.expected_unit})")
        
        # Solve this step
        success, value, unit, error = solve_step(step, state)
        
        if not success:
            print(f"  ✗ FAILED: {error}")
            return {
                "success": False,
                "error": f"Step {step.step_id} failed: {error}",
                "final_answer": None,
                "final_unit": None,
                "state": state,
                "plan": plan_obj,
                "failed_at_step": step.step_id
            }
        
        # Update state
        state.variables[step.output].value = value
        state.variables[step.output].unit = unit
        state.variables[step.output].source_step = step.step_id
        
        print(f"  ✓ {step.output} = {value} {unit}")
        
        # Check for unit mismatch
        if unit != step.expected_unit:
            print(f"  ⚠ WARNING: Expected unit {step.expected_unit}, got {unit}")
    
    # Step 3: Extract final answer
    print("\n" + "="*60)
    print("RESULT")
    print("="*60)
    
    final_var = state.variables[plan_obj.final_output]
    
    print(f"\nFinal answer: {plan_obj.final_output} = {final_var.value} {final_var.unit}")
    
    return {
        "success": True,
        "final_answer": final_var.value,
        "final_unit": final_var.unit,
        "state": state,
        "plan": plan_obj,
        "error": None
    }


def test_orchestrator():
    """Test the orchestrator on sample problems"""
    
    problems = [
        "Compare the energy output of a black-body radiator (such as an incandescent lamp) at two different wavelengths by calculating the ratio of the energy output at 450 nm (blue light) to that at 700 nm (red light) at 298 K. Use the Planck distribution."
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