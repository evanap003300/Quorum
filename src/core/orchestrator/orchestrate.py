import os
import sys
import time
import asyncio
import importlib.util
from typing import Dict, Any, Optional

# Add orchestrator directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from planner.planner import plan
from solver.solver import solve_step, _validate_result
from planner.schema import StateObject, Plan
from vision import analyze_problem_image

# Import Sandbox and init_sandbox using importlib (handles filename with dash)
Sandbox = None
init_sandbox = None

try:
    from e2b_code_interpreter import Sandbox

    # Use importlib to handle python_interpreter-e2b (has dash, can't be imported normally)
    spec = importlib.util.spec_from_file_location(
        "python_interpreter_e2b",
        os.path.join(os.path.dirname(__file__), "solver", "python_interpreter-e2b", "main.py")
    )
    if spec and spec.loader:
        python_interpreter_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(python_interpreter_module)
        init_sandbox = getattr(python_interpreter_module, 'init_sandbox', None)
except Exception:
    # Silently fall back - system will work with per-step sandboxes
    pass


def solve_problem(problem: str = "", image_path: Optional[str] = None) -> Dict[str, Any]:
    # Validate inputs
    if not problem and not image_path:
        raise ValueError("Must provide either 'problem' text or 'image_path'")

    problem_start_time = time.time()
    vision_cost = 0.0
    vision_time = 0.0

    # Step 0: Vision analysis (if image provided)
    if image_path:
        print("="*60)
        print("VISION ANALYSIS")
        print("="*60)

        vision_start_time = time.time()
        try:
            problem_text, diagram_context, vision_cost = analyze_problem_image(image_path)
            vision_time = time.time() - vision_start_time

            print(f"Image: {image_path}")
            print(f"Extracted problem text: {problem_text[:100]}...")
            if diagram_context != "None":
                print(f"Diagram context: {diagram_context[:100]}...")
            print(f"Vision analysis time: {vision_time:.2f}s | Cost: ${vision_cost:.4f}\n")

            # Merge with provided text (if any)
            if problem:
                # If both provided, combine them
                enriched_problem = f"{problem}\n\nFrom image:\n{problem_text}"
                if diagram_context != "None":
                    enriched_problem += f"\n\nDiagram context: {diagram_context}"
                problem = enriched_problem
            else:
                # Use extracted text
                problem = problem_text
                if diagram_context != "None":
                    problem += f"\n\nDiagram context: {diagram_context}"

        except Exception as e:
            return {
                "success": False,
                "error": f"Vision analysis failed: {e}",
                "final_answer": None,
                "final_unit": None,
                "state": None,
                "plan": None,
                "total_time": time.time() - problem_start_time,
                "plan_time": 0,
                "execution_time": 0,
                "plan_cost": 0,
                "execution_cost": 0,
                "vision_cost": vision_cost,
                "total_cost": vision_cost
            }

    print(f"Problem: {problem}\n")

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
            "execution_cost": 0,
            "vision_cost": vision_cost,
            "total_cost": vision_cost
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
    sandbox = None

    # Create and initialize hot sandbox if available
    if Sandbox is None or init_sandbox is None:
        print(f"⚠ Hot sandbox unavailable (Sandbox={Sandbox is not None}, init={init_sandbox is not None})")
        print(f"  Using per-step sandboxes (slower - ~20s per step)\n")
        sandbox = None
    else:
        try:
            sandbox = Sandbox()
            asyncio.run(init_sandbox(sandbox))
            print(f"✓ Created hot sandbox - reusing across all steps")
            print(f"  Expected speedup: 60-90% faster per step\n")
        except Exception as e:
            print(f"⚠ Hot sandbox creation failed: {e}")
            print(f"  Falling back to per-step sandboxes (slower - ~20s per step)\n")
            if sandbox:
                try:
                    sandbox.kill()
                except:
                    pass
            sandbox = None

    for i, step in enumerate(plan_obj.steps, 1):
        outputs = step.get_outputs()
        output_info = ", ".join(outputs) if len(outputs) > 1 else step.output
        expected_unit_info = str(step.expected_units) if len(outputs) > 1 else step.expected_unit

        print(f"\nStep {step.step_id}/{len(plan_obj.steps)}: {step.description}")
        print(f"  Operation: {step.operation}")
        print(f"  Output: {output_info} ({expected_unit_info})")

        step_start_time = time.time()
        # Solve this step (passing hot sandbox)
        success, value, unit, error, cost = solve_step(step, state, sandbox)
        step_time = time.time() - step_start_time
        total_cost += cost

        if not success:
            print(f"  ✗ FAILED: {error}")
            # Cleanup sandbox before returning
            if sandbox:
                try:
                    sandbox.kill()
                except:
                    pass
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
                "vision_cost": vision_cost,
                "total_cost": plan_cost + total_cost + vision_cost
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
                        # Cleanup sandbox before returning
                        if sandbox:
                            try:
                                sandbox.kill()
                            except:
                                pass
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
                            "vision_cost": vision_cost,
                            "total_cost": plan_cost + total_cost + vision_cost
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
                # Cleanup sandbox before returning
                if sandbox:
                    try:
                        sandbox.kill()
                    except:
                        pass
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
                    "vision_cost": vision_cost,
                    "total_cost": plan_cost + total_cost + vision_cost
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

    # Cleanup hot sandbox after all steps complete
    if sandbox:
        try:
            sandbox.kill()
        except:
            pass

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
    if vision_cost > 0:
        print(f"Vision analysis:  {vision_time:.2f}s     | Cost: ${vision_cost:.4f}")
    print(f"Planning time:    {plan_time:.2f}s     | Cost: ${plan_cost:.4f}")
    print(f"Execution time:   {execution_time:.2f}s    | Cost: ${total_cost:.4f}")
    print(f"Total time:       {total_time:.2f}s")
    print(f"Total cost:       ${plan_cost + total_cost + vision_cost:.4f}")
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
        "vision_cost": vision_cost,
        "total_cost": plan_cost + total_cost + vision_cost
    }


def test_orchestrator():
    """Test the orchestrator on sample problems"""

    # Get absolute path to test image
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_image_path = os.path.join(current_dir, 'test_image.png')

    problems = [
        """A block with large mass $M$ slides with speed $V_0$ on a frictionless table towards a wall. It collides elastically with a ball with small mass $m$, which is initially at rest at a distance $L$ from the wall. The ball slides towards the wall, bounces elastically, and then proceeds to bounce back and forth between the block and the wall.

[Diagram Description: A block labeled $M$ moves to the right with velocity $V_0$ toward a small dot labeled $m$. To the right is a vertical wall. The distance between the objects and the wall is labeled $L$.]

(a) How close does the block come to the wall?

(b) How many times does the ball bounce off the block, by the time the block makes its closest approach to the wall?

Assume that $M \\gg m$, and give your answers to leading order in $m/M$."""
    ]
    
    for i, problem in enumerate(problems, 1):
        print("\n" + "="*70)
        print(f"PROBLEM {i}")
        print("="*70)
        
        result = solve_problem(problem, image_path=test_image_path)
        
        if result["success"]:
            print(f"\n✓ SUCCESS: {result['final_answer']} {result['final_unit']}")
        else:
            print(f"\n✗ FAILED: {result['error']}")
        
        print("\n")


if __name__ == "__main__":
    test_orchestrator()