import os
import sys
import time
import asyncio
import importlib.util
from typing import Dict, Any, Optional

# Add orchestrator directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from planner.planner import plan
from solver.solver import solve_step
from solver.swarm import solve_step_with_swarm
from solver.parsing import validate_result
from planner.schema import StateObject, Plan
from planner.critics.physics_lawyer import audit_plan
from planner.revisor import revise_plan
from vision import analyze_problem_image, analyze_problem_image_with_cv_tools
from tools.evaluation.code_interpreter import init_sandbox
# Import Sandbox from e2b
Sandbox = None

try:
    from e2b_code_interpreter import Sandbox
except Exception:
    # Silently fall back - system will work with per-step sandboxes
    pass


async def solve_problem(problem: str = "", image_path: Optional[str] = None, expected_unit: str = "") -> Dict[str, Any]:
    """Solve a problem using the multi-agent orchestrator (planner + swarm).

    Args:
        problem: Problem text to solve
        image_path: Optional path to problem image
        expected_unit: Expected output unit from ground truth (for unit hints)

    Returns:
        Result dictionary with solution and metadata
    """
    # Validate inputs
    if not problem and not image_path:
        raise ValueError("Must provide either 'problem' text or 'image_path'")

    problem_start_time = time.time()
    vision_cost = 0.0
    vision_time = 0.0
    intermediate_paths = []

    # Step 0: Vision analysis (if image provided)
    if image_path:
        print("="*60)
        print("VISION ANALYSIS (WITH CV TOOLS)")
        print("="*60)

        vision_start_time = time.time()
        try:
            problem_text, diagram_context, vision_cost, intermediate_paths = analyze_problem_image_with_cv_tools(image_path)
            vision_time = time.time() - vision_start_time

            print(f"Image: {image_path}")
            print(f"Extracted problem text: {problem_text[:100]}...")
            if diagram_context != "None":
                print(f"Diagram context: {diagram_context[:100]}...")

            # Log intermediate images for debugging
            if intermediate_paths:
                print(f"\n✓ Image preprocessing: {len(intermediate_paths)} steps")
                for i, path in enumerate(intermediate_paths, 1):
                    from pathlib import Path
                    print(f"  Step {i}: {Path(path).name}")

            print(f"\nVision analysis time: {vision_time:.2f}s | Cost: ${vision_cost:.4f}\n")

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

        # Initialize problem context for state
        state.problem_context = {}

        # Add image_path to state context for OBSERVE operations
        if image_path:
            state.problem_context["image_path"] = image_path

        # Add expected_unit to state context for solver prompts
        if expected_unit:
            state.problem_context["expected_unit"] = expected_unit
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

    # Step 1.5: Review plan with Physics Lawyer (optional - can be disabled)
    review_cost = 0.0
    review_time = 0.0
    skip_review = os.getenv("SKIP_PHYSICS_REVIEW", "false").lower() == "true"

    if skip_review:
        print(f"ℹ Physics review skipped (SKIP_PHYSICS_REVIEW=true)\n")
    else:
        print("="*60)
        print("PHYSICS REVIEW")
        print("="*60)

        plan_obj, review_cost, review_time = _review_and_revise_plan(problem, plan_obj, max_revisions=2)

    # Step 2: Execute each step
    print("="*60)
    print("EXECUTION")
    print("="*60)

    total_cost = 0.0
    sandbox = None

    # Note: Hot sandbox disabled - e2b API has changed
    # Using per-step sandboxes instead (each step gets its own)
    sandbox = None
    print(f"ℹ Using per-step sandboxes")
    print(f"  (Hot sandbox API changed in e2b - upgrade available for speedup)\n")

    # Start timing execution after sandbox is ready
    execution_start_time = time.time()

    for i, step in enumerate(plan_obj.steps, 1):
        outputs = step.get_outputs()
        output_info = ", ".join(outputs) if len(outputs) > 1 else step.output
        expected_unit_info = str(step.expected_units) if len(outputs) > 1 else step.expected_unit

        print(f"\nStep {step.step_id}/{len(plan_obj.steps)}: {step.description}")
        print(f"  Operation: {step.operation}")
        print(f"  Output: {output_info} ({expected_unit_info})")

        step_start_time = time.time()

        # K-AHEAD SWARM WITH RETRY: Execute swarm up to 3 times
        max_swarm_attempts = 3
        swarm_attempt = 0
        last_swarm_error = None
        success = False

        while swarm_attempt < max_swarm_attempts:
            swarm_attempt += 1

            # Execute swarm (k=3 parallel agents with majority voting)
            # Pass previous error to enable agents to learn from failures
            success, value, unit, error, cost = await solve_step_with_swarm(
                step, state, sandbox, k=3, previous_error=last_swarm_error
            )
            total_cost += cost

            if success:
                if swarm_attempt > 1:
                    print(f"  ✓ RECOVERED: Swarm succeeded on attempt {swarm_attempt}/{max_swarm_attempts}")
                break
            else:
                last_swarm_error = error
                if swarm_attempt < max_swarm_attempts:
                    print(f"  ↻ Swarm attempt {swarm_attempt}/{max_swarm_attempts} failed, retrying...")

        # 4.3: Single agent fallback when swarm completely fails
        if not success:
            print(f"  ⚠ Swarm failed after {max_swarm_attempts} attempts, trying single agent fallback...")
            from solver.solver import solve_step
            try:
                fallback_success, fallback_value, fallback_unit, fallback_error, fallback_cost = await solve_step(
                    step, state, sandbox,
                    error_context=f"Previous swarm attempts failed with: {last_swarm_error}"
                )
                total_cost += fallback_cost
                if fallback_success:
                    success = True
                    value = fallback_value
                    unit = fallback_unit
                    error = None
                    print(f"  ✓ RECOVERED: Single agent fallback succeeded!")
                else:
                    last_swarm_error = f"Single agent fallback also failed: {fallback_error}"
            except Exception as e:
                print(f"  ✗ Single agent fallback exception: {e}")
                last_swarm_error = f"Swarm and single agent both failed. Last error: {last_swarm_error}"

        step_time = time.time() - step_start_time

        if not success:
            print(f"  ✗ FAILED: {last_swarm_error}")
            # Cleanup sandbox before returning
            if sandbox:
                try:
                    sandbox.kill()
                except:
                    pass
            return {
                "success": False,
                "error": f"Step {step.step_id} failed after {max_swarm_attempts} swarm attempts: {last_swarm_error}",
                "final_answer": None,
                "final_unit": None,
                "state": state,
                "plan": plan_obj,
                "failed_at_step": step.step_id,
                "total_time": time.time() - problem_start_time,
                "plan_time": plan_time,
                "review_time": review_time,
                "execution_time": time.time() - execution_start_time,
                "plan_cost": plan_cost,
                "review_cost": review_cost,
                "execution_cost": total_cost,
                "vision_cost": vision_cost,
                "total_cost": plan_cost + review_cost + total_cost + vision_cost
            }

        # Update state - handle both single and multiple outputs
        if len(outputs) > 1:
            # Multi-output step - validate that solver returned dicts
            if not (isinstance(value, dict) and isinstance(unit, dict)):
                error_msg = f"Multi-output step expected dicts but got value={type(value).__name__}, unit={type(unit).__name__}"
                print(f"  ✗ FAILED: {error_msg}")
                if sandbox:
                    try:
                        sandbox.kill()
                    except:
                        pass
                return {
                    "success": False,
                    "error": f"Step {step.step_id} failed: {error_msg}",
                    "final_answer": None,
                    "final_unit": None,
                    "state": state,
                    "plan": plan_obj,
                    "failed_at_step": step.step_id,
                    "total_time": time.time() - problem_start_time,
                    "plan_time": plan_time,
                    "review_time": review_time,
                    "execution_time": time.time() - execution_start_time,
                    "plan_cost": plan_cost,
                    "review_cost": review_cost,
                    "execution_cost": total_cost,
                    "vision_cost": vision_cost,
                    "total_cost": plan_cost + review_cost + total_cost + vision_cost
                }
            print(f"DEBUG: Multi-output step received:")
            print(f"  Expected variables: {outputs}")
            print(f"  Received keys: {list(value.keys())}")
            print(f"  Values: {value}")
            print(f"  Units: {unit}")

            for var_name in outputs:
                if var_name in value and var_name in unit:
                    # Only update if value is not None
                    if value[var_name] is not None:
                        # Validate result before updating state
                        is_valid, error_msg = validate_result(value[var_name], unit[var_name], step, state)
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
                                "review_time": review_time,
                                "execution_time": time.time() - execution_start_time,
                                "plan_cost": plan_cost,
                                "review_cost": review_cost,
                                "execution_cost": total_cost,
                                "vision_cost": vision_cost,
                                "total_cost": plan_cost + review_cost + total_cost + vision_cost
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
                        print(f"  ⚠ {var_name} = None (missing from OBSERVE response)")
                else:
                    print(f"  ⚠ {var_name} missing from response dicts")
        else:
            # Single-output step
            var_name = outputs[0]

            # Validate result before updating state
            is_valid, error_msg = validate_result(value, unit, step, state)
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
                    "review_time": review_time,
                    "execution_time": time.time() - execution_start_time,
                    "plan_cost": plan_cost,
                    "review_cost": review_cost,
                    "execution_cost": total_cost,
                    "vision_cost": vision_cost,
                    "total_cost": plan_cost + review_cost + total_cost + vision_cost
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
    print(f"Physics review:   {review_time:.2f}s     | Cost: ${review_cost:.4f}")
    print(f"Execution time:   {execution_time:.2f}s    | Cost: ${total_cost:.4f}")
    print(f"Total time:       {total_time:.2f}s")
    print(f"Total cost:       ${plan_cost + review_cost + total_cost + vision_cost:.4f}")
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
        "review_time": review_time,
        "execution_time": execution_time,
        "plan_cost": plan_cost,
        "review_cost": review_cost,
        "execution_cost": total_cost,
        "vision_cost": vision_cost,
        "total_cost": plan_cost + review_cost + total_cost + vision_cost
    }



def _review_and_revise_plan(problem: str, plan_obj: Plan, max_revisions: int = 2):
    """
    Review a plan with the Physics Lawyer and revise if needed.

    Args:
        problem: The original problem statement
        plan_obj: The plan to review
        max_revisions: Maximum number of revision attempts

    Returns:
        Tuple of (approved or best-effort plan, total review cost, total review time)
    """
    review_start_time = time.time()
    total_review_cost = 0.0

    for attempt in range(max_revisions):
        print(f"⚖️ Physics Lawyer reviewing plan (Pass {attempt + 1}/{max_revisions})...")

        # Audit the plan
        lawyer_start_time = time.time()
        audit, lawyer_cost = audit_plan(problem, plan_obj)
        lawyer_time = time.time() - lawyer_start_time
        total_review_cost += lawyer_cost

        if audit.is_approved:
            print(f"✅ Plan approved by Physics Lawyer.")
            if audit.reasoning:
                print(f"   Reasoning: {audit.reasoning}")
            print(f"   Time: {lawyer_time:.2f}s | Cost: ${lawyer_cost:.4f}\n")
            total_review_time = time.time() - review_start_time
            return plan_obj, total_review_cost, total_review_time

        # Plan was rejected - show critiques
        print(f"❌ Physics Lawyer found {len(audit.critiques)} issue(s):")
        for critique in audit.critiques:
            severity_icon = "🔴" if critique.get("severity") == "BLOCKING" else "🟡"
            step_idx = critique.get("step_index", "?")
            error = critique.get("error", "Unknown error")
            correction = critique.get("correction", "No correction suggested")
            print(f"   {severity_icon} Step {step_idx}: {error}")
            print(f"      → Fix: {correction}")
        print(f"   Time: {lawyer_time:.2f}s | Cost: ${lawyer_cost:.4f}")

        # If this is the last attempt, proceed with warning
        if attempt == max_revisions - 1:
            print(f"⚠️ Max revisions reached. Proceeding with best-effort plan.\n")
            total_review_time = time.time() - review_start_time
            return plan_obj, total_review_cost, total_review_time

        # Otherwise, revise the plan
        print(f"\n🔧 Revisor is patching the plan...")
        try:
            revisor_start_time = time.time()
            plan_obj, revisor_cost = revise_plan(problem, plan_obj, audit.critiques)
            revisor_time = time.time() - revisor_start_time
            total_review_cost += revisor_cost
            print(f"✓ Plan revised.")
            print(f"   Time: {revisor_time:.2f}s | Cost: ${revisor_cost:.4f}")
            print(f"Retrying audit...\n")
        except Exception as e:
            print(f"⚠️ Revision failed: {e}")
            print(f"   Proceeding with original plan.\n")
            total_review_time = time.time() - review_start_time
            return plan_obj, total_review_cost, total_review_time

    total_review_time = time.time() - review_start_time
    return plan_obj, total_review_cost, total_review_time


def test_orchestrator():
    """Test the orchestrator on sample problems"""

    # Get absolute path to test image
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vector_field_path = os.path.join(current_dir, 'vector_field_image.png')

    problems = [
        """You find an apple core of height h. What volume of apple was eaten? (In this
problem, an apple is a perfect sphere, and the height of the core is the height of
the cylindrical part of its boundary.)"""
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