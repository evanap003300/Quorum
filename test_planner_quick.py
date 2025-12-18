#!/usr/bin/env python3
"""
Quick test script for the Planner.

Run with: python test_planner_quick.py
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.orchestrator.planner.planner import Planner
from src.core.orchestrator.planner.schemas import AtomicStep, Plan, StateObject
from src.core.orchestrator.planner.validators import validate_step_atomicity


def test_schema_validation():
    """Test that schema validation works correctly."""
    print("\n" + "="*70)
    print("TEST 1: Schema Validation")
    print("="*70)

    # Test 1a: Valid atomic step
    try:
        step = AtomicStep(
            type="algebraic",
            description="Solve for acceleration using Newton's 2nd law",
            inputs=["F", "m"],
            output="a",
            expected_units="m/s^2",
            justification="F = ma, so a = F/m"
        )
        print("✅ Valid atomic step created successfully")
    except Exception as e:
        print(f"❌ Failed to create valid step: {e}")
        return False

    # Test 1b: Invalid step with compound operation
    try:
        bad_step = AtomicStep(
            type="algebraic",
            description="Solve for a and substitute into equation",
            inputs=["F", "m"],
            output="a",
            justification="Test"
        )
        print("❌ Compound step should have been rejected!")
        return False
    except ValueError as e:
        print(f"✅ Compound step correctly rejected: {str(e)[:80]}...")

    # Test 1c: Invalid step with multiple verbs
    try:
        bad_step = AtomicStep(
            type="algebraic",
            description="Differentiate the position to find velocity then acceleration",
            inputs=["x"],
            output="a",
            justification="Test"
        )
        print("❌ Multi-verb step should have been rejected!")
        return False
    except ValueError as e:
        print(f"✅ Multi-verb step correctly rejected: {str(e)[:80]}...")

    return True


def test_atomicity_validator():
    """Test the atomicity validator."""
    print("\n" + "="*70)
    print("TEST 2: Atomicity Validator")
    print("="*70)

    # Valid step
    try:
        step = AtomicStep(
            type="algebraic",
            description="Isolate v_f from kinematic equation v_f = v0 + at",
            inputs=["v0", "a", "t"],
            output="v_f",
            justification="Standard constant acceleration kinematic equation"
        )
        validate_step_atomicity(step)
        print("✅ Valid atomic step passed atomicity check")
    except Exception as e:
        print(f"❌ Valid step failed: {e}")
        return False

    # Invalid step with "and"
    try:
        bad_step = AtomicStep(
            type="algebraic",
            description="Compute acceleration and substitute velocity",
            inputs=["F", "m", "v0"],
            output="result",
            justification="Test"
        )
        validate_step_atomicity(bad_step)
        print("❌ 'and' step should have been rejected!")
        return False
    except ValueError as e:
        print(f"✅ 'and' step correctly rejected")

    return True


def test_state_object():
    """Test StateObject creation and audit log."""
    print("\n" + "="*70)
    print("TEST 3: State Object and Audit Log")
    print("="*70)

    try:
        state = StateObject(
            problem_text="Find the velocity of a falling object",
            domain="physics",
            goal="Find v_f"
        )
        print(f"✅ StateObject created with ID: {state.state_id}")

        # Test audit log
        state.log_event("test_event", "This is a test event")
        state.log_event("another_event", "Another test event")

        if len(state.audit_log) == 2:
            print(f"✅ Audit log works: {len(state.audit_log)} events logged")
        else:
            print(f"❌ Audit log has wrong number of events: {len(state.audit_log)}")
            return False

        return True
    except Exception as e:
        print(f"❌ Failed to create StateObject: {e}")
        return False


def test_planner_basic():
    """Test basic planner initialization."""
    print("\n" + "="*70)
    print("TEST 4: Planner Initialization")
    print("="*70)

    try:
        planner = Planner(
            model="openai/gpt-4o-mini",
            temperature=0.1,
            max_retries=2
        )
        print(f"✅ Planner initialized successfully")
        print(f"   Model: {planner.model}")
        print(f"   Temperature: {planner.temperature}")
        print(f"   Max Retries: {planner.max_retries}")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize Planner: {e}")
        return False


def test_planner_planning():
    """Test actual plan generation (calls LLM)."""
    print("\n" + "="*70)
    print("TEST 5: Plan Generation (Requires API Key)")
    print("="*70)

    try:
        planner = Planner(model="openai/gpt-4o-mini")

        problem = "A ball is thrown upward at 15 m/s. How high does it go?"
        print(f"\nGenerating plan for: {problem}")
        print("(This will call the LLM... please wait)")

        state = planner.create_plan(problem, domain="physics")

        print(f"\n✅ Plan generated successfully!")
        print(f"   Plan Version: {state.plan.plan_version}")
        print(f"   Number of Steps: {len(state.plan.steps)}")
        print(f"   Variables: {list(state.plan.variables.keys())}")
        print(f"   Assumptions: {state.plan.assumptions}")

        print(f"\n📋 Generated Steps:")
        for i, step in enumerate(state.plan.steps, 1):
            print(f"   Step {i}: {step.step_id}")
            print(f"      Description: {step.description}")
            print(f"      Type: {step.type}")
            print(f"      Inputs: {step.inputs} → Output: {step.output}")

        print(f"\n📝 Audit Trail ({len(state.audit_log)} events):")
        for event in state.audit_log:
            print(f"   [{event['event']}] {event['message'][:60]}")

        return True

    except Exception as e:
        print(f"❌ Plan generation failed: {e}")
        print(f"   (Make sure OPEN_ROUTER_KEY is set in .env)")
        return False


def main():
    """Run all tests."""
    print("\n" + "🧪 PLANNER TEST SUITE 🧪".center(70))
    print("="*70)

    results = {
        "Schema Validation": test_schema_validation(),
        "Atomicity Validator": test_atomicity_validator(),
        "State Object": test_state_object(),
        "Planner Init": test_planner_basic(),
        "Plan Generation": test_planner_planning(),
    }

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print("="*70)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
