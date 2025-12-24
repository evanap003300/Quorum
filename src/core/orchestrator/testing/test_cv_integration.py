#!/usr/bin/env python3
"""
Integration test for CV tools in vision system.
Verifies that all modules import correctly and tool definitions are valid.
"""

import sys
import os

# Add parent directory (orchestrator) to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_imports():
    """Test that all required modules import successfully."""
    print("Testing imports...")

    try:
        from vision import (
            analyze_problem_image,
            analyze_problem_image_with_cv_tools,
            CV_TOOLS
        )
        print("✓ vision module imports successfully")
    except ImportError as e:
        print(f"✗ Failed to import vision: {e}")
        return False

    try:
        from prompts.vision import VISION_PROMPT, ENHANCED_VISION_PROMPT
        print("✓ vision prompts import successfully")
    except ImportError as e:
        print(f"✗ Failed to import vision prompts: {e}")
        return False

    try:
        from tools.vision import (
            get_image_metadata,
            detect_content_regions,
            detect_shadows_and_artifacts,
            apply_grid,
            crop_grid_square,
            crop_quadrant,
            binarize_image,
            enhance_clarity
        )
        print("✓ All 8 CV tools import successfully")
    except ImportError as e:
        print(f"✗ Failed to import CV tools: {e}")
        return False

    return True


def test_cv_tools_schema():
    """Test that CV tools are properly defined."""
    print("\nTesting CV tools schema...")

    try:
        from vision import CV_TOOLS
    except ImportError as e:
        print(f"✗ Failed to import CV_TOOLS: {e}")
        return False

    if not isinstance(CV_TOOLS, list):
        print(f"✗ CV_TOOLS is not a list: {type(CV_TOOLS)}")
        return False

    if len(CV_TOOLS) != 8:
        print(f"✗ Expected 8 CV tools, got {len(CV_TOOLS)}")
        return False

    tool_names = set()
    for tool_def in CV_TOOLS:
        if not isinstance(tool_def, dict):
            print(f"✗ Tool definition is not a dict: {tool_def}")
            return False

        if tool_def.get("type") != "function":
            print(f"✗ Tool type is not 'function': {tool_def.get('type')}")
            return False

        func_name = tool_def.get("function", {}).get("name")
        if not func_name:
            print(f"✗ Tool has no name: {tool_def}")
            return False

        tool_names.add(func_name)

    expected_tools = {
        "get_image_metadata",
        "detect_content_regions",
        "detect_shadows_and_artifacts",
        "apply_grid",
        "crop_grid_square",
        "crop_quadrant",
        "binarize_image",
        "enhance_clarity"
    }

    if tool_names != expected_tools:
        print(f"✗ Tool names mismatch.")
        print(f"  Expected: {expected_tools}")
        print(f"  Got: {tool_names}")
        return False

    print(f"✓ All 8 CV tools are properly defined")
    return True


def test_prompts():
    """Test that prompts are properly defined."""
    print("\nTesting prompts...")

    try:
        from prompts.vision import VISION_PROMPT, ENHANCED_VISION_PROMPT
    except ImportError as e:
        print(f"✗ Failed to import prompts: {e}")
        return False

    if not isinstance(VISION_PROMPT, str) or len(VISION_PROMPT) == 0:
        print(f"✗ VISION_PROMPT is not a valid string")
        return False

    if not isinstance(ENHANCED_VISION_PROMPT, str) or len(ENHANCED_VISION_PROMPT) == 0:
        print(f"✗ ENHANCED_VISION_PROMPT is not a valid string")
        return False

    # Check that enhanced prompt contains tool references
    tool_names = [
        "get_image_metadata",
        "detect_content_regions",
        "apply_grid",
        "crop_grid_square",
        "binarize_image",
        "enhance_clarity"
    ]

    for tool_name in tool_names:
        if tool_name not in ENHANCED_VISION_PROMPT:
            print(f"✗ ENHANCED_VISION_PROMPT doesn't mention {tool_name}")
            return False

    print(f"✓ Both prompts are valid and complete")
    return True


def test_functions_exist():
    """Test that new functions exist and have correct signatures."""
    print("\nTesting function signatures...")

    try:
        from vision import analyze_problem_image_with_cv_tools
        import inspect

        sig = inspect.signature(analyze_problem_image_with_cv_tools)
        params = list(sig.parameters.keys())

        expected_params = ["image_path", "max_iterations", "save_intermediates"]
        if params != expected_params:
            print(f"✗ Function parameters mismatch")
            print(f"  Expected: {expected_params}")
            print(f"  Got: {params}")
            return False

        print(f"✓ analyze_problem_image_with_cv_tools has correct signature")
    except Exception as e:
        print(f"✗ Error testing function signature: {e}")
        return False

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("CV TOOLS INTEGRATION TEST")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("CV Tools Schema", test_cv_tools_schema),
        ("Prompts", test_prompts),
        ("Function Signatures", test_functions_exist),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All integration tests passed!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
