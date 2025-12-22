"""
Simple integration tests for vision module and orchestrate image handling.
Tests basic functionality without requiring actual images or API calls.
"""

import sys
import os
from pathlib import Path

# Add parent directory (orchestrator) to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision import _validate_image, _parse_vision_response


def test_validate_image_missing_file():
    """Test that missing file raises FileNotFoundError"""
    try:
        _validate_image("/nonexistent/image.jpg")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        assert "not found" in str(e).lower()
        print("✓ test_validate_image_missing_file passed")


def test_validate_image_unsupported_format():
    """Test that unsupported formats raise ValueError"""
    # Create a temporary file with unsupported extension
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as f:
        temp_path = f.name

    try:
        _validate_image(temp_path)
        assert False, "Should have raised ValueError for unsupported format"
    except ValueError as e:
        assert "unsupported" in str(e).lower()
        print("✓ test_validate_image_unsupported_format passed")
    finally:
        os.remove(temp_path)


def test_parse_vision_response_well_formatted():
    """Test parsing well-formatted vision response"""
    response = """PROBLEM TEXT:
A car accelerates at 2 m/s² for 5 seconds. What is its final velocity?

DIAGRAM CONTEXT:
None"""

    problem_text, diagram_context = _parse_vision_response(response)

    assert "accelerates" in problem_text
    assert "2 m/s" in problem_text
    assert diagram_context == "None"
    print("✓ test_parse_vision_response_well_formatted passed")


def test_parse_vision_response_with_diagram():
    """Test parsing response with diagram context"""
    response = """PROBLEM TEXT:
Find the angle theta in a right triangle.

DIAGRAM CONTEXT:
Right triangle with sides labeled a, b, and c. The angle theta is at the base."""

    problem_text, diagram_context = _parse_vision_response(response)

    assert "angle theta" in problem_text
    assert "Right triangle" in diagram_context
    assert diagram_context != "None"
    print("✓ test_parse_vision_response_with_diagram passed")


def test_parse_vision_response_malformed():
    """Test that malformed response raises ValueError"""
    response = "This is not formatted correctly"

    try:
        _parse_vision_response(response)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid response format" in str(e).lower()
        print("✓ test_parse_vision_response_malformed passed")


def test_solve_problem_no_inputs():
    """Test that solve_problem raises error with no inputs"""
    from orchestrate import solve_problem

    try:
        solve_problem()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must provide either" in str(e).lower()
        print("✓ test_solve_problem_no_inputs passed")


def test_solve_problem_text_only():
    """Test that solve_problem accepts text input (backward compatibility)"""
    from orchestrate import solve_problem

    # This will fail on planning (no OpenRouter key), but tests the signature
    result = solve_problem(problem="Simple test problem")

    # Should return error dict (planning failed) but success should be False
    assert isinstance(result, dict)
    assert "success" in result
    assert "vision_cost" in result  # New field should be present
    assert result["vision_cost"] == 0.0  # No vision cost without image
    print("✓ test_solve_problem_text_only passed")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("VISION INTEGRATION TESTS")
    print("="*60 + "\n")

    tests = [
        test_validate_image_missing_file,
        test_validate_image_unsupported_format,
        test_parse_vision_response_well_formatted,
        test_parse_vision_response_with_diagram,
        test_parse_vision_response_malformed,
        test_solve_problem_no_inputs,
        test_solve_problem_text_only,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1

    print(f"\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
