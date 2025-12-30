#!/usr/bin/env python3
"""Test the robust number extraction for edge cases."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from benchmarking.evaluation.answer_comparator import extract_number

# Test cases: (input, expected_output, description)
test_cases = [
    (10.5, 10.5, "Direct float"),
    ("10.5", 10.5, "String number"),
    ("1.5e-10", 1.5e-10, "Scientific notation lowercase"),
    ("3E8", 3e8, "Scientific notation uppercase"),
    ("1,000.5", 1000.5, "Comma separator"),
    ("1_000_000", 1000000.0, "Underscore separator"),
    ("The answer is 42", 42.0, "Embedded in text"),
    ("acos(2/3)", 0.8410686705, "Symbolic expression acos"),
    ("sqrt(2)", 1.41421356, "Square root"),
    ("pi/4", 0.7853981, "Pi division"),
    ("15%", 0.15, "Percentage"),
    ("1/2", 0.5, "Fraction"),
    ("-5.5", -5.5, "Negative number"),
    ("ANSWER: 98.5 m/s", 98.5, "With unit suffix"),
    ("Final answer: 3.14159", 3.14159, "With prefix text"),
]

def test_extraction():
    """Run all extraction tests."""
    print("=" * 80)
    print("TESTING ROBUST NUMBER EXTRACTION")
    print("=" * 80)

    passed = 0
    failed = 0

    for input_val, expected, description in test_cases:
        extracted_val, method = extract_number(input_val)

        # Check if extraction succeeded
        if extracted_val is None:
            print(f"\n❌ FAILED: {description}")
            print(f"   Input: {input_val!r}")
            print(f"   Expected: {expected}")
            print(f"   Got: None (method: {method})")
            failed += 1
            continue

        # Check if value is close enough (within 0.001% for floating point comparison)
        tolerance = abs(expected) * 0.00001 + 1e-12
        if abs(extracted_val - expected) > tolerance:
            print(f"\n❌ FAILED: {description}")
            print(f"   Input: {input_val!r}")
            print(f"   Expected: {expected}")
            print(f"   Got: {extracted_val} (method: {method})")
            print(f"   Difference: {abs(extracted_val - expected)}")
            failed += 1
        else:
            print(f"✅ PASSED: {description}")
            print(f"   Input: {input_val!r} → {extracted_val} (via {method})")
            passed += 1

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed}/{len(test_cases)} tests passed")
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {failed} test(s) failed")
    print("=" * 80)

    return failed == 0

if __name__ == "__main__":
    success = test_extraction()
    sys.exit(0 if success else 1)
