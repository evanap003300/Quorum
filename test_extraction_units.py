#!/usr/bin/env python3
"""Test extraction with units that contain special characters like / and ^."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from benchmarking.evaluation.answer_comparator import extract_number

# Test cases with units: (input, expected_output, description)
test_cases = [
    ("5", 5.0, "Plain number"),
    ("5 m/s", 5.0, "Number with m/s unit"),
    ("10 1/s", 10.0, "Number with 1/s unit"),
    ("10 s^-1", 10.0, "Number with s^-1 unit"),
    ("42 kg/m^3", 42.0, "Number with kg/m^3 unit"),
    ("3.5 J K^-1 mol^-1", 3.5, "Complex unit from thermodynamics"),
    ("1.5e-10 m^-3", 1.5e-10, "Scientific notation with unit"),
    ("The answer is 7 m/s", 7.0, "Answer embedded in sentence with unit"),
    ("ANSWER: 98.5 m/s", 98.5, "Answer tag with unit"),
    ("Final answer: 15 Hz", 15.0, "With frequency unit"),
]

def test_extraction():
    """Run all extraction tests."""
    print("=" * 80)
    print("TESTING NUMBER EXTRACTION WITH UNITS")
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
