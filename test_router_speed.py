#!/usr/bin/env python3
"""Quick test to measure router speed improvement."""

import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.core.router import classify_problem

# Test problems covering different tiers
TEST_PROBLEMS = [
    # EASY
    "What is Newton's Second Law?",
    # MEDIUM
    "A ball is thrown at 20 m/s at 45 degrees. Find the maximum height and range.",
    # HARD
    "Derive the partition function for a system of N non-interacting identical particles and explain how it relates to thermodynamic properties.",
]

def test_router_speed():
    """Test router speed on sample problems."""
    print("🚦 Router Speed Test")
    print("=" * 60)

    times = []

    for i, problem in enumerate(TEST_PROBLEMS, 1):
        print(f"\nTest {i}: {problem[:50]}...")

        start = time.time()
        classification, cost = classify_problem(problem)
        elapsed = time.time() - start
        times.append(elapsed)

        print(f"  Tier: {classification.tier}")
        print(f"  Confidence: {classification.confidence:.2f}")
        print(f"  Reasoning: {classification.reasoning}")
        print(f"  Key indicators: {', '.join(classification.key_indicators)}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Cost: ${cost:.6f}")

    print("\n" + "=" * 60)
    print(f"Average time: {sum(times) / len(times):.2f}s")
    print(f"Total time: {sum(times):.2f}s")

    if sum(times) / len(times) < 1.0:
        print("✅ Router is now fast! (<1s average)")
    elif sum(times) / len(times) < 2.0:
        print("⚠️  Router is acceptable (1-2s average)")
    else:
        print("❌ Router still too slow (>2s average)")

if __name__ == "__main__":
    test_router_speed()
