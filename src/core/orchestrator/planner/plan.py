"""
DEPRECATED: This module has been superseded by planner.py

The simple plan() function in this module is deprecated. Use the new Planner class
from planner.py instead, which provides:

- Structured JSON output with validated schemas
- Atomic step enforcement
- Critique-refine loop integration
- Comprehensive error handling
- Full audit trail and state management

Migration guide:

OLD (plan.py):
    from src.core.orchestrator.planner.plan import plan
    result = plan("What is 2 + 2?")  # Returns unstructured string

NEW (planner.py):
    from src.core.orchestrator.planner.planner import Planner
    planner = Planner()
    state = planner.create_plan("What is 2 + 2?", domain="mathematics")
    # Returns structured StateObject with validated Plan

See src/core/orchestrator/planner/planner.py for the full implementation.
"""

import os
import warnings
from dotenv import load_dotenv
from openai import OpenAI

# Deprecation warning
warnings.warn(
    "plan.py is deprecated. Use planner.py and the Planner class instead.",
    DeprecationWarning,
    stacklevel=2
)

# Load environment variables from .env file
load_dotenv()

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_KEY")
)

def plan(text: str) -> str:
    """
    DEPRECATED: Use Planner class from planner.py instead.

    Takes in text describing a problem and returns a planning strategy from the LLM.

    Args:
        text: The problem description to plan for

    Returns:
        The LLM's planning response (unstructured string)
    """
    warnings.warn(
        "plan() is deprecated. Use Planner.create_plan() from planner.py instead.",
        DeprecationWarning,
        stacklevel=2
    )
    completion = client.chat.completions.create(
        model="google/gemini-3-pro-preview",
        messages=[
            {
                "role": "system",
                "content": "You are an expert planner. Your job is to break down complex problems into clear, actionable steps. Provide a detailed plan that outlines the approach to solve the given problem."
            },
            {
                "role": "user",
                "content": text
            }
        ]
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    # Example usage (deprecated - for reference only)
    print(plan("What is 9 + 10"))