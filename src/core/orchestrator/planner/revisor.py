"""Plan Revisor - fixes plans flagged by critics."""

import os
import json
import sys
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from schema import Plan
from prompts.revisor import REVISOR_PROMPT

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_KEY")
)


def revise_plan(problem: str, original_plan: Plan, critiques: List[Dict[str, Any]]) -> Plan:
    """
    Revise a plan based on critiques from the Physics Lawyer.

    Args:
        problem: The original problem statement
        original_plan: The plan that was flagged
        critiques: List of critique dictionaries from the Physics Lawyer

    Returns:
        Revised Plan object with corrections applied
    """
    model = "google/gemini-3-pro-preview"

    # Convert plan to dict for JSON handling
    plan_dict = _plan_to_dict(original_plan)

    # Format the revision request
    revision_message = f"""PROBLEM:
{problem}

ORIGINAL PLAN:
{json.dumps(plan_dict, indent=2)}

CRITIQUES TO ADDRESS:
{json.dumps(critiques, indent=2)}

Please revise the plan to address all critiques while maintaining the same JSON schema.
"""

    # Call LLM to revise
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": REVISOR_PROMPT},
            {"role": "user", "content": revision_message}
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )

    # Parse revised plan
    raw_response = completion.choices[0].message.content
    revised_data = json.loads(raw_response)

    # If the response is wrapped in a "plan" key, unwrap it
    if "plan" in revised_data and isinstance(revised_data["plan"], dict):
        revised_data = revised_data["plan"]

    # Reconstruct as Plan object
    revised_plan = Plan(**revised_data)

    return revised_plan


def _plan_to_dict(plan: Plan) -> Dict[str, Any]:
    """Convert a Plan object to a dictionary for JSON serialization."""
    return plan.model_dump()
