"""Plan Revisor - fixes plans flagged by critics."""

import os
import json
import sys
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
import google.generativeai as genai

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from schema import Plan
from prompts.revisor import REVISOR_PROMPT
from solver.parsing import calculate_cost
from config.pricing import MODEL_PRICING

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def _calculate_revisor_cost(completion, model: str) -> float:
    """
    Calculate the cost of a revisor completion based on input/output tokens.

    Args:
        completion: Google GenerativeAI response object with usage info
        model: Model name to look up pricing

    Returns:
        Cost in USD
    """
    if model not in MODEL_PRICING:
        return 0.0

    pricing = MODEL_PRICING[model]
    usage = completion.usage_metadata

    # Calculate cost: (tokens * price_per_million) / 1_000_000
    input_cost = (usage.prompt_token_count * pricing["input"]) / 1_000_000
    output_cost = (usage.candidates_token_count * pricing["output"]) / 1_000_000

    return input_cost + output_cost


def revise_plan(problem: str, original_plan: Plan, critiques: List[Dict[str, Any]]) -> Tuple[Plan, float]:
    """
    Revise a plan based on critiques from the Physics Lawyer.

    Args:
        problem: The original problem statement
        original_plan: The plan that was flagged
        critiques: List of critique dictionaries from the Physics Lawyer

    Returns:
        Tuple of (Revised Plan object with corrections applied, cost in USD)
    """
    model = "gemini-3-flash-preview"

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

    # Create the model instance
    client = genai.GenerativeModel(
        model_name=model,
        system_instruction=REVISOR_PROMPT,
        generation_config={"temperature": 0.1, "response_mime_type": "application/json"}
    )

    # Call LLM to revise
    completion = client.generate_content(revision_message)

    # Calculate cost
    cost = _calculate_revisor_cost(completion, model)

    # Parse revised plan
    raw_response = completion.text
    revised_data = json.loads(raw_response)

    # If the response is wrapped in a "plan" key, unwrap it
    if "plan" in revised_data and isinstance(revised_data["plan"], dict):
        revised_data = revised_data["plan"]

    # Reconstruct as Plan object
    revised_plan = Plan(**revised_data)

    return revised_plan, cost


def _plan_to_dict(plan: Plan) -> Dict[str, Any]:
    """Convert a Plan object to a dictionary for JSON serialization."""
    return plan.model_dump()
