import os
import json
import sys
from typing import Tuple
from dotenv import load_dotenv
import google.generativeai as genai

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from schema import StateObject, Plan
from prompts.planning import PLANNER_PROMPT
from config.pricing import MODEL_PRICING

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# PLANNER_PROMPT imported from prompts/planning.py
# MODEL_PRICING imported from config/pricing.py


def _calculate_plan_cost(completion, model: str) -> float:
    """
    Calculate the cost of a planning completion based on input/output tokens.

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


def plan(problem: str) -> Tuple[StateObject, Plan, float]:
    """
    Generate a structured plan for solving a physics/math problem.

    Args:
        problem: The problem text to solve

    Returns:
        Tuple of (StateObject, Plan, cost)
        - StateObject: Initial problem state with variables
        - Plan: Step-by-step solution plan
        - cost: Cost in USD for this planning step
    """

    model = "gemini-3-pro-preview"

    # Create the model instance
    client = genai.GenerativeModel(
        model_name=model,
        system_instruction=PLANNER_PROMPT,
        generation_config={"temperature": 0.1, "response_mime_type": "application/json"}
    )

    # Call LLM
    completion = client.generate_content(problem)

    # Calculate cost
    cost = _calculate_plan_cost(completion, model)

    # Parse JSON
    raw_response = completion.text
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
        print(f"Response length: {len(raw_response)}")
        print(f"First 500 chars: {raw_response[:500]}")
        print(f"Last 500 chars: {raw_response[-500:]}")
        raise

    # Validate with Pydantic (this will raise errors if schema is wrong)
    state = StateObject(**data['state'])
    plan = Plan(**data['plan'])

    return state, plan, cost
