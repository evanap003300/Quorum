import os
import json
import sys
from typing import Tuple
from dotenv import load_dotenv
from openai import OpenAI

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from schema import StateObject, Plan
from prompts.planning import PLANNER_PROMPT
from config.pricing import MODEL_PRICING

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_KEY")
)

# PLANNER_PROMPT imported from prompts/planning.py
# MODEL_PRICING imported from config/pricing.py


def _calculate_plan_cost(completion, model: str) -> float:
    """
    Calculate the cost of a planning completion based on input/output tokens.

    Args:
        completion: OpenAI completion object with usage info
        model: Model name to look up pricing

    Returns:
        Cost in USD
    """
    if model not in MODEL_PRICING:
        return 0.0

    pricing = MODEL_PRICING[model]
    usage = completion.usage

    # Calculate cost: (tokens * price_per_million) / 1_000_000
    input_cost = (usage.prompt_tokens * pricing["input"]) / 1_000_000
    output_cost = (usage.completion_tokens * pricing["output"]) / 1_000_000

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

    model = "google/gemini-3-pro-preview"

    # Call LLM
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": problem}
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )

    # Calculate cost
    cost = _calculate_plan_cost(completion, model)

    # Parse JSON
    raw_response = completion.choices[0].message.content
    data = json.loads(raw_response)

    # Validate with Pydantic (this will raise errors if schema is wrong)
    state = StateObject(**data['state'])
    plan = Plan(**data['plan'])

    return state, plan, cost
