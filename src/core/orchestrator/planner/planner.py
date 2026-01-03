import os
import json
import sys
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
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

# Timeout settings for planning API calls
PLAN_TIMEOUT_SECONDS = 60
MAX_PLAN_RETRIES = 2

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


def _generate_plan_with_model(client, problem: str):
    """Internal function to call the model - used for timeout wrapping."""
    return client.generate_content(problem)


def plan(problem: str) -> Tuple[StateObject, Plan, float]:
    """
    Generate a structured plan for solving a physics/math problem.

    Uses timeout handling and retry logic to handle API timeouts gracefully.

    Args:
        problem: The problem text to solve

    Returns:
        Tuple of (StateObject, Plan, cost)
        - StateObject: Initial problem state with variables
        - Plan: Step-by-step solution plan
        - cost: Cost in USD for this planning step

    Raises:
        TimeoutError: If planning fails after all retries
        ValueError: If response cannot be parsed
    """

    model = "gemini-3-pro-preview"
    total_cost = 0.0
    last_error = None

    # Create the model instance
    client = genai.GenerativeModel(
        model_name=model,
        system_instruction=PLANNER_PROMPT,
        generation_config={"temperature": 0.1, "response_mime_type": "application/json"}
    )

    for attempt in range(MAX_PLAN_RETRIES):
        try:
            # Use ThreadPoolExecutor for timeout control on sync API call
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_generate_plan_with_model, client, problem)
                try:
                    completion = future.result(timeout=PLAN_TIMEOUT_SECONDS)
                except FuturesTimeoutError:
                    if attempt < MAX_PLAN_RETRIES - 1:
                        print(f"  ⚠ Planning timeout after {PLAN_TIMEOUT_SECONDS}s (attempt {attempt + 1}/{MAX_PLAN_RETRIES}), retrying...")
                        continue
                    else:
                        raise TimeoutError(f"Planning timed out after {MAX_PLAN_RETRIES} attempts ({PLAN_TIMEOUT_SECONDS}s each)")

            # Calculate cost
            cost = _calculate_plan_cost(completion, model)
            total_cost += cost

            # Parse JSON
            raw_response = completion.text
            try:
                data = json.loads(raw_response)
            except json.JSONDecodeError as e:
                if attempt < MAX_PLAN_RETRIES - 1:
                    print(f"  ⚠ JSON parse error (attempt {attempt + 1}/{MAX_PLAN_RETRIES}): {e}")
                    print(f"  Response preview: {raw_response[:200]}...")
                    continue
                print(f"JSON Parse Error: {e}")
                print(f"Response length: {len(raw_response)}")
                print(f"First 500 chars: {raw_response[:500]}")
                print(f"Last 500 chars: {raw_response[-500:]}")
                raise

            # Validate JSON structure before accessing keys
            if data is None or not isinstance(data, dict):
                if attempt < MAX_PLAN_RETRIES - 1:
                    print(f"  ⚠ Planner returned invalid JSON (not a dict), retrying...")
                    continue
                raise ValueError(f"Planner returned invalid JSON: {raw_response[:200]}")

            if 'state' not in data or 'plan' not in data:
                if attempt < MAX_PLAN_RETRIES - 1:
                    print(f"  ⚠ Planner JSON missing 'state' or 'plan' keys, retrying...")
                    continue
                raise ValueError(f"Planner JSON missing required keys. Got: {list(data.keys())}")

            # Validate with Pydantic (this will raise errors if schema is wrong)
            state = StateObject(**data['state'])
            plan_obj = Plan(**data['plan'])

            return state, plan_obj, total_cost

        except TimeoutError:
            raise
        except Exception as e:
            last_error = e
            if attempt < MAX_PLAN_RETRIES - 1:
                print(f"  ⚠ Planning error (attempt {attempt + 1}/{MAX_PLAN_RETRIES}): {type(e).__name__}: {str(e)[:100]}")
                continue
            raise

    # Should not reach here, but just in case
    raise last_error if last_error else RuntimeError("Planning failed with unknown error")
