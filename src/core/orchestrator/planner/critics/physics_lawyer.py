"""Physics Lawyer critic - audits plans for conceptual physics errors."""

import os
import json
import sys
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv
import google.generativeai as genai

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from prompts.critics import PHYSICS_LAWYER_PROMPT
from planner.schema import Plan
from config.pricing import MODEL_PRICING

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def _calculate_lawyer_cost(completion, model: str) -> float:
    """
    Calculate the cost of a physics lawyer completion based on input/output tokens.

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


class AuditResult:
    """Result of physics audit."""
    def __init__(self, data: Dict[str, Any]):
        self.status = data.get("status", "REJECTED")
        self.reasoning = data.get("reasoning", "")
        self.critiques = data.get("critiques", [])
        self.is_approved = self.status == "APPROVED"

    def __repr__(self):
        return f"AuditResult(status={self.status}, critiques={len(self.critiques)})"


def audit_plan(problem: str, plan: Plan) -> Tuple[AuditResult, float]:
    """
    Audit a plan for physics violations using the Physics Lawyer.

    Args:
        problem: The original problem statement
        plan: The proposed solution plan

    Returns:
        Tuple of (AuditResult with status and critiques, cost in USD)
    """
    model = "gemini-3-pro-preview"

    # Format plan as text for audit
    plan_text = _format_plan_for_audit(plan)

    # Construct the audit prompt
    audit_message = f"""PROBLEM:
{problem}

PROPOSED PLAN:
{plan_text}"""

    # Create the model instance
    client = genai.GenerativeModel(
        model_name=model,
        system_instruction=PHYSICS_LAWYER_PROMPT,
        generation_config={"temperature": 0.1, "response_mime_type": "application/json"}
    )

    # Call LLM with physics lawyer prompt
    completion = client.generate_content(audit_message)

    # Calculate cost
    cost = _calculate_lawyer_cost(completion, model)

    # Parse response
    raw_response = completion.text
    audit_data = json.loads(raw_response)

    return AuditResult(audit_data), cost


def _format_plan_for_audit(plan: Plan) -> str:
    """Format a plan as readable text for audit."""
    lines = []
    lines.append(f"Domain: {plan.domain if hasattr(plan, 'domain') else 'N/A'}")
    lines.append(f"Approach: {plan.approach}")
    lines.append(f"Final Output: {plan.final_output}")
    lines.append(f"\nSteps ({len(plan.steps)} total):")

    for step in plan.steps:
        lines.append(f"\n  Step {step.step_id}: {step.description}")
        lines.append(f"    Operation: {step.operation}")
        lines.append(f"    Output: {step.output}")
        if hasattr(step, 'formula') and step.formula:
            lines.append(f"    Formula: {step.formula}")
        if hasattr(step, 'inputs') and step.inputs:
            lines.append(f"    Inputs: {step.inputs}")

    return "\n".join(lines)
