"""Physics Lawyer critic - audits plans for conceptual physics errors."""

import os
import json
import sys
from typing import Dict, List, Any
from dotenv import load_dotenv
from openai import OpenAI

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from prompts.critics import PHYSICS_LAWYER_PROMPT
from planner.schema import Plan

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_KEY")
)


class AuditResult:
    """Result of physics audit."""
    def __init__(self, data: Dict[str, Any]):
        self.status = data.get("status", "REJECTED")
        self.reasoning = data.get("reasoning", "")
        self.critiques = data.get("critiques", [])
        self.is_approved = self.status == "APPROVED"

    def __repr__(self):
        return f"AuditResult(status={self.status}, critiques={len(self.critiques)})"


def audit_plan(problem: str, plan: Plan) -> AuditResult:
    """
    Audit a plan for physics violations using the Physics Lawyer.

    Args:
        problem: The original problem statement
        plan: The proposed solution plan

    Returns:
        AuditResult with status and specific critiques
    """
    model = "google/gemini-3-pro-preview"

    # Format plan as text for audit
    plan_text = _format_plan_for_audit(plan)

    # Construct the audit prompt
    audit_message = f"""PROBLEM:
{problem}

PROPOSED PLAN:
{plan_text}"""

    # Call LLM with physics lawyer prompt
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PHYSICS_LAWYER_PROMPT},
            {"role": "user", "content": audit_message}
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )

    # Parse response
    raw_response = completion.choices[0].message.content
    audit_data = json.loads(raw_response)

    return AuditResult(audit_data)


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
