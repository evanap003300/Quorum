"""
LLM prompts for the planning system.

This module provides system prompts and user prompts that guide the LLM to generate
valid, atomic plans in structured JSON format. Prompts emphasize:
1. Atomicity (each step = one operation)
2. JSON-only output
3. Short, focused justifications
4. Explicit assumptions
"""

from .schemas import StateObject


def get_planning_system_prompt() -> str:
    """
    Generate the system prompt for plan generation.

    This prompt sets the expert role and establishes the constraints and requirements.
    Returns:
        System prompt string for the planner LLM
    """
    return """You are an expert physics and mathematics problem planner. Your job is to:

1. **Parse** the natural language problem into structured components
2. **Generate** a minimal, atomic plan consisting of single-operation steps
3. **Track** all variables and explicit assumptions
4. **Output** ONLY valid JSON (no explanations, no text outside JSON)

## CRITICAL RULES FOR ATOMICITY

Each step must perform **EXACTLY ONE OPERATION**. No compound operations.

### ✓ VALID ATOMIC STEPS:
- "Isolate a from equation F = m*a" (single algebraic operation)
- "Substitute v0=5 into kinematic equation" (single substitution)
- "Apply law of conservation of energy" (single principle application)
- "Solve ODE: dv/dt = a using integration" (single differential operation)
- "Compute matrix inverse using Gaussian elimination" (single matrix operation)

### ✗ INVALID COMPOUND STEPS:
- "Solve for a and substitute into the next equation" (TWO operations: solve AND substitute)
- "Differentiate position to get velocity, then acceleration" (TWO operations: differentiate AND differentiate again)
- "First isolate x, then substitute y" (explicit multi-step sequence)
- "Combine forces and apply Newton's second law" (TWO operations: combine AND apply)

## STEP TYPE DEFINITIONS

Each step must be one of these types:
- **algebraic**: Solve equations, isolate variables, perform algebraic manipulations
- **symbolic**: Manipulate symbolic expressions, simplify, factor, expand
- **matrix**: Matrix operations (multiply, invert, eigenvalues)
- **ode**: Solve differential equations, integrate, differentiate
- **substitute**: Replace variables with values or other expressions

## JUSTIFICATION REQUIREMENTS

- Must be ≤ 50 tokens (~200 characters)
- Must cite the principle/law applied (e.g., "Newton's 2nd law", "Kinematics formula")
- Must be concise and focused (not a full explanation)

## VARIABLE TRACKING

- List ALL variables used in the plan (both knowns and intermediates)
- Include units for all physical quantities
- If a variable is missing from knowns, flag it with "assumption_needed"

## ASSUMPTION HANDLING

List all explicit assumptions:
- Physical approximations (e.g., "constant acceleration", "no air resistance")
- Standard constants (e.g., "g:9.81 m/s^2")
- Simplifying assumptions (e.g., "treat as point mass")

## OUTPUT FORMAT

Return ONLY a valid JSON object matching the Plan schema. Example structure:
{
  "plan_version": 1,
  "variables": {
    "v0": {"unit": "m/s", "desc": "initial velocity"},
    "a": {"unit": "m/s^2", "desc": "acceleration"}
  },
  "assumptions": ["constant acceleration", "no air resistance"],
  "steps": [
    {
      "step_id": "P1",
      "type": "algebraic",
      "description": "Isolate a from F = m*a",
      "inputs": ["F", "m"],
      "output": "a",
      "expected_units": "m/s^2",
      "tolerance": 1e-8,
      "justification": "Apply Newton's 2nd law (F = m*a) to solve for acceleration"
    }
  ],
  "composition": {"P1": []},
  "metadata": {}
}

## REMEMBER

- NO numeric computation (this is symbolic planning only)
- NO freeform text in JSON fields
- Each step = ONE operation (check your descriptions for 'and', 'then', 'also')
- Every justification must cite a principle
- If unsure about variables/values, flag with assumption_needed instead of guessing"""


def get_planning_user_prompt(state: StateObject) -> str:
    """
    Generate the user prompt for a specific planning task.

    Args:
        state: The StateObject containing problem context

    Returns:
        User prompt string with problem description and context
    """
    # Format known variables if any
    knowns_str = ""
    if state.knowns:
        knowns_lines = []
        for var_name, var_info in state.knowns.items():
            if isinstance(var_info, dict):
                value = var_info.get("value", "?")
                unit = var_info.get("unit", "")
                knowns_lines.append(f"  - {var_name} = {value} {unit}")
            else:
                knowns_lines.append(f"  - {var_name} = {var_info}")
        knowns_str = "Known variables:\n" + "\n".join(knowns_lines) + "\n\n"

    # Format constraints if any
    constraints_str = ""
    if state.constraints:
        constraints_str = "Constraints:\n"
        for constraint in state.constraints:
            constraints_str += f"  - {constraint}\n"
        constraints_str += "\n"

    # Format unknowns if any
    unknowns_str = ""
    if state.unknowns:
        unknowns_str = f"Find: {', '.join(state.unknowns)}\n\n"

    prompt = f"""Problem (Domain: {state.domain}):
{state.problem_text}

{knowns_str}{unknowns_str}{constraints_str}Goal: Generate a minimal, atomic plan to solve this problem.

IMPORTANT REMINDERS:
1. Each step must be ATOMIC (single operation only)
2. Output VALID JSON ONLY (no text before/after)
3. Justifications must cite physics/math principles (≤50 tokens each)
4. No numeric computation - this is symbolic planning
5. If information is missing, flag it instead of assuming

Generate the complete Plan JSON now:"""

    return prompt


def get_refinement_system_prompt() -> str:
    """
    Generate the system prompt for plan refinement.

    Used when the plan needs to be refined based on critic feedback.

    Returns:
        System prompt string for refinement
    """
    return """You are an expert problem solver refining a plan based on critique feedback.

Your task:
1. Review the original plan
2. Examine the critic feedback (list of issues and suggestions)
3. Produce a refined plan that addresses the blocking/warning issues
4. Maintain atomicity (each step = one operation)
5. Output ONLY valid JSON (no text)

## REFINEMENT RULES

- Address all "blocking" severity critiques
- Try to address "warning" severity critiques if possible
- Do NOT introduce new issues while fixing existing ones
- Increment plan_version
- Keep steps atomic (do NOT combine steps to "fix" a problem)
- If a step must be broken into multiple steps, do so (better to have more steps)
- If unsure how to fix an issue, provide a detailed justification explaining why

## OUTPUT FORMAT

Return the refined Plan JSON with:
- Incremented plan_version
- Modified steps addressing the critiques
- Updated justifications explaining how each fix addresses the critique
- Clear audit trail (annotations welcome in justifications)"""


def get_refinement_user_prompt(plan_json: str, critiques_json: str) -> str:
    """
    Generate the user prompt for plan refinement.

    Args:
        plan_json: The current plan as JSON string
        critiques_json: List of critiques as JSON string

    Returns:
        User prompt string with plan and critique context
    """
    return f"""Original Plan:
{plan_json}

Critique Feedback:
{critiques_json}

Task: Refine the plan to address the blocking and warning issues while maintaining atomicity.

Output the refined Plan JSON only (no explanations):"""


# Few-shot examples for in-context learning
FEW_SHOT_EXAMPLES = [
    {
        "name": "Simple Kinematics",
        "problem": "A car accelerates uniformly from 0 to 60 mph in 5 seconds. What is its acceleration?",
        "plan": {
            "plan_version": 1,
            "variables": {
                "v0": {"unit": "m/s", "desc": "initial velocity (0 at start)"},
                "v_f_mph": {"unit": "mi/h", "desc": "final velocity (60 mph)"},
                "v_f": {"unit": "m/s", "desc": "final velocity (SI units)"},
                "t": {"unit": "s", "desc": "time interval (5 seconds)"},
                "a": {"unit": "m/s^2", "desc": "acceleration (unknown)"}
            },
            "assumptions": ["constant acceleration", "no air resistance", "conversion factor: 1 mph = 0.44704 m/s"],
            "steps": [
                {
                    "step_id": "P1",
                    "type": "substitute",
                    "description": "Convert 60 mph to m/s using conversion factor",
                    "inputs": ["v_f_mph"],
                    "output": "v_f",
                    "expected_units": "m/s",
                    "tolerance": 0.01,
                    "justification": "Convert to SI units for consistent calculation"
                },
                {
                    "step_id": "P2",
                    "type": "algebraic",
                    "description": "Apply kinematics equation a = (v_f - v0) / t",
                    "inputs": ["v_f", "v0", "t"],
                    "output": "a",
                    "expected_units": "m/s^2",
                    "tolerance": 0.01,
                    "justification": "Standard kinematic formula for constant acceleration"
                }
            ],
            "composition": {"P2": ["P1"]},
            "metadata": {"difficulty": "introductory", "domain": "kinematics"}
        }
    },
    {
        "name": "Spring Energy",
        "problem": "A spring with k=100 N/m is compressed 0.1 m. Find the potential energy stored.",
        "plan": {
            "plan_version": 1,
            "variables": {
                "k": {"unit": "N/m", "desc": "spring constant (given)"},
                "x": {"unit": "m", "desc": "compression distance (given)"},
                "PE": {"unit": "J", "desc": "elastic potential energy (unknown)"}
            },
            "assumptions": ["Hooke's law applies", "ideal spring (no damping)"],
            "steps": [
                {
                    "step_id": "P1",
                    "type": "algebraic",
                    "description": "Apply spring potential energy formula PE = (1/2)*k*x^2",
                    "inputs": ["k", "x"],
                    "output": "PE",
                    "expected_units": "J",
                    "tolerance": 0.01,
                    "justification": "Elastic potential energy from Hooke's law"
                }
            ],
            "composition": {"P1": []},
            "metadata": {"difficulty": "introductory", "domain": "energy"}
        }
    },
    {
        "name": "Invalid - Compound Operation",
        "problem": "Solve for x then substitute into equation 2.",
        "status": "REJECTED",
        "reason": "Contains compound operation (solve AND substitute). Should be two separate steps."
    }
]


# Export all prompt functions
__all__ = [
    "get_planning_system_prompt",
    "get_planning_user_prompt",
    "get_refinement_system_prompt",
    "get_refinement_user_prompt",
    "FEW_SHOT_EXAMPLES",
]
