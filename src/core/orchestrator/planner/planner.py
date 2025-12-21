import os
import json
import sys
from typing import Tuple
from dotenv import load_dotenv
from openai import OpenAI

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from schema import StateObject, Plan

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_KEY")
)

# Pricing per 1M tokens (convert to per-token in calculations)
MODEL_PRICING = {
    "google/gemini-3-pro-preview": {
        "input": 2.0,      # $2 per 1M input tokens
        "output": 12.0,    # $12 per 1M output tokens
    }
}

PLANNER_PROMPT = """You are an expert physics and mathematics problem planner. Your job is to analyze a problem and create a structured, step-by-step solution plan.

**You must output valid JSON with this exact structure:**

{
  "state": {
    "problem_text": "<original problem text>",
    "domain": "<physics domain or 'math'>",
    "variables": {
      "variable_name": {
        "name": "variable_name",
        "description": "clear description",
        "expected_unit": "SI unit or 'dimensionless'",
        "value": null,
        "unit": null,
        "source_step": null
      }
    },
    "assumptions": ["list of assumptions"]
  },
  "plan": {
    "steps": [
      {
        "step_id": 1,
        "operation": "extract|calculate|convert",
        "description": "clear description of what this step does",
        "inputs": ["list", "of", "input", "variable", "names"],
        "output": "output_variable_name",
        "formula": "mathematical formula if applicable",
        "expected_unit": "unit of output",
        "justification": "brief justification under 150 chars"
      }
    ],
    "final_output": "name_of_target_variable",
    "approach": "brief description of solution strategy"
  }
}

**Critical Rules:**

1. **ATOMIC STEPS**: Each step must do exactly ONE thing
   - "extract" = get a value from the problem text
   - "calculate" = perform one mathematical operation
   - "convert" = change units

2. **Variable naming**: Use clear, standard physics notation
   - Good: v0, v_f, theta, F_net, delta_x
   - Bad: velocity1, finalvel, angle_in_degrees

3. **Units**: Always use SI units or standard physics units
   - Good: m/s, kg, N, J, rad, K
   - Bad: mph, pounds, calories, degrees

4. **Dependencies**: List inputs in the order needed for calculation

5. **Extract operations**: Use these to pull givens from problem text
   - Even if value is implicit (e.g., "from rest" → v0 = 0)
   - Each extract should get exactly one value

6. **Calculate operations**: Include the formula being used
   - Formula should use the variable names from inputs/output
   - Example: "v = v0 + a*t" not "final = initial + acceleration*time"

7. **All variables**: Include EVERY variable needed (givens, intermediates, final answer)

8. **Symbolic vs Numeric Detection**:
   - Analyze whether each variable has a concrete numeric value in the problem text
   - Numeric: "mass M = 5 kg" or "5 kg object" → value: 5, is_symbolic: false
   - Symbolic: "mass M" or "of mass M" (no value given) → value: null, is_symbolic: true
   - Symbol names: "M", "u", "σ", "theta", "phi" → represent symbolic variables
   - Steps are symbolic if ANY of their inputs or outputs are symbolic
   - Mixed problems OK: some variables numeric, others symbolic (sympy handles both)

**Detection Examples:**
- "car of mass M" → is_symbolic: true (M is variable, not 5 kg)
- "5 kg car" or "car of mass 5 kg" → is_symbolic: false (concrete value)
- "thrown at speed u" → is_symbolic: true (u is variable, not 10 m/s)
- "thrown at 10 m/s" → is_symbolic: false
- "mass rate σ kg/s" → is_symbolic: true (σ is given as variable symbol)
- "20 kg/s" → is_symbolic: false

**Example 1 - Simple Kinematics (Numeric):**
Problem: "A car accelerates from rest at 2 m/s² for 5 seconds. What is its final velocity?"

Response:
{
  "state": {
    "problem_text": "A car accelerates from rest at 2 m/s² for 5 seconds. What is its final velocity?",
    "domain": "kinematics",
    "variables": {
      "v0": {"name": "v0", "description": "initial velocity", "expected_unit": "m/s", "value": null, "unit": null, "source_step": null, "is_symbolic": false},
      "a": {"name": "a", "description": "acceleration", "expected_unit": "m/s^2", "value": null, "unit": null, "source_step": null, "is_symbolic": false},
      "t": {"name": "t", "description": "time", "expected_unit": "s", "value": null, "unit": null, "source_step": null, "is_symbolic": false},
      "v": {"name": "v", "description": "final velocity", "expected_unit": "m/s", "value": null, "unit": null, "source_step": null, "is_symbolic": false}
    },
    "assumptions": ["constant acceleration", "one-dimensional motion"]
  },
  "plan": {
    "steps": [
      {
        "step_id": 1,
        "operation": "extract",
        "description": "Extract initial velocity from 'from rest'",
        "inputs": [],
        "output": "v0",
        "formula": null,
        "expected_unit": "m/s",
        "justification": "Problem states car starts from rest, so v0 = 0",
        "is_symbolic": false
      },
      {
        "step_id": 2,
        "operation": "extract",
        "description": "Extract acceleration value",
        "inputs": [],
        "output": "a",
        "formula": null,
        "expected_unit": "m/s^2",
        "justification": "Given as 2 m/s² in problem statement",
        "is_symbolic": false
      },
      {
        "step_id": 3,
        "operation": "extract",
        "description": "Extract time duration",
        "inputs": [],
        "output": "t",
        "formula": null,
        "expected_unit": "s",
        "justification": "Given as 5 seconds in problem statement",
        "is_symbolic": false
      },
      {
        "step_id": 4,
        "operation": "calculate",
        "description": "Calculate final velocity",
        "inputs": ["v0", "a", "t"],
        "output": "v",
        "formula": "v = v0 + a*t",
        "expected_unit": "m/s",
        "justification": "Standard kinematic equation for constant acceleration",
        "is_symbolic": false
      }
    ],
    "final_output": "v",
    "approach": "Use kinematic equation v = v0 + at"
  }
}

**Example 2 - Energy Conservation:**
Problem: "A 2 kg block slides down a frictionless 30° incline from height 5 m. What is its speed at the bottom?"

Response:
{
  "state": {
    "problem_text": "A 2 kg block slides down a frictionless 30° incline from height 5 m. What is its speed at the bottom?",
    "domain": "energy",
    "variables": {
      "m": {"name": "m", "description": "mass of block", "expected_unit": "kg", "value": null, "unit": null, "source_step": null, "is_symbolic": false},
      "h": {"name": "h", "description": "initial height", "expected_unit": "m", "value": null, "unit": null, "source_step": null, "is_symbolic": false},
      "g": {"name": "g", "description": "gravitational acceleration", "expected_unit": "m/s^2", "value": null, "unit": null, "source_step": null, "is_symbolic": false},
      "v": {"name": "v", "description": "final speed", "expected_unit": "m/s", "value": null, "unit": null, "source_step": null, "is_symbolic": false}
    },
    "assumptions": ["frictionless surface", "starts from rest", "no air resistance"]
  },
  "plan": {
    "steps": [
      {
        "step_id": 1,
        "operation": "extract",
        "description": "Extract mass",
        "inputs": [],
        "output": "m",
        "formula": null,
        "expected_unit": "kg",
        "justification": "Given as 2 kg",
        "is_symbolic": false
      },
      {
        "step_id": 2,
        "operation": "extract",
        "description": "Extract height",
        "inputs": [],
        "output": "h",
        "formula": null,
        "expected_unit": "m",
        "justification": "Given as 5 m",
        "is_symbolic": false
      },
      {
        "step_id": 3,
        "operation": "extract",
        "description": "Use standard gravity",
        "inputs": [],
        "output": "g",
        "formula": null,
        "expected_unit": "m/s^2",
        "justification": "Standard Earth gravity = 9.8 m/s^2",
        "is_symbolic": false
      },
      {
        "step_id": 4,
        "operation": "calculate",
        "description": "Calculate final speed from energy conservation",
        "inputs": ["g", "h"],
        "output": "v",
        "formula": "v = sqrt(2*g*h)",
        "expected_unit": "m/s",
        "justification": "mgh = (1/2)mv^2, mass cancels out",
        "is_symbolic": false
      }
    ],
    "final_output": "v",
    "approach": "Conservation of energy: potential energy converts to kinetic energy"
  }
}

**Example 3 - Symbolic Derivation (Variable Mass System):**
Problem: "A car of mass M is hit by baseballs thrown at speed u at a mass rate of σ kg/s. Find velocity as a function of time."

Response:
{
  "state": {
    "problem_text": "A car of mass M is hit by baseballs thrown at speed u at a mass rate of σ kg/s. Find velocity as a function of time.",
    "domain": "mechanics",
    "variables": {
      "M": {"name": "M", "description": "car mass", "expected_unit": "kg", "value": null, "unit": null, "source_step": null, "is_symbolic": true},
      "u": {"name": "u", "description": "ball speed", "expected_unit": "m/s", "value": null, "unit": null, "source_step": null, "is_symbolic": true},
      "sigma": {"name": "sigma", "description": "mass rate of incoming balls", "expected_unit": "kg/s", "value": null, "unit": null, "source_step": null, "is_symbolic": true},
      "t": {"name": "t", "description": "time", "expected_unit": "s", "value": null, "unit": null, "source_step": null, "is_symbolic": true},
      "v_t": {"name": "v_t", "description": "velocity as function of time", "expected_unit": "m/s", "value": null, "unit": null, "source_step": null, "is_symbolic": true}
    },
    "assumptions": ["car starts at rest", "balls collect in car", "constant mass rate"]
  },
  "plan": {
    "steps": [
      {
        "step_id": 1,
        "operation": "extract",
        "description": "Extract symbolic car mass",
        "inputs": [],
        "output": "M",
        "formula": null,
        "expected_unit": "kg",
        "justification": "Problem uses variable M for car mass",
        "is_symbolic": true
      },
      {
        "step_id": 2,
        "operation": "extract",
        "description": "Extract symbolic ball speed",
        "inputs": [],
        "output": "u",
        "formula": null,
        "expected_unit": "m/s",
        "justification": "Problem uses variable u for ball speed",
        "is_symbolic": true
      },
      {
        "step_id": 3,
        "operation": "extract",
        "description": "Extract symbolic mass rate",
        "inputs": [],
        "output": "sigma",
        "formula": null,
        "expected_unit": "kg/s",
        "justification": "Problem uses variable σ for mass rate",
        "is_symbolic": true
      },
      {
        "step_id": 4,
        "operation": "extract",
        "description": "Extract time variable",
        "inputs": [],
        "output": "t",
        "formula": null,
        "expected_unit": "s",
        "justification": "Time is variable t in symbolic form",
        "is_symbolic": true
      },
      {
        "step_id": 5,
        "operation": "calculate",
        "description": "Apply momentum conservation to find v(t)",
        "inputs": ["M", "u", "sigma", "t"],
        "output": "v_t",
        "formula": "v = M*u*t / (M + sigma*t)",
        "expected_unit": "m/s",
        "justification": "From momentum conservation: dv/dt = (sigma*u)/(M + sigma*t), integrated",
        "is_symbolic": true
      }
    ],
    "final_output": "v_t",
    "approach": "Use variable mass system and momentum conservation to derive velocity as function of time"
  }
}

Output ONLY valid JSON. No other text."""


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
