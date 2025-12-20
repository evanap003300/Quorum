import os
import json
from typing import Tuple
from dotenv import load_dotenv
from openai import OpenAI
from schema import StateObject, Plan

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_KEY")
)

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

**Example 1 - Simple Kinematics:**
Problem: "A car accelerates from rest at 2 m/s² for 5 seconds. What is its final velocity?"

Response:
{
  "state": {
    "problem_text": "A car accelerates from rest at 2 m/s² for 5 seconds. What is its final velocity?",
    "domain": "kinematics",
    "variables": {
      "v0": {"name": "v0", "description": "initial velocity", "expected_unit": "m/s", "value": null, "unit": null, "source_step": null},
      "a": {"name": "a", "description": "acceleration", "expected_unit": "m/s^2", "value": null, "unit": null, "source_step": null},
      "t": {"name": "t", "description": "time", "expected_unit": "s", "value": null, "unit": null, "source_step": null},
      "v": {"name": "v", "description": "final velocity", "expected_unit": "m/s", "value": null, "unit": null, "source_step": null}
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
        "justification": "Problem states car starts from rest, so v0 = 0"
      },
      {
        "step_id": 2,
        "operation": "extract",
        "description": "Extract acceleration value",
        "inputs": [],
        "output": "a",
        "formula": null,
        "expected_unit": "m/s^2",
        "justification": "Given as 2 m/s² in problem statement"
      },
      {
        "step_id": 3,
        "operation": "extract",
        "description": "Extract time duration",
        "inputs": [],
        "output": "t",
        "formula": null,
        "expected_unit": "s",
        "justification": "Given as 5 seconds in problem statement"
      },
      {
        "step_id": 4,
        "operation": "calculate",
        "description": "Calculate final velocity",
        "inputs": ["v0", "a", "t"],
        "output": "v",
        "formula": "v = v0 + a*t",
        "expected_unit": "m/s",
        "justification": "Standard kinematic equation for constant acceleration"
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
      "m": {"name": "m", "description": "mass of block", "expected_unit": "kg", "value": null, "unit": null, "source_step": null},
      "h": {"name": "h", "description": "initial height", "expected_unit": "m", "value": null, "unit": null, "source_step": null},
      "g": {"name": "g", "description": "gravitational acceleration", "expected_unit": "m/s^2", "value": null, "unit": null, "source_step": null},
      "v": {"name": "v", "description": "final speed", "expected_unit": "m/s", "value": null, "unit": null, "source_step": null}
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
        "justification": "Given as 2 kg"
      },
      {
        "step_id": 2,
        "operation": "extract",
        "description": "Extract height",
        "inputs": [],
        "output": "h",
        "formula": null,
        "expected_unit": "m",
        "justification": "Given as 5 m"
      },
      {
        "step_id": 3,
        "operation": "extract",
        "description": "Use standard gravity",
        "inputs": [],
        "output": "g",
        "formula": null,
        "expected_unit": "m/s^2",
        "justification": "Standard Earth gravity = 9.8 m/s^2"
      },
      {
        "step_id": 4,
        "operation": "calculate",
        "description": "Calculate final speed from energy conservation",
        "inputs": ["g", "h"],
        "output": "v",
        "formula": "v = sqrt(2*g*h)",
        "expected_unit": "m/s",
        "justification": "mgh = (1/2)mv^2, mass cancels out"
      }
    ],
    "final_output": "v",
    "approach": "Conservation of energy: potential energy converts to kinetic energy"
  }
}

Output ONLY valid JSON. No other text."""


def plan(problem: str) -> Tuple[StateObject, Plan]:
    """
    Generate a structured plan for solving a physics/math problem.
    
    Args:
        problem: The problem text to solve
        
    Returns:
        Tuple of (StateObject, Plan)
    """
    
    # Call LLM
    completion = client.chat.completions.create(
        model="google/gemini-3-pro-preview",
        messages=[
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": problem}
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    
    # Parse JSON
    raw_response = completion.choices[0].message.content
    data = json.loads(raw_response)
    
    # Validate with Pydantic (this will raise errors if schema is wrong)
    state = StateObject(**data['state'])
    plan = Plan(**data['plan'])
    
    return state, plan
