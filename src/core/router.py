"""Physics problem difficulty router using Gemini 3.0 Flash."""

from enum import Enum
from typing import Optional, Tuple
import json
import os
import sys

from pydantic import BaseModel, Field

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# Add pricing config to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'orchestrator', 'config'))
from pricing import MODEL_PRICING


class DifficultyTier(str, Enum):
    """Problem difficulty tiers."""
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"


class ProblemClassification(BaseModel):
    """Structured output from router."""
    tier: DifficultyTier = Field(description="Difficulty tier: EASY, MEDIUM, or HARD")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in classification (0-1)")
    reasoning: str = Field(description="Why this tier was selected")
    key_indicators: list[str] = Field(description="Specific problem features that influenced classification")


ROUTER_PROMPT = """Classify this physics/chemistry/math problem difficulty. Output JSON only.

Tiers:
- EASY: Simple recall, single formula, unit conversion (e.g., "What is F=ma?", "Convert 72 km/h to m/s")
- MEDIUM: Standard textbook, 2-3 steps, explicit values given, familiar patterns
- HARD: Classify as HARD if ANY of these apply:
  * Thermodynamics (heat, work, entropy, enthalpy) - sign conventions are tricky
  * Multi-step derivation (4+ steps)
  * Requires choosing between approaches
  * Implicit constraints or approximations
  * Quantum mechanics, statistical mechanics, relativity
  * Problem says "derive", "show that", "prove"
  * Symbolic answer expected

When in doubt between MEDIUM and HARD, choose HARD.

{
  "tier": "EASY"|"MEDIUM"|"HARD",
  "confidence": <0-1>,
  "reasoning": "<single-phrase>",
  "key_indicators": ["feature1", "feature2"]
}"""


def classify_problem(problem: str, image_path: Optional[str] = None) -> Tuple[ProblemClassification, float]:
    """
    Classify problem difficulty using Gemini 3.0 Flash.

    Args:
        problem: Problem text
        image_path: Optional problem image (not used for routing currently)

    Returns:
        Tuple of (classification, cost_usd)
    """
    if not GENAI_AVAILABLE:
        print("WARNING: google.generativeai not available, defaulting to MEDIUM tier")
        fallback = ProblemClassification(
            tier=DifficultyTier.MEDIUM,
            confidence=0.0,
            reasoning="google.generativeai not available",
            key_indicators=["library_unavailable"]
        )
        return fallback, 0.0

    try:
        # Configure Gemini
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

        # Use Flash model for fast, cheap classification
        # Note: Removed response_mime_type as it may cause empty responses
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={
                "temperature": 0.0,  # Deterministic output
                "max_output_tokens": 200  # Allow some flexibility
            }
        )

        # Generate classification
        response = model.generate_content([ROUTER_PROMPT, f"\n\nProblem:\n{problem}"])

        # Validate response before parsing
        response_text = response.text if response.text else ""
        if not response_text.strip():
            raise ValueError(f"Empty response from router. Candidates: {response.candidates}")

        # Extract JSON from response (may have markdown code blocks or extra text)
        import re
        json_match = re.search(r'\{[^{}]*"tier"[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            result_json = json.loads(json_match.group())
        else:
            # Try parsing the whole response as JSON
            result_json = json.loads(response_text)

        classification = ProblemClassification(**result_json)

        # Calculate cost
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count
        output_tokens = usage.candidates_token_count

        cost = 0.0
        # Check for flash pricing (try multiple model name formats)
        flash_pricing = MODEL_PRICING.get("gemini-2.0-flash") or MODEL_PRICING.get("gemini-3-flash-preview")
        if flash_pricing:
            cost = ((input_tokens * flash_pricing["input"]) + (output_tokens * flash_pricing["output"])) / 1_000_000

        return classification, cost

    except Exception as e:
        # Fallback to MEDIUM tier on error (safe middle ground)
        print(f"Router error: {e}")
        fallback = ProblemClassification(
            tier=DifficultyTier.MEDIUM,
            confidence=0.5,
            reasoning=f"Router failed, defaulting to MEDIUM tier. Error: {str(e)[:100]}",
            key_indicators=["router_failure"]
        )
        return fallback, 0.0
