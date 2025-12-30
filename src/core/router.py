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


ROUTER_PROMPT = """Classify this physics problem as EASY, MEDIUM, or HARD. Output JSON only.

Tiers:
- EASY: Definitions, one-step formulas, unit conversion, recall
- MEDIUM: Textbook level, 2-4 steps, familiar patterns, homework
- HARD: Ambiguous, multi-step derivation, advanced topics (QM/GR), requires planning

List key features (1-3 words each). Tag reasoning as one phrase.

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
        model = genai.GenerativeModel(
            model_name="gemini-3-flash-preview",
            generation_config={
                "temperature": 0.0,  # Deterministic output
                "response_mime_type": "application/json",
                "max_output_tokens": 150  # Force brevity (~5-10 tokens for reasoning + 2-5 for indicators)
            }
        )

        # Generate classification
        response = model.generate_content([ROUTER_PROMPT, f"\n\nProblem:\n{problem}"])

        # Parse structured output
        result_json = json.loads(response.text)
        classification = ProblemClassification(**result_json)

        # Calculate cost
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count
        output_tokens = usage.candidates_token_count

        cost = 0.0
        if "gemini-3-flash-preview" in MODEL_PRICING:
            pricing = MODEL_PRICING["gemini-3-flash-preview"]
            cost = ((input_tokens * pricing["input"]) + (output_tokens * pricing["output"])) / 1_000_000

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
