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


ROUTER_PROMPT = """You are a physics problem difficulty classifier. Analyze the given problem and classify it into one of three tiers:

## EASY Tier
Problems that are:
- Simple definitions or concept recall
- One-step arithmetic or formula application
- Direct unit conversions
- Looking up standard constants
- No derivation or multi-step reasoning needed

Examples:
- "What is Newton's Second Law?"
- "Convert 72 km/h to m/s"
- "If F=ma, m=5kg, and a=2m/s², find F"
- "What is the speed of light?"

## MEDIUM Tier
Problems that are:
- Standard textbook level (AP Physics, Calculus I-II)
- 2-4 clearly defined steps
- Multiple equations but familiar patterns
- Some algebra, basic calculus, or trigonometry
- Typical homework problems with clear solution path
- Solvable in 3-7 minutes by competent student

Examples:
- "A ball is thrown at 20 m/s at 45°. Find maximum height and range."
- "Calculate equivalent resistance of three resistors (10Ω, 20Ω, 30Ω) in parallel"
- "A 2kg mass on a spring oscillates with period 0.5s. Find spring constant."

## HARD Tier
Problems that are:
- Advanced undergraduate or graduate level
- Ambiguous problem statements requiring interpretation
- Multi-page derivations or proofs
- Multiple competing approaches needed
- Requires strategic planning and breakdown
- SciBench-level difficulty
- Problems with complex diagrams requiring spatial reasoning
- Advanced topics: quantum mechanics, relativity, statistical mechanics

Examples:
- "Derive partition function for a system of N non-interacting particles"
- "Find electric field everywhere for uniformly charged disk of radius R"
- Problems from physics olympiads or graduate qualifying exams

## Classification Guidelines
1. **Ambiguity**: If problem statement is ambiguous → HARD
2. **Diagram dependency**: If understanding requires complex diagram → HARD
3. **Calculation length**: If solution requires >4 distinct steps → HARD
4. **Domain complexity**: Advanced physics (QM, GR, stat mech) → usually HARD
5. **Confidence**: If borderline, err toward higher tier

Respond with JSON matching this schema:
{
  "tier": "EASY" | "MEDIUM" | "HARD",
  "confidence": <float 0-1>,
  "reasoning": "<why this tier>",
  "key_indicators": ["<indicator 1>", "<indicator 2>", ...]
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
                "temperature": 0.1,
                "response_mime_type": "application/json"
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
