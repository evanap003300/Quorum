import os
import base64
from pathlib import Path
from typing import Tuple
from dotenv import load_dotenv
from openai import OpenAI

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB limit

# Model pricing (per 1M tokens)
MODEL_PRICING = {
    "openai/gpt-4o": {
        "input": 2.50,      # $2.50 per 1M input tokens
        "output": 10.0,     # $10 per 1M output tokens
        "image": 0.003613   # $0.003613 per image
    }
}

VISION_PROMPT = """Analyze this physics or mathematics problem image and extract all information.

**PROBLEM TEXT**: The complete textual problem statement
- Include all given values with units
- Include the question being asked
- Preserve mathematical notation as plain text

**DIAGRAM CONTEXT**: Description of visual elements
- Diagrams, graphs, or figures
- Geometric configurations
- Visual relationships between elements
- Labels and annotations

Output EXACTLY in this format:

PROBLEM TEXT:
[Complete problem statement here]

DIAGRAM CONTEXT:
[Description of visual elements, or "None" if no diagrams]

Be precise and complete. Extract ALL information visible in the image."""


def _validate_image(image_path: str) -> None:
    """Validate image exists and is supported format."""
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: {path.suffix}. "
            f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )

    file_size = path.stat().st_size
    if file_size > MAX_IMAGE_SIZE:
        raise ValueError(
            f"Image too large: {file_size / 1024 / 1024:.1f}MB (max: {MAX_IMAGE_SIZE / 1024 / 1024:.0f}MB)"
        )


def _encode_image_to_base64(image_path: str) -> Tuple[str, str]:
    """
    Encode image to base64 for API transmission.

    Returns:
        Tuple of (base64_string, mime_type)
    """
    path = Path(image_path)

    # Determine MIME type
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    mime_type = mime_types.get(path.suffix.lower(), 'image/jpeg')

    # Read and encode
    with open(image_path, 'rb') as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')

    return encoded, mime_type


def _call_vision_api(base64_image: str, mime_type: str) -> Tuple[str, float]:
    """
    Call GPT-4o vision API to analyze image.

    Returns:
        Tuple of (response_text, cost)
    """
    load_dotenv()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_KEY")
    )

    model = "openai/gpt-4o"

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": VISION_PROMPT
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        temperature=0.1,
        max_tokens=2000
    )

    # Calculate cost
    pricing = MODEL_PRICING[model]
    usage = completion.usage

    input_cost = (usage.prompt_tokens * pricing["input"]) / 1_000_000
    output_cost = (usage.completion_tokens * pricing["output"]) / 1_000_000
    image_cost = pricing["image"]

    total_cost = input_cost + output_cost + image_cost

    response_text = completion.choices[0].message.content
    return response_text, total_cost


def _parse_vision_response(response: str) -> Tuple[str, str]:
    """
    Parse structured response from vision API.

    Returns:
        Tuple of (problem_text, diagram_context)

    Raises:
        ValueError: If response format is invalid
    """
    # Split by section markers
    sections = response.split("DIAGRAM CONTEXT:")

    if len(sections) != 2:
        raise ValueError(
            f"Invalid response format - missing DIAGRAM CONTEXT section. "
            f"Response preview: {response[:200]}"
        )

    # Extract problem text (remove "PROBLEM TEXT:" header)
    problem_section = sections[0].replace("PROBLEM TEXT:", "").strip()

    # Extract diagram context
    diagram_section = sections[1].strip()

    # Validate non-empty problem text
    if not problem_section:
        raise ValueError("Problem text section is empty")

    # Normalize "None" for diagram context
    if diagram_section.lower() in ("none", "none.", "n/a", "not applicable"):
        diagram_section = "None"

    return problem_section, diagram_section


def analyze_problem_image(image_path: str) -> Tuple[str, str, float]:
    """
    Analyze a problem image using GPT-4o vision API.

    Args:
        image_path: Absolute path to image file (jpg, png, etc.)

    Returns:
        Tuple of (problem_text, diagram_context, cost)
        - problem_text: Extracted text describing the problem
        - diagram_context: Description of visual/diagram elements (or "None")
        - cost: USD cost of vision API call

    Raises:
        FileNotFoundError: If image_path doesn't exist
        ValueError: If image format is unsupported or response is malformed
        Exception: If API call fails
    """

    try:
        # Validate image
        _validate_image(image_path)

        # Encode to base64
        base64_image, mime_type = _encode_image_to_base64(image_path)

        # Call vision API
        response, cost = _call_vision_api(base64_image, mime_type)

        # Parse response
        problem_text, diagram_context = _parse_vision_response(response)

        return problem_text, diagram_context, cost

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Image file error: {e}")
    except ValueError as e:
        raise ValueError(f"Image validation or parsing error: {e}")
    except Exception as e:
        raise Exception(f"Vision API call failed: {e}")
