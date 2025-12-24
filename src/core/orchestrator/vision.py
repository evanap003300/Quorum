import os
import base64
import sys
import json
from pathlib import Path
from typing import Tuple, List, Union, Dict, Any
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from prompts.vision import VISION_PROMPT, ENHANCED_VISION_PROMPT
from config.pricing import MODEL_PRICING

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB limit

# Vision-specific pricing (image cost per image, model costs from MODEL_PRICING)
VISION_IMAGE_COST = 0.003613  # $0.003613 per image

# Computer Vision Tools - Available to LLM via function calling
# These are the 8 high-impact tools optimized for latency while maintaining accuracy
CV_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_image_metadata",
            "description": "Inspect image resolution and quality. Check if image is suitable for cropping/preprocessing. Use this FIRST to assess image viability.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_content_regions",
            "description": "Automatically detect text blocks and diagrams in the image. Returns recommended crop regions and which grid cells contain content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_area": {
                        "type": "integer",
                        "description": "Minimum region area in pixels (default 1000)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_shadows_and_artifacts",
            "description": "Analyze image for quality issues like shadows, blur, or faint text. Returns list of detected problems and tool recommendations to fix them.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_grid",
            "description": "Overlay a 10x10 coordinate grid with alphanumeric labels (rows A-J, columns 0-9) onto the image. Enables spatial reference. Grid cells referenced as 'D5' etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "grid_size": {
                        "type": "integer",
                        "description": "Number of grid divisions per side (default 10)"
                    },
                    "line_color": {
                        "type": "string",
                        "enum": ["red", "blue", "yellow", "green", "cyan", "magenta"],
                        "description": "Color of grid lines (default red)"
                    },
                    "alpha": {
                        "type": "number",
                        "description": "Transparency 0.0-1.0 (default 0.5)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "crop_grid_square",
            "description": "Zoom into a specific grid square identified by alphanumeric reference (e.g., 'D5'). Include surrounding context with margin.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cell_ref": {
                        "type": "string",
                        "pattern": "^[A-J][0-9]$",
                        "description": "Grid cell reference, e.g., 'D5' (row A-J, column 0-9)"
                    },
                    "margin": {
                        "type": "integer",
                        "description": "Pixels to extend beyond cell for context (default 200)"
                    }
                },
                "required": ["cell_ref"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "crop_quadrant",
            "description": "Quick crop to 25% section of image. Use when you want to focus on a corner or specific quadrant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "quadrant": {
                        "type": "string",
                        "enum": ["top_left", "top_right", "bottom_left", "bottom_right", "center"],
                        "description": "Which section to crop"
                    }
                },
                "required": ["quadrant"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "binarize_image",
            "description": "Convert image to strict black and white using adaptive thresholding. Removes shadows, reduces texture, improves text clarity. Use when image has poor lighting or shadows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "block_size": {
                        "type": "integer",
                        "description": "Block size for local threshold calculation (must be odd, 3-21, default 11)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "enhance_clarity",
            "description": "Boost contrast and sharpen image. Use when text is faint, washed out, or blurry.",
            "parameters": {
                "type": "object",
                "properties": {
                    "contrast": {
                        "type": "boolean",
                        "description": "Enable contrast enhancement (default true)"
                    },
                    "contrast_factor": {
                        "type": "number",
                        "description": "Contrast strength multiplier (default 1.5, range 0.5-3.0)"
                    },
                    "sharpen": {
                        "type": "boolean",
                        "description": "Enable sharpening (default true)"
                    },
                    "sharpen_factor": {
                        "type": "number",
                        "description": "Sharpening strength (default 2.0, range 0.5-3.0)"
                    }
                },
                "required": []
            }
        }
    }
]


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

    # Prefer direct OpenAI API for vision, fall back to OpenRouter
    if os.getenv("OPENAI_API_KEY"):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = "gpt-4o"
    else:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPEN_ROUTER_KEY")
        )
        model = "gpt-4.1-mini-2025-04-14"

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
    pricing = MODEL_PRICING.get(model, {"input": 0, "output": 0})
    usage = completion.usage

    input_cost = (usage.prompt_tokens * pricing["input"]) / 1_000_000
    output_cost = (usage.completion_tokens * pricing["output"]) / 1_000_000
    image_cost = VISION_IMAGE_COST

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


def _execute_cv_tool(tool_name: str, image: Image.Image, args: dict) -> Union[Image.Image, Tuple[Image.Image, Dict], Dict]:
    """
    Execute a CV tool and return result.

    Args:
        tool_name: Name of the CV tool to execute
        image: PIL Image object
        args: Arguments for the tool

    Returns:
        - Image if tool modifies image (crop, enhance, etc.)
        - Tuple(Image, metadata dict) if tool modifies image and returns metadata
        - Dict with metadata if tool only analyzes (no image modification)
    """
    from tools.vision import (
        get_image_metadata,
        detect_content_regions,
        detect_shadows_and_artifacts,
        apply_grid,
        crop_grid_square,
        crop_quadrant,
        binarize_image,
        enhance_clarity
    )

    tool_map = {
        "get_image_metadata": lambda: get_image_metadata(image),
        "detect_content_regions": lambda: detect_content_regions(image, **args),
        "detect_shadows_and_artifacts": lambda: detect_shadows_and_artifacts(image),
        "apply_grid": lambda: apply_grid(image, **args),
        "crop_grid_square": lambda: crop_grid_square(image, **args),
        "crop_quadrant": lambda: crop_quadrant(image, **args),
        "binarize_image": lambda: binarize_image(image, **args),
        "enhance_clarity": lambda: enhance_clarity(image, **args),
    }

    if tool_name not in tool_map:
        raise ValueError(f"Unknown CV tool: {tool_name}")

    try:
        return tool_map[tool_name]()
    except Exception as e:
        # Return error as metadata
        return {"error": str(e), "tool": tool_name, "error_type": type(e).__name__}


def _save_intermediate_image(
    image: Image.Image,
    original_path: str,
    step: int,
    tool_name: str
) -> str:
    """
    Save intermediate image to disk for debugging.

    Args:
        image: PIL Image to save
        original_path: Path to original image (used to determine output directory)
        step: Step number
        tool_name: Name of tool that produced this image

    Returns:
        Path to saved image
    """
    path = Path(original_path)
    stem = path.stem
    ext = path.suffix
    parent = path.parent

    # Format: original_name_step01_tool_name.png
    output_name = f"{stem}_step{step:02d}_{tool_name}.png"
    output_path = parent / output_name

    try:
        image.save(output_path)
        return str(output_path)
    except Exception as e:
        print(f"Warning: Could not save intermediate image {output_path}: {e}")
        return None


def _encode_image_to_base64_pil(image: Image.Image) -> Tuple[str, str]:
    """
    Encode PIL Image to base64 string.

    Args:
        image: PIL Image object

    Returns:
        Tuple of (base64_string, mime_type)
    """
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return encoded, "image/png"


def _calculate_vision_cost(completion, model: str) -> float:
    """
    Calculate cost for a vision completion.

    Args:
        completion: OpenAI completion response
        model: Model name

    Returns:
        Cost in USD
    """
    pricing = MODEL_PRICING.get(model, {"input": 0, "output": 0})
    usage = completion.usage

    input_cost = (usage.prompt_tokens * pricing["input"]) / 1_000_000
    output_cost = (usage.completion_tokens * pricing["output"]) / 1_000_000

    return input_cost + output_cost


def analyze_problem_image_with_cv_tools(
    image_path: str,
    max_iterations: int = 10,
    save_intermediates: bool = True
) -> Tuple[str, str, float, List[str]]:
    """
    Analyze a physics problem image using GPT-4o with CV tools via function calling.

    The LLM can iteratively preprocess the image (crop, enhance, detect regions) before
    extracting the problem text. Intermediate images are saved for debugging.

    Args:
        image_path: Path to image file
        max_iterations: Maximum number of tool calls before forcing extraction (default 10)
        save_intermediates: Save intermediate images to disk (default True)

    Returns:
        Tuple of (problem_text, diagram_context, cost, intermediate_image_paths)
        - problem_text: Extracted problem statement
        - diagram_context: Description of diagrams ("None" if no diagrams)
        - cost: Total USD cost of analysis
        - intermediate_image_paths: List of saved intermediate image paths

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If analysis fails or max iterations reached
    """
    try:
        # Validate image
        _validate_image(image_path)

        # Load image
        current_image = Image.open(image_path).convert('RGB')
        intermediate_paths = []

        # Initialize OpenAI client
        load_dotenv()
        if os.getenv("OPENAI_API_KEY"):
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model = "gpt-4o"
        else:
            # Fallback to OpenRouter
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPEN_ROUTER_KEY")
            )
            model = "openai/gpt-4o"

        # Encode initial image
        base64_image, mime_type = _encode_image_to_base64_pil(current_image)

        # Initialize messages
        messages = [
            {
                "role": "system",
                "content": ENHANCED_VISION_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this physics problem image. Use the CV tools as needed to improve image quality and focus on relevant regions before extracting the problem."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        total_cost = VISION_IMAGE_COST  # Initial image encoding cost
        step_count = 0

        for iteration in range(max_iterations):
            # Call GPT-4o with CV tools
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=CV_TOOLS,
                temperature=0.1,
                max_tokens=3000
            )

            # Track cost
            total_cost += _calculate_vision_cost(completion, model)

            choice = completion.choices[0]
            assistant_message = choice.message

            # Check if tools were called
            if choice.finish_reason == "tool_calls" and assistant_message.tool_calls:
                # Append assistant message to history
                messages.append(assistant_message)

                # Execute each tool call
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    step_count += 1

                    try:
                        # Execute CV tool
                        tool_result = _execute_cv_tool(tool_name, current_image, args)

                        # Handle different return types from CV tools
                        if isinstance(tool_result, tuple) and isinstance(tool_result[0], Image.Image):
                            # Tool returned (image, metadata)
                            current_image = tool_result[0]
                            metadata = tool_result[1]
                            result_text = json.dumps(metadata, indent=2)

                            # Save intermediate image
                            if save_intermediates:
                                saved_path = _save_intermediate_image(
                                    current_image, image_path, step_count, tool_name
                                )
                                if saved_path:
                                    intermediate_paths.append(saved_path)

                            # Re-encode updated image
                            base64_image, mime_type = _encode_image_to_base64_pil(current_image)
                            total_cost += VISION_IMAGE_COST  # Cost for new image

                            # Append tool result with updated image
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"Tool: {tool_name}\n\nResult:\n{result_text}"
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime_type};base64,{base64_image}"
                                        }
                                    }
                                ]
                            })

                        elif isinstance(tool_result, Image.Image):
                            # Tool returned just image (no metadata)
                            current_image = tool_result

                            if save_intermediates:
                                saved_path = _save_intermediate_image(
                                    current_image, image_path, step_count, tool_name
                                )
                                if saved_path:
                                    intermediate_paths.append(saved_path)

                            base64_image, mime_type = _encode_image_to_base64_pil(current_image)
                            total_cost += VISION_IMAGE_COST

                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"Tool: {tool_name}\n\nImage updated successfully."
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime_type};base64,{base64_image}"
                                        }
                                    }
                                ]
                            })

                        else:
                            # Tool returned metadata only (debugging/analysis tools)
                            result_text = json.dumps(tool_result, indent=2)

                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Tool: {tool_name}\n\nResult:\n{result_text}"
                            })

                    except Exception as e:
                        # Tool execution failed - report to LLM
                        error_msg = f"Tool execution failed: {str(e)}"
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": error_msg
                        })

                # Continue loop for next iteration
                continue

            else:
                # No tool calls - LLM is ready to extract
                if assistant_message.content:
                    try:
                        problem_text, diagram_context = _parse_vision_response(
                            assistant_message.content
                        )
                        return problem_text, diagram_context, total_cost, intermediate_paths

                    except ValueError as e:
                        # Parsing failed - retry with format reminder
                        if iteration < max_iterations - 1:
                            messages.append({
                                "role": "assistant",
                                "content": assistant_message.content
                            })
                            messages.append({
                                "role": "user",
                                "content": "Please provide your response in the required format:\n\nPROBLEM TEXT:\n[Complete problem statement with all values and units]\n\nDIAGRAM CONTEXT:\n[Diagram description, or 'None' if no diagrams]"
                            })
                            continue
                        else:
                            raise ValueError(f"Failed to parse vision response: {e}")

        # Max iterations reached
        raise ValueError(
            f"Max tool iterations ({max_iterations}) reached without successful extraction. "
            f"Last response did not match required format."
        )

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Image file error: {e}")
    except ValueError as e:
        raise ValueError(f"Vision analysis error: {e}")
    except Exception as e:
        raise Exception(f"Vision API call failed: {e}")


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


def analyze_observation(
    image_path: str,
    question: str,
    model: str = "gemini-3-flash-preview",
    max_iterations: int = 5
) -> Tuple[Union[str, float], str, float]:
    """
    Query Gemini Flash vision model for a specific observation about the image.
    Used by OBSERVE operation during solver execution.

    The vision model can use CV tools (crop, enhance, detect regions, etc.)
    to improve its view before answering the observation question.

    Args:
        image_path: Path to the image file
        question: The specific question to ask (from build_observe_prompt)
        model: Vision model to use (default: Gemini Flash for speed)
        max_iterations: Max tool calls before forcing answer (default: 5)

    Returns:
        Tuple of (value, unit, cost)
        - value: Qualitative observation (e.g., "positive", "diverging")
        - unit: Unit from response (usually "dimensionless")
        - cost: API cost for this call
    """
    try:
        print(f"DEBUG: analyze_observation starting with question length={len(question)}")
        import google.generativeai as genai

        # Load and validate image
        if not os.path.exists(image_path):
            raise ValueError(f"Image not found: {image_path}")

        current_image = Image.open(image_path)
        total_cost = 0.0

        # Define CV tools for observation via function declarations
        from tools.vision import (
            get_image_metadata,
            detect_content_regions,
            apply_grid,
            crop_grid_square,
            crop_quadrant,
            binarize_image,
            enhance_clarity
        )

        # Create function declarations for Gemini
        from google.generativeai.types import FunctionDeclaration, Tool

        cv_tool_functions = [
            FunctionDeclaration(
                name="get_image_metadata",
                description="Get image dimensions and quality assessment",
                parameters={"type": "object", "properties": {}, "required": []}
            ),
            FunctionDeclaration(
                name="detect_content_regions",
                description="Detect text and diagram regions in the image",
                parameters={
                    "type": "object",
                    "properties": {
                        "min_area": {"type": "integer", "description": "Minimum region area"}
                    }
                }
            ),
            FunctionDeclaration(
                name="apply_grid",
                description="Overlay grid with alphanumeric labels (A0-J9)",
                parameters={
                    "type": "object",
                    "properties": {
                        "grid_size": {"type": "integer", "description": "Grid size (default 10)"},
                        "line_color": {"type": "string", "enum": ["red", "blue", "green"]}
                    }
                }
            ),
            FunctionDeclaration(
                name="crop_grid_square",
                description="Crop to specific grid cell (e.g., 'D5')",
                parameters={
                    "type": "object",
                    "properties": {
                        "cell_ref": {"type": "string", "description": "Grid cell reference"},
                        "margin": {"type": "integer", "description": "Margin around cell"}
                    },
                    "required": ["cell_ref"]
                }
            ),
            FunctionDeclaration(
                name="crop_quadrant",
                description="Crop to 25% section of image",
                parameters={
                    "type": "object",
                    "properties": {
                        "quadrant": {"type": "string", "enum": ["top_left", "top_right", "bottom_left", "bottom_right", "center"]}
                    },
                    "required": ["quadrant"]
                }
            ),
            FunctionDeclaration(
                name="binarize_image",
                description="Convert to black and white to improve clarity",
                parameters={
                    "type": "object",
                    "properties": {
                        "block_size": {"type": "integer", "description": "Adaptive threshold block size"}
                    }
                }
            ),
            FunctionDeclaration(
                name="enhance_clarity",
                description="Boost contrast and sharpen image",
                parameters={
                    "type": "object",
                    "properties": {
                        "contrast": {"type": "boolean"},
                        "contrast_factor": {"type": "number"},
                        "sharpen": {"type": "boolean"},
                        "sharpen_factor": {"type": "number"}
                    }
                }
            )
        ]

        # Create Tool object for Gemini
        cv_tools = Tool(function_declarations=cv_tool_functions)

        # Tool execution map
        tool_map = {
            "get_image_metadata": lambda args: get_image_metadata(current_image),
            "detect_content_regions": lambda args: detect_content_regions(current_image, **args),
            "apply_grid": lambda args: apply_grid(current_image, **args),
            "crop_grid_square": lambda args: crop_grid_square(current_image, **args),
            "crop_quadrant": lambda args: crop_quadrant(current_image, **args),
            "binarize_image": lambda args: binarize_image(current_image, **args),
            "enhance_clarity": lambda args: enhance_clarity(current_image, **args)
        }

        # Initialize Gemini with tools and JSON output
        load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        vision_model = genai.GenerativeModel(
            model_name=model,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 500,
                "response_mime_type": "application/json"
            },
            tools=[cv_tools]  # ENABLE TOOL CALLING
        )

        # Create system prompt
        system_prompt = f"""You are analyzing a physics diagram to extract specific visual properties.

You have access to CV tools to improve your view:
- apply_grid + crop_grid_square: Zoom into specific regions
- crop_quadrant: Quick crop to 25% sections
- enhance_clarity: Improve faint or blurry features
- binarize_image: Remove shadows, improve text

WORKFLOW:
1. If the feature you need to observe is small or unclear, use CV tools to get a better view
2. Once you have a clear view, answer the observation question
3. Respond in JSON format with qualitative description

RESPOND WITH THIS JSON STRUCTURE:
{{
  "value": "<qualitative observation (e.g., 'diverging', 'increasing', 'positive')>",
  "description": "<detailed explanation of what you observed>",
  "unit": "dimensionless"
}}

{question}

IMPORTANT: Respond with valid JSON only. No other text.
"""

        # Start chat session
        chat = vision_model.start_chat()
        response = chat.send_message([system_prompt, current_image])

        # Calculate initial cost
        pricing = MODEL_PRICING.get(model, {"input": 0, "output": 0})
        usage = response.usage_metadata
        total_cost += (usage.prompt_token_count * pricing["input"] +
                       usage.candidates_token_count * pricing["output"]) / 1_000_000

        # Tool calling loop
        for iteration in range(max_iterations):
            # Check if model wants to use tools
            if (response.candidates and
                response.candidates[0].content.parts and
                hasattr(response.candidates[0].content.parts[0], 'function_call')):

                function_call = response.candidates[0].content.parts[0].function_call
                tool_name = function_call.name
                args = dict(function_call.args)

                # Execute CV tool
                if tool_name in tool_map:
                    try:
                        result = tool_map[tool_name](args)

                        # Handle different return types
                        if isinstance(result, tuple) and isinstance(result[0], Image.Image):
                            current_image = result[0]
                            metadata = result[1]
                            tool_response = {"status": "success", "metadata": metadata}
                        elif isinstance(result, Image.Image):
                            current_image = result
                            tool_response = {"status": "success", "message": "Image updated"}
                        else:
                            tool_response = result

                        # Send tool result + updated image back to model
                        response = chat.send_message([
                            genai.protos.Part(function_response=genai.protos.FunctionResponse(
                                name=tool_name,
                                response=tool_response
                            )),
                            current_image
                        ])

                        # Track cost
                        usage = response.usage_metadata
                        total_cost += (usage.prompt_token_count * pricing["input"] +
                                       usage.candidates_token_count * pricing["output"]) / 1_000_000

                    except Exception as e:
                        # Tool execution failed, send error
                        response = chat.send_message(
                            genai.protos.Part(function_response=genai.protos.FunctionResponse(
                                name=tool_name,
                                response={"status": "error", "message": str(e)}
                            ))
                        )
                        usage = response.usage_metadata
                        total_cost += (usage.prompt_token_count * pricing["input"] +
                                       usage.candidates_token_count * pricing["output"]) / 1_000_000
            else:
                # Model provided JSON response - parse it
                try:
                    if response.text:
                        try:
                            response_json = json.loads(response.text)
                            value_with_unit = response_json.get("value", "")

                            # Split value and unit (last space separates them)
                            if " " in value_with_unit:
                                parts = value_with_unit.rsplit(" ", 1)
                                value = parts[0]
                                unit = parts[1]
                            else:
                                value = value_with_unit
                                unit = "dimensionless"

                            return value, unit, total_cost
                        except json.JSONDecodeError as parse_err:
                            # JSON parse failed, fallback to raw text
                            print(f"⚠ Failed to parse OBSERVE response as JSON: {parse_err}")
                            print(f"  Raw response: {response.text[:200]}")
                            output_text = response.text.strip()
                            return output_text, "dimensionless", total_cost
                    else:
                        print(f"⚠ Empty OBSERVE response text")
                        return "", "dimensionless", total_cost
                except Exception as e:
                    print(f"⚠ Error in OBSERVE response handling: {e}")
                    return "", "dimensionless", total_cost

        # Max iterations reached - return best effort response
        try:
            if response and response.text:
                try:
                    response_json = json.loads(response.text)
                    value_with_unit = response_json.get("value", "")

                    # Split value and unit (last space separates them)
                    if " " in value_with_unit:
                        parts = value_with_unit.rsplit(" ", 1)
                        value = parts[0]
                        unit = parts[1]
                    else:
                        value = value_with_unit
                        unit = "dimensionless"

                    return value, unit, total_cost
                except json.JSONDecodeError as parse_err:
                    # Fallback to raw text
                    print(f"⚠ Max iterations reached - failed to parse JSON: {parse_err}")
                    output_text = response.text.strip()
                    return output_text, "dimensionless", total_cost
            else:
                print(f"⚠ No response or empty text at max iterations")
                return "", "dimensionless", total_cost
        except Exception as e:
            print(f"⚠ Error processing max-iterations response: {e}")
            return "", "dimensionless", total_cost

    except Exception as e:
        import traceback
        print(f"⚠ FATAL error in analyze_observation: {e}")
        traceback.print_exc()
        raise Exception(f"analyze_observation failed: {str(e)}")


def analyze_observation_multi(
    image_path: str,
    question: str,
    var_names: List[str],
    step: Any = None,
    state: Any = None,
    model: str = "gemini-3-flash-preview",
    max_iterations: int = 5
) -> Tuple[Dict[str, Union[str, float]], Dict[str, str], float]:
    """
    Query Gemini Flash vision model for multiple observations about the image.
    The vision model can use CV tools to improve its view.

    Args:
        image_path: Path to the image file
        question: The question with multiple variable requests
        var_names: List of variable names to extract
        step: The current step (optional, for context)
        state: Current state (optional, for context)
        model: Vision model to use
        max_iterations: Max tool calls before forcing answer

    Returns:
        Tuple of (values_dict, units_dict, cost)
    """
    try:
        print(f"DEBUG: analyze_observation_multi called with var_names={var_names}")
        import google.generativeai as genai

        # Load image
        if not os.path.exists(image_path):
            raise ValueError(f"Image not found: {image_path}")

        print(f"DEBUG: Image loaded from {image_path}")
        current_image = Image.open(image_path)
        total_cost = 0.0

        # Define CV tools
        from tools.vision import (
            get_image_metadata,
            detect_content_regions,
            apply_grid,
            crop_grid_square,
            crop_quadrant,
            binarize_image,
            enhance_clarity
        )

        # Create function declarations
        from google.generativeai.types import FunctionDeclaration, Tool

        cv_tool_functions = [
            FunctionDeclaration(
                name="get_image_metadata",
                description="Get image dimensions and quality assessment",
                parameters={"type": "object", "properties": {}}
            ),
            FunctionDeclaration(
                name="detect_content_regions",
                description="Detect text and diagram regions",
                parameters={
                    "type": "object",
                    "properties": {"min_area": {"type": "integer"}}
                }
            ),
            FunctionDeclaration(
                name="apply_grid",
                description="Overlay grid (A0-J9)",
                parameters={
                    "type": "object",
                    "properties": {
                        "grid_size": {"type": "integer"},
                        "line_color": {"type": "string", "enum": ["red", "blue", "green"]}
                    }
                }
            ),
            FunctionDeclaration(
                name="crop_grid_square",
                description="Crop to grid cell",
                parameters={
                    "type": "object",
                    "properties": {
                        "cell_ref": {"type": "string"},
                        "margin": {"type": "integer"}
                    },
                    "required": ["cell_ref"]
                }
            ),
            FunctionDeclaration(
                name="crop_quadrant",
                description="Crop to 25% section",
                parameters={
                    "type": "object",
                    "properties": {
                        "quadrant": {"type": "string", "enum": ["top_left", "top_right", "bottom_left", "bottom_right", "center"]}
                    },
                    "required": ["quadrant"]
                }
            ),
            FunctionDeclaration(
                name="binarize_image",
                description="Convert to B&W",
                parameters={
                    "type": "object",
                    "properties": {"block_size": {"type": "integer"}}
                }
            ),
            FunctionDeclaration(
                name="enhance_clarity",
                description="Boost contrast and sharpen",
                parameters={
                    "type": "object",
                    "properties": {
                        "contrast": {"type": "boolean"},
                        "contrast_factor": {"type": "number"},
                        "sharpen": {"type": "boolean"},
                        "sharpen_factor": {"type": "number"}
                    }
                }
            )
        ]

        # Create Tool object for Gemini
        cv_tools = Tool(function_declarations=cv_tool_functions)

        # Tool map
        tool_map = {
            "get_image_metadata": lambda args: get_image_metadata(current_image),
            "detect_content_regions": lambda args: detect_content_regions(current_image, **args),
            "apply_grid": lambda args: apply_grid(current_image, **args),
            "crop_grid_square": lambda args: crop_grid_square(current_image, **args),
            "crop_quadrant": lambda args: crop_quadrant(current_image, **args),
            "binarize_image": lambda args: binarize_image(current_image, **args),
            "enhance_clarity": lambda args: enhance_clarity(current_image, **args)
        }

        # Initialize Gemini with JSON response format
        load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        vision_model = genai.GenerativeModel(
            model_name=model,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 1000,
                "response_mime_type": "application/json"
            },
            tools=[cv_tools]  # ENABLE TOOL CALLING
        )

        # Create prompt
        system_prompt = f"""You are analyzing a physics diagram to extract specific visual properties.

You have CV tools available to improve your view before answering.

{question}

RESPOND WITH THIS JSON STRUCTURE FOR MULTIPLE VARIABLES:
{{
  "var_name_1": {{
    "value": "<qualitative observation>",
    "description": "<detailed explanation>",
    "unit": "dimensionless"
  }},
  "var_name_2": {{
    "value": "<qualitative observation>",
    "description": "<detailed explanation>",
    "unit": "dimensionless"
  }}
}}

IMPORTANT: Respond with valid JSON only. No other text.
"""

        # Start chat
        print(f"DEBUG: Starting chat session for multi-output observation")
        chat = vision_model.start_chat()
        print(f"DEBUG: Sending message to vision model")
        response = chat.send_message([system_prompt, current_image])
        print(f"DEBUG: Got response, type={type(response)}, response={response}")

        # Track cost
        pricing = MODEL_PRICING.get(model, {"input": 0, "output": 0})
        print(f"DEBUG: Getting usage metadata")
        usage = response.usage_metadata
        print(f"DEBUG: Got usage, tokens={usage.prompt_token_count}")
        total_cost += (usage.prompt_token_count * pricing["input"] +
                       usage.candidates_token_count * pricing["output"]) / 1_000_000

        # Tool calling loop
        print(f"DEBUG: Starting tool calling loop")
        for iteration in range(max_iterations):
            print(f"DEBUG: Tool loop iteration {iteration}")
            if (response.candidates and
                response.candidates[0].content.parts and
                hasattr(response.candidates[0].content.parts[0], 'function_call')):

                function_call = response.candidates[0].content.parts[0].function_call
                tool_name = function_call.name
                args = dict(function_call.args)

                if tool_name in tool_map:
                    try:
                        result = tool_map[tool_name](args)

                        if isinstance(result, tuple) and isinstance(result[0], Image.Image):
                            current_image = result[0]
                            metadata = result[1]
                            tool_response = {"status": "success", "metadata": metadata}
                        elif isinstance(result, Image.Image):
                            current_image = result
                            tool_response = {"status": "success"}
                        else:
                            tool_response = result

                        response = chat.send_message([
                            genai.protos.Part(function_response=genai.protos.FunctionResponse(
                                name=tool_name,
                                response=tool_response
                            )),
                            current_image
                        ])

                        usage = response.usage_metadata
                        total_cost += (usage.prompt_token_count * pricing["input"] +
                                       usage.candidates_token_count * pricing["output"]) / 1_000_000

                    except Exception as e:
                        response = chat.send_message(
                            genai.protos.Part(function_response=genai.protos.FunctionResponse(
                                name=tool_name,
                                response={"status": "error", "message": str(e)}
                            ))
                        )
                        usage = response.usage_metadata
                        total_cost += (usage.prompt_token_count * pricing["input"] +
                                       usage.candidates_token_count * pricing["output"]) / 1_000_000
            else:
                # Parse multi-output JSON response
                try:
                    if response.text:
                        response_json = json.loads(response.text)

                        # Extract values and units for each variable
                        values_dict = {}
                        units_dict = {}

                        for var_name in var_names:
                            if var_name in response_json:
                                value_with_unit = response_json[var_name]

                                # Split value and unit (last space separates them)
                                if isinstance(value_with_unit, str) and " " in value_with_unit:
                                    parts = value_with_unit.rsplit(" ", 1)
                                    value = parts[0]
                                    unit = parts[1]
                                else:
                                    value = str(value_with_unit)
                                    unit = "dimensionless"

                                values_dict[var_name] = value
                                units_dict[var_name] = unit
                            else:
                                values_dict[var_name] = None
                                units_dict[var_name] = "unknown"

                        return values_dict, units_dict, total_cost
                    else:
                        # Empty response
                        values_dict = {v: None for v in var_names}
                        units_dict = {v: "unknown" for v in var_names}
                        return values_dict, units_dict, total_cost
                except (json.JSONDecodeError, ValueError, KeyError) as parse_error:
                    # JSON parsing failed - log for debugging
                    print(f"⚠ JSON parse error in OBSERVE response: {parse_error}")
                    if response.text:
                        print(f"  Raw response: {response.text[:200]}")
                    # Fallback: return None for all variables
                    values_dict = {v: None for v in var_names}
                    units_dict = {v: "unknown" for v in var_names}
                    return values_dict, units_dict, total_cost

        # Max iterations reached - try to parse final response
        try:
            if response and response.text:
                try:
                    response_json = json.loads(response.text)
                except json.JSONDecodeError as parse_err:
                    print(f"⚠ Failed to parse final OBSERVE response as JSON: {parse_err}")
                    print(f"  Raw response: {response.text[:200]}")
                    # Return None for all variables
                    values_dict = {v: None for v in var_names}
                    units_dict = {v: "unknown" for v in var_names}
                    return values_dict, units_dict, total_cost

                values_dict = {}
                units_dict = {}

                for var_name in var_names:
                    if var_name in response_json:
                        value_with_unit = response_json[var_name]

                        # Split value and unit (last space separates them)
                        if isinstance(value_with_unit, str) and " " in value_with_unit:
                            parts = value_with_unit.rsplit(" ", 1)
                            value = parts[0]
                            unit = parts[1]
                        else:
                            value = str(value_with_unit)
                            unit = "dimensionless"

                        values_dict[var_name] = value
                        units_dict[var_name] = unit
                    else:
                        values_dict[var_name] = None
                        units_dict[var_name] = "unknown"

                return values_dict, units_dict, total_cost
            else:
                print(f"⚠ No response or empty response text from OBSERVE")
                values_dict = {v: None for v in var_names}
                units_dict = {v: "unknown" for v in var_names}
                return values_dict, units_dict, total_cost
        except (ValueError, KeyError) as e:
            print(f"⚠ Error processing OBSERVE final response: {e}")
            values_dict = {v: None for v in var_names}
            units_dict = {v: "unknown" for v in var_names}
            return values_dict, units_dict, total_cost

    except Exception as e:
        import traceback
        print(f"⚠ FATAL error in analyze_observation_multi: {e}")
        traceback.print_exc()
        raise Exception(f"analyze_observation_multi failed: {str(e)}")
