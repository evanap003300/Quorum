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
