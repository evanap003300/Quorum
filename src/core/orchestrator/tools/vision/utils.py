"""Shared utilities for Computer Vision tools."""

from typing import Union, Tuple, Dict, Any
from pathlib import Path
from PIL import Image
import os


def load_image(image_input: Union[str, Image.Image]) -> Image.Image:
    """
    Load an image from file path or return PIL Image object.

    Args:
        image_input: File path (str) or PIL Image object

    Returns:
        PIL Image object

    Raises:
        FileNotFoundError: If file path doesn't exist
        ValueError: If file is not a valid image
    """
    if isinstance(image_input, Image.Image):
        return image_input

    if not isinstance(image_input, str):
        raise ValueError(f"Expected str or PIL Image, got {type(image_input)}")

    path = Path(image_input)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_input}")

    try:
        img = Image.open(path)
        # Force load to ensure validity
        img.load()
        return img.convert('RGB')
    except Exception as e:
        raise ValueError(f"Failed to load image '{image_input}': {e}")


def validate_coordinates(
    img: Image.Image,
    x1: int,
    y1: int,
    x2: int,
    y2: int
) -> Tuple[int, int, int, int]:
    """
    Validate and clamp crop coordinates to image bounds.

    Args:
        img: PIL Image
        x1, y1: Top-left corner
        x2, y2: Bottom-right corner

    Returns:
        Tuple of clamped coordinates (x1, y1, x2, y2)

    Raises:
        ValueError: If coordinates are invalid
    """
    width, height = img.size

    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError(
            f"Coordinates ({x1}, {y1}, {x2}, {y2}) exceed image bounds ({width}, {height})"
        )

    if x1 >= x2 or y1 >= y2:
        raise ValueError(
            f"Invalid coordinates: top-left ({x1}, {y1}) must be before bottom-right ({x2}, {y2})"
        )

    return x1, y1, x2, y2


def grid_to_coordinates(
    img: Image.Image,
    cell_ref: str,
    grid_size: int = 10,
    margin: int = 200
) -> Tuple[int, int, int, int]:
    """
    Convert grid cell reference (e.g., "D5") to pixel coordinates.

    Grid layout: Rows A-J (top to bottom), Columns 0-9 (left to right)
    Each cell's center is returned with margin around it.

    Args:
        img: PIL Image
        cell_ref: Cell reference like "D5" (row D, column 5)
        grid_size: Number of grid divisions (default 10x10)
        margin: Pixels to extend beyond cell borders

    Returns:
        Tuple of (x1, y1, x2, y2) pixel coordinates

    Raises:
        ValueError: If cell_ref format is invalid
    """
    if len(cell_ref) != 2:
        raise ValueError(f"Invalid cell reference: '{cell_ref}'. Use format like 'D5'")

    row_char = cell_ref[0].upper()
    col_char = cell_ref[1]

    if row_char < 'A' or row_char > chr(ord('A') + grid_size - 1):
        raise ValueError(
            f"Invalid row '{row_char}'. Must be A-{chr(ord('A') + grid_size - 1)}"
        )

    try:
        col = int(col_char)
    except ValueError:
        raise ValueError(f"Invalid column '{col_char}'. Must be 0-{grid_size - 1}")

    if col < 0 or col >= grid_size:
        raise ValueError(f"Column {col} out of range 0-{grid_size - 1}")

    row = ord(row_char) - ord('A')
    width, height = img.size

    # Calculate cell boundaries
    cell_width = width / grid_size
    cell_height = height / grid_size

    cell_x1 = int(col * cell_width)
    cell_y1 = int(row * cell_height)
    cell_x2 = int((col + 1) * cell_width)
    cell_y2 = int((row + 1) * cell_height)

    # Apply margin
    x1 = max(0, cell_x1 - margin)
    y1 = max(0, cell_y1 - margin)
    x2 = min(width, cell_x2 + margin)
    y2 = min(height, cell_y2 + margin)

    return x1, y1, x2, y2


def get_quadrant_coordinates(
    img: Image.Image,
    quadrant: str
) -> Tuple[int, int, int, int]:
    """
    Get pixel coordinates for a quadrant of the image.

    Args:
        img: PIL Image
        quadrant: One of "top_left", "top_right", "bottom_left", "bottom_right", "center"

    Returns:
        Tuple of (x1, y1, x2, y2) pixel coordinates

    Raises:
        ValueError: If quadrant is invalid
    """
    width, height = img.size
    half_w, half_h = width // 2, height // 2

    quadrants = {
        "top_left": (0, 0, half_w, half_h),
        "top_right": (half_w, 0, width, half_h),
        "bottom_left": (0, half_h, half_w, height),
        "bottom_right": (half_w, half_h, width, height),
        "center": (
            width // 4,
            height // 4,
            3 * width // 4,
            3 * height // 4
        )
    }

    if quadrant.lower() not in quadrants:
        valid = list(quadrants.keys())
        raise ValueError(f"Invalid quadrant '{quadrant}'. Must be one of: {valid}")

    return quadrants[quadrant.lower()]


def assess_image_size(width: int, height: int) -> str:
    """
    Assess image resolution and return size category.

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Size category: "TINY", "SMALL", "MEDIUM", "LARGE", or "XLARGE"
    """
    area = width * height
    min_dim = min(width, height)

    if min_dim < 300:
        return "TINY"
    elif min_dim < 600:
        return "SMALL"
    elif min_dim < 1200:
        return "MEDIUM"
    elif min_dim < 2000:
        return "LARGE"
    else:
        return "XLARGE"


def calculate_resolution_gain(
    crop_area: Tuple[int, int, int, int],
    original_size: Tuple[int, int]
) -> float:
    """
    Calculate resolution gain from cropping.

    Resolution gain = original_area / crop_area
    Higher values mean more zoom (better for small text).

    Args:
        crop_area: (x1, y1, x2, y2) crop coordinates
        original_size: (width, height) original image size

    Returns:
        Magnification factor (1.0 = no change, 4.0 = 4x zoom)
    """
    x1, y1, x2, y2 = crop_area
    crop_width = x2 - x1
    crop_height = y2 - y1

    orig_width, orig_height = original_size

    orig_area = orig_width * orig_height
    crop_area_px = crop_width * crop_height

    if crop_area_px == 0:
        return 0.0

    return orig_area / crop_area_px


def create_grid_metadata(
    img: Image.Image,
    grid_size: int = 10
) -> Dict[str, Any]:
    """
    Create metadata dictionary for grid overlay.

    Returns information about grid cells and their pixel coordinates.

    Args:
        img: PIL Image
        grid_size: Number of grid divisions (default 10x10)

    Returns:
        Dictionary with grid information and cell coordinates
    """
    width, height = img.size
    cell_width = width / grid_size
    cell_height = height / grid_size

    cells = {}
    for row in range(grid_size):
        row_char = chr(ord('A') + row)
        for col in range(grid_size):
            cell_ref = f"{row_char}{col}"
            x1 = int(col * cell_width)
            y1 = int(row * cell_height)
            x2 = int((col + 1) * cell_width)
            y2 = int((row + 1) * cell_height)

            cells[cell_ref] = {
                "pixel_bounds": [x1, y1, x2, y2],
                "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                "width_px": x2 - x1,
                "height_px": y2 - y1
            }

    return {
        "grid_size": grid_size,
        "image_size": {"width": width, "height": height},
        "cell_dimensions": {
            "width": int(cell_width),
            "height": int(cell_height)
        },
        "cells": cells
    }
