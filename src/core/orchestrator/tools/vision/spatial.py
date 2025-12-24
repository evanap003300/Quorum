"""Spatial tools for navigation and focus - cropping and grid overlays."""

from typing import Union, Tuple, Dict, Any
from PIL import Image, ImageDraw
from .utils import (
    load_image,
    validate_coordinates,
    get_quadrant_coordinates,
    grid_to_coordinates,
    create_grid_metadata,
    calculate_resolution_gain
)


def crop_quadrant(
    image_input: Union[str, Image.Image],
    quadrant: str
) -> Image.Image:
    """
    Crop a quadrant of the image - fast pre-cropping without coordinates.

    Args:
        image_input: File path or PIL Image
        quadrant: One of "top_left", "top_right", "bottom_left", "bottom_right", "center"

    Returns:
        Cropped PIL Image

    Example:
        >>> img = crop_quadrant("photo.jpg", "top_right")
        >>> # Returns top-right 25% of the image
    """
    img = load_image(image_input)
    x1, y1, x2, y2 = get_quadrant_coordinates(img, quadrant)
    return img.crop((x1, y1, x2, y2))


def crop_region(
    image_input: Union[str, Image.Image],
    x1: int,
    y1: int,
    x2: int,
    y2: int
) -> Image.Image:
    """
    Precise crop using pixel coordinates.

    Use after grid analysis to zoom into specific regions identified by coordinates.

    Args:
        image_input: File path or PIL Image
        x1, y1: Top-left corner in pixels
        x2, y2: Bottom-right corner in pixels

    Returns:
        Cropped PIL Image

    Example:
        >>> img = crop_region("photo.jpg", 100, 50, 500, 300)
        >>> # Returns region from (100,50) to (500,300)
    """
    img = load_image(image_input)
    x1, y1, x2, y2 = validate_coordinates(img, x1, y1, x2, y2)
    return img.crop((x1, y1, x2, y2))


def apply_grid(
    image_input: Union[str, Image.Image],
    grid_size: int = 10,
    line_width: int = 2,
    line_color: str = "red",
    alpha: float = 0.5
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Overlay a coordinate grid on the image to enable spatial reference.

    Creates a 10x10 grid with alphanumeric labels (A-J rows, 0-9 columns).
    Cells are referenced as "D5" meaning row D, column 5.

    Args:
        image_input: File path or PIL Image
        grid_size: Number of grid divisions per side (default 10x10)
        line_width: Width of grid lines in pixels
        line_color: Color of grid lines ("red", "blue", "yellow", etc.)
        alpha: Transparency of overlay (0.0 = invisible, 1.0 = opaque)

    Returns:
        Tuple of:
        - Annotated PIL Image with grid overlay
        - Metadata dict with grid cell information and pixel coordinates

    Example:
        >>> img, metadata = apply_grid("photo.jpg")
        >>> # Image has red grid overlay with labels
        >>> # metadata["cells"]["D5"] contains pixel bounds of cell D5
    """
    img = load_image(image_input)
    width, height = img.size

    # Create overlay layer
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    cell_width = width / grid_size
    cell_height = height / grid_size

    # Convert RGB to RGBA for blending
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    # Parse color
    color_rgb = _parse_color(line_color)
    line_color_rgba = (*color_rgb, int(255 * alpha))

    # Draw vertical lines
    for col in range(grid_size + 1):
        x = int(col * cell_width)
        draw.line([(x, 0), (x, height)], fill=line_color_rgba, width=line_width)

    # Draw horizontal lines
    for row in range(grid_size + 1):
        y = int(row * cell_height)
        draw.line([(0, y), (width, y)], fill=line_color_rgba, width=line_width)

    # Draw labels
    font_size = max(8, int(cell_width / 3))
    label_positions = []

    for row in range(grid_size):
        row_char = chr(ord('A') + row)
        for col in range(grid_size):
            # Position label at top-left of cell
            x = int(col * cell_width) + 3
            y = int(row * cell_height) + 3

            label = f"{row_char}{col}"
            draw.text(
                (x, y),
                label,
                fill=(*color_rgb, int(255)),
                font=None  # Use default font
            )

            label_positions.append({
                "label": label,
                "pixel_position": (x, y)
            })

    # Composite overlay onto original image
    result = Image.alpha_composite(img, overlay)

    # Get grid metadata
    metadata = create_grid_metadata(img, grid_size)

    return result.convert('RGB'), metadata


def crop_grid_square(
    image_input: Union[str, Image.Image],
    cell_ref: str,
    margin: int = 200
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Crop to a specific grid square with context margin.

    After using apply_grid(), agent can request this tool to zoom into
    a specific cell. Includes margin pixels around the cell for context.

    Args:
        image_input: File path or PIL Image
        cell_ref: Cell reference like "D5" (row D, column 5)
        margin: Pixels to extend beyond cell borders for context (default 200)

    Returns:
        Tuple of:
        - Cropped PIL Image (zoomed to cell)
        - Metadata dict with crop bounds and resolution gain info

    Example:
        >>> img, meta = crop_grid_square("photo.jpg", "D5", margin=150)
        >>> # Returns zoomed view of cell D5 with 150px context around it
        >>> # meta["resolution_gain"] shows magnification factor
    """
    img = load_image(image_input)
    original_size = img.size

    x1, y1, x2, y2 = grid_to_coordinates(img, cell_ref, grid_size=10, margin=margin)

    cropped = img.crop((x1, y1, x2, y2))
    resolution_gain = calculate_resolution_gain((x1, y1, x2, y2), original_size)

    metadata = {
        "cell_reference": cell_ref,
        "crop_bounds": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "crop_dimensions": {
            "width": x2 - x1,
            "height": y2 - y1
        },
        "original_size": {"width": original_size[0], "height": original_size[1]},
        "resolution_gain": round(resolution_gain, 2),
        "gain_description": f"{resolution_gain:.1f}x magnification"
    }

    return cropped, metadata


def _parse_color(color: str) -> Tuple[int, int, int]:
    """
    Parse color name to RGB tuple.

    Args:
        color: Color name like "red", "blue", "yellow", etc.

    Returns:
        Tuple of (R, G, B) values 0-255
    """
    color_map = {
        "red": (255, 0, 0),
        "blue": (0, 0, 255),
        "green": (0, 255, 0),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "orange": (255, 165, 0),
        "purple": (128, 0, 128),
        "pink": (255, 192, 203),
        "gray": (128, 128, 128),
    }

    color_lower = color.lower().strip()
    if color_lower not in color_map:
        # Default to red if unknown
        return color_map["red"]

    return color_map[color_lower]
