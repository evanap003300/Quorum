"""Content detection tools for automatic region identification."""

from typing import Union, List, Dict, Any, Tuple
from PIL import Image, ImageDraw
import cv2
import numpy as np
from .utils import load_image, calculate_resolution_gain


def detect_content_regions(
    image_input: Union[str, Image.Image],
    min_area: int = 1000
) -> Dict[str, Any]:
    """
    Automatically identify "interesting" regions in the image.

    Detects text blocks, diagrams, and other content areas.
    Returns bounding boxes that help the agent focus on relevant parts.

    Uses edge detection and contour analysis to find regions with high content density.

    Args:
        image_input: File path or PIL Image
        min_area: Minimum region area in pixels to consider (default 1000)

    Returns:
        Dictionary with:
        - regions: List of detected regions with bounds and descriptions
        - total_content_area: Pixel count of all detected content
        - content_coverage: Percentage of image containing content
        - recommended_crops: Top crops to focus on high-content areas
        - grid_recommendations: Grid cells with high content density

    Example:
        >>> results = detect_content_regions("physics_problem.jpg")
        >>> for region in results['regions']:
        ...     print(f"Found {region['type']}: {region['bounds']}")
    """
    img = load_image(image_input)
    width, height = img.size

    # Convert to grayscale and find edges
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    # Use Canny edge detection
    edges = cv2.Canny(img_cv, 50, 150)

    # Dilate to connect nearby edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter and analyze contours
    regions = []
    total_content_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        x2, y2 = x + w, y + h

        # Estimate content type based on aspect ratio and area
        aspect_ratio = w / h if h > 0 else 0
        region_type = _classify_region(aspect_ratio, area, width, height)

        # Calculate content density in this region
        region_edges = edges[y:y2, x:x2]
        content_density = np.sum(region_edges > 0) / (w * h) if (w * h) > 0 else 0

        region_info = {
            "bounds": {"x1": x, "y1": y, "x2": x2, "y2": y2},
            "dimensions": {"width": w, "height": h},
            "area": int(area),
            "type": region_type,
            "aspect_ratio": round(aspect_ratio, 2),
            "content_density": round(float(content_density), 2),
            "center": {"x": (x + x2) // 2, "y": (y + y2) // 2}
        }

        regions.append(region_info)
        total_content_area += area

    # Sort by area (largest first)
    regions.sort(key=lambda r: r["area"], reverse=True)

    # Calculate coverage
    content_coverage = (total_content_area / (width * height)) * 100

    return {
        "total_regions": len(regions),
        "regions": regions,
        "content_area": {
            "total_pixels": total_content_area,
            "coverage_percentage": round(content_coverage, 1)
        },
        "recommended_crops": _recommend_crops(regions, width, height),
        "grid_recommendations": _identify_grid_hotspots(regions, width, height, grid_size=10),
        "analysis": {
            "average_region_size": int(total_content_area / len(regions)) if regions else 0,
            "largest_region": regions[0] if regions else None,
            "content_distribution": _analyze_content_distribution(regions, width, height)
        }
    }


def detect_text_regions(
    image_input: Union[str, Image.Image]
) -> Dict[str, Any]:
    """
    Detect regions likely to contain text.

    More specialized than detect_content_regions - focuses on text blocks.
    Useful for finding equations, problem statements, and labels.

    Args:
        image_input: File path or PIL Image

    Returns:
        Dictionary with:
        - text_regions: List of detected text areas
        - text_density_heatmap: Which areas are text-heavy
        - equation_areas: Regions likely containing mathematical content
    """
    img = load_image(image_input)
    width, height = img.size

    # Convert to grayscale
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    # Use morphological operations to find text
    # Text has fine details - use smaller kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(img_cv, cv2.MORPH_GRADIENT, kernel)

    # Threshold
    _, binary = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    text_regions = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # Text regions typically have specific size ranges
        if area < 100 or area > 50000:  # Skip very small or very large
            continue

        x, y, w, h = cv2.boundingRect(contour)
        x2, y2 = x + w, y + h

        # Check if this looks like text (height-width ratio typical of text lines)
        aspect = h / w if w > 0 else 0

        # Text typically taller than wide (typical line aspect ratio 0.1 - 2.0)
        if 0.1 < aspect < 2.0:
            text_regions.append({
                "bounds": {"x1": x, "y1": y, "x2": x2, "y2": y2},
                "dimensions": {"width": w, "height": h},
                "aspect_ratio": round(aspect, 2),
                "area": int(area)
            })

    return {
        "text_regions_found": len(text_regions),
        "text_regions": text_regions,
        "text_coverage_estimate": round(
            (sum(r["area"] for r in text_regions) / (width * height)) * 100, 1
        )
    }


def detect_diagram_regions(
    image_input: Union[str, Image.Image]
) -> Dict[str, Any]:
    """
    Detect regions likely to contain diagrams or visual content.

    Distinguishes between text and visual content (graphs, sketches, charts).
    Useful for identifying where physics diagrams are located.

    Args:
        image_input: File path or PIL Image

    Returns:
        Dictionary with:
        - diagram_regions: List of detected visual/diagram areas
        - diagram_density: Which parts of image are diagram-heavy
        - largest_visual_element: Position of biggest non-text content
    """
    img = load_image(image_input)
    width, height = img.size

    # Use different edge detection for diagrams
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    # Laplacian for diagrams (picks up curves and shapes)
    laplacian = cv2.Laplacian(img_cv, cv2.CV_64F)
    laplacian_uint8 = cv2.convertScaleAbs(laplacian)

    # Threshold
    _, binary = cv2.threshold(laplacian_uint8, 50, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    diagram_regions = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # Diagrams are typically medium to large areas
        if area < 500 or area > 500000:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        x2, y2 = x + w, y + h

        diagram_regions.append({
            "bounds": {"x1": x, "y1": y, "x2": x2, "y2": y2},
            "dimensions": {"width": w, "height": h},
            "area": int(area),
            "center": {"x": (x + x2) // 2, "y": (y + y2) // 2}
        })

    # Sort by area
    diagram_regions.sort(key=lambda r: r["area"], reverse=True)

    return {
        "diagram_regions_found": len(diagram_regions),
        "diagram_regions": diagram_regions[:5],  # Top 5
        "largest_diagram": diagram_regions[0] if diagram_regions else None,
        "diagram_density_percentage": round(
            (sum(r["area"] for r in diagram_regions) / (width * height)) * 100, 1
        )
    }


def highlight_content_regions(
    image_input: Union[str, Image.Image],
    color: str = "red",
    alpha: float = 0.3
) -> Image.Image:
    """
    Overlay detected content regions onto image for visualization.

    Helps agent see where the system detected content.

    Args:
        image_input: File path or PIL Image
        color: Box color ("red", "blue", "green", etc.)
        alpha: Overlay transparency (0.0-1.0)

    Returns:
        Image with bounding boxes drawn around detected regions
    """
    img = load_image(image_input)
    results = detect_content_regions(img)

    # Create overlay
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay, 'RGBA')

    # Parse color
    color_rgb = _parse_color_to_rgb(color)
    box_color = (*color_rgb, int(255 * alpha))

    # Draw boxes for each region
    for region in results["regions"]:
        bounds = region["bounds"]
        x1, y1, x2, y2 = bounds["x1"], bounds["y1"], bounds["x2"], bounds["y2"]

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)

        # Add label
        label = f"{region['type']}"
        draw.text((x1 + 5, y1 + 5), label, fill=box_color)

    return overlay


# Helper functions

def _classify_region(aspect_ratio: float, area: int, img_w: int, img_h: int) -> str:
    """Classify a region by its shape and size."""
    img_area = img_w * img_h
    relative_size = area / img_area

    if 0.8 < aspect_ratio < 1.2:
        if relative_size > 0.1:
            return "large_block"
        else:
            return "compact_element"
    elif aspect_ratio > 1.5:
        return "horizontal_bar"
    elif aspect_ratio < 0.67:
        return "vertical_bar"
    else:
        return "irregular_shape"


def _recommend_crops(
    regions: List[Dict[str, Any]],
    img_width: int,
    img_height: int,
    num_recommendations: int = 3
) -> List[Dict[str, Any]]:
    """Generate crop recommendations based on detected regions."""
    if not regions:
        return []

    recommendations = []

    for i, region in enumerate(regions[:num_recommendations]):
        bounds = region["bounds"]
        x1, y1, x2, y2 = bounds["x1"], bounds["y1"], bounds["x2"], bounds["y2"]

        # Add margin
        margin = 20
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(img_width, x2 + margin)
        y2 = min(img_height, y2 + margin)

        resolution_gain = calculate_resolution_gain((x1, y1, x2, y2), (img_width, img_height))

        recommendations.append({
            "index": i + 1,
            "region_type": region["type"],
            "crop_bounds": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "resolution_gain": round(resolution_gain, 1),
            "content_density": region["content_density"]
        })

    return recommendations


def _identify_grid_hotspots(
    regions: List[Dict[str, Any]],
    img_width: int,
    img_height: int,
    grid_size: int = 10
) -> List[Dict[str, Any]]:
    """Identify grid cells with high content concentration."""
    cell_width = img_width / grid_size
    cell_height = img_height / grid_size

    # Count content in each grid cell
    grid_content = {}

    for region in regions:
        bounds = region["bounds"]
        x1, y1, x2, y2 = bounds["x1"], bounds["y1"], bounds["x2"], bounds["y2"]

        # Find which grid cells this region overlaps
        start_col = int(x1 / cell_width)
        end_col = int(x2 / cell_width)
        start_row = int(y1 / cell_height)
        end_row = int(y2 / cell_height)

        for row in range(start_row, min(end_row + 1, grid_size)):
            for col in range(start_col, min(end_col + 1, grid_size)):
                cell_ref = f"{chr(ord('A') + row)}{col}"
                if cell_ref not in grid_content:
                    grid_content[cell_ref] = 0
                grid_content[cell_ref] += region["area"]

    # Sort by content amount
    hotspots = [
        {
            "cell": cell_ref,
            "content_area": content_area,
            "recommendation": f"High content in {cell_ref} - consider crop_grid_square('{cell_ref}')"
        }
        for cell_ref, content_area in sorted(
            grid_content.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
    ]

    return hotspots


def _analyze_content_distribution(
    regions: List[Dict[str, Any]],
    img_width: int,
    img_height: int
) -> str:
    """Describe how content is distributed in the image."""
    if not regions:
        return "Empty or very sparse content"

    # Analyze centers
    centers_x = [r["center"]["x"] for r in regions]
    centers_y = [r["center"]["y"] for r in regions]

    center_x_avg = np.mean(centers_x)
    center_y_avg = np.mean(centers_y)

    # Determine distribution
    if center_x_avg < img_width * 0.3:
        x_dist = "left-aligned"
    elif center_x_avg > img_width * 0.7:
        x_dist = "right-aligned"
    else:
        x_dist = "centered horizontally"

    if center_y_avg < img_height * 0.3:
        y_dist = "top-aligned"
    elif center_y_avg > img_height * 0.7:
        y_dist = "bottom-aligned"
    else:
        y_dist = "distributed vertically"

    return f"Content is {x_dist} and {y_dist}"


def _parse_color_to_rgb(color: str) -> Tuple[int, int, int]:
    """Parse color name to RGB."""
    color_map = {
        "red": (255, 0, 0),
        "blue": (0, 0, 255),
        "green": (0, 255, 0),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
    }
    return color_map.get(color.lower(), (255, 0, 0))
