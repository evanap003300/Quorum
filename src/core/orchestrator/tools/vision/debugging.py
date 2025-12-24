"""Debugging tools for image metadata and quality assessment."""

from typing import Union, Dict, Any
from PIL import Image
import numpy as np
from .utils import load_image, assess_image_size


def get_image_metadata(
    image_input: Union[str, Image.Image]
) -> Dict[str, Any]:
    """
    Inspect image metadata and assess quality.

    Returns resolution, aspect ratio, and size assessment to help agent
    decide if cropping/zooming will be effective.

    Args:
        image_input: File path or PIL Image

    Returns:
        Dictionary with:
        - width, height: Image dimensions in pixels
        - aspect_ratio: Width/height ratio
        - size_assessment: "TINY", "SMALL", "MEDIUM", "LARGE", or "XLARGE"
        - size_assessment_details: Human-readable guidance
        - cropping_recommendation: Whether cropping will help
        - pixel_count: Total pixel count
        - min_dimension: Smaller of width/height

    Example:
        >>> info = get_image_metadata("photo.jpg")
        >>> if info['size_assessment'] == 'TINY':
        ...     print("Cropping won't help - image too small")
    """
    img = load_image(image_input)
    width, height = img.size

    aspect_ratio = width / height if height > 0 else 0
    pixel_count = width * height
    min_dim = min(width, height)
    size_assessment = assess_image_size(width, height)

    # Provide guidance
    guidance_map = {
        "TINY": {
            "description": "Very low resolution (<300px min dimension)",
            "recommendation": "Cropping will NOT help - insufficient pixel data. Consider requesting higher-resolution scan.",
            "can_crop_effectively": False,
        },
        "SMALL": {
            "description": "Low resolution (300-600px min dimension)",
            "recommendation": "Cropping possible but limited benefit. Text may remain blurry even when zoomed.",
            "can_crop_effectively": True,
        },
        "MEDIUM": {
            "description": "Moderate resolution (600-1200px min dimension)",
            "recommendation": "Good for cropping. Zooming into quadrants or grid squares will help readability.",
            "can_crop_effectively": True,
        },
        "LARGE": {
            "description": "High resolution (1200-2000px min dimension)",
            "recommendation": "Excellent for detailed cropping. Multiple zoom levels possible.",
            "can_crop_effectively": True,
        },
        "XLARGE": {
            "description": "Very high resolution (>2000px min dimension)",
            "recommendation": "Optimal for cropping - can zoom deeply into details.",
            "can_crop_effectively": True,
        }
    }

    guidance = guidance_map.get(size_assessment, guidance_map["MEDIUM"])

    return {
        "width": width,
        "height": height,
        "size": {
            "width_px": width,
            "height_px": height
        },
        "aspect_ratio": round(aspect_ratio, 2),
        "aspect_ratio_description": _describe_aspect_ratio(aspect_ratio),
        "pixel_count": pixel_count,
        "min_dimension": min_dim,
        "size_assessment": size_assessment,
        "size_assessment_details": guidance["description"],
        "can_crop_effectively": guidance["can_crop_effectively"],
        "cropping_recommendation": guidance["recommendation"],
        "layout_type": _infer_layout_type(aspect_ratio)
    }


def analyze_image_contrast(
    image_input: Union[str, Image.Image]
) -> Dict[str, Any]:
    """
    Analyze image contrast and brightness distribution.

    Helps determine if contrast enhancement would be beneficial.

    Args:
        image_input: File path or PIL Image

    Returns:
        Dictionary with:
        - brightness: Mean pixel brightness (0-255)
        - contrast: Standard deviation of brightness
        - brightness_range: [min, max] brightness values
        - contrast_assessment: "LOW", "MODERATE", "HIGH"
        - recommendation: Suggested enhancements
    """
    img = load_image(image_input)
    img_array = np.array(img.convert('L'))  # Convert to grayscale

    brightness = np.mean(img_array)
    contrast = np.std(img_array)
    min_bright = np.min(img_array)
    max_bright = np.max(img_array)
    brightness_range = max_bright - min_bright

    # Assess contrast
    if contrast < 30:
        assessment = "LOW"
        recommendation = "Image has low contrast. Try enhance_clarity() or stretch_contrast()."
    elif contrast < 60:
        assessment = "MODERATE"
        recommendation = "Contrast is acceptable. Enhancement may still improve readability."
    else:
        assessment = "HIGH"
        recommendation = "Excellent contrast. Enhancement likely unnecessary."

    return {
        "brightness": {
            "mean": round(float(brightness), 1),
            "min": int(min_bright),
            "max": int(max_bright),
            "range": int(brightness_range)
        },
        "contrast": {
            "std_dev": round(float(contrast), 1),
            "assessment": assessment,
            "recommendation": recommendation
        },
        "histogram_info": {
            "darker_pixels": int(np.sum(img_array < 128)),
            "lighter_pixels": int(np.sum(img_array >= 128)),
            "distribution": _describe_brightness_distribution(img_array)
        }
    }


def detect_shadows_and_artifacts(
    image_input: Union[str, Image.Image]
) -> Dict[str, Any]:
    """
    Detect potential shadows and artifacts in the image.

    Helps determine if binarization or other preprocessing would help.

    Args:
        image_input: File path or PIL Image

    Returns:
        Dictionary with:
        - has_likely_shadows: Boolean
        - shadow_coverage_estimate: Percentage of image affected
        - artifact_assessment: Text describing detected issues
        - suggested_tools: List of tools that might help
    """
    img = load_image(image_input)
    img_array = np.array(img.convert('L'))

    # Detect shadow regions (dark areas that may be shadows)
    shadow_mask = img_array < 100
    shadow_coverage = np.sum(shadow_mask) / shadow_mask.size * 100

    has_shadows = shadow_coverage > 15  # More than 15% of image is very dark

    # Detect edge contrast (potential presence of text)
    from scipy import ndimage
    edges = ndimage.sobel(img_array)
    edge_density = np.sum(edges > 50) / edges.size * 100

    # Assess artifacts
    artifacts = []
    suggested_tools = []

    if shadow_coverage > 20:
        artifacts.append("Significant shadow regions detected")
        suggested_tools.append("binarize_image()")

    if np.std(img_array) < 30:
        artifacts.append("Low contrast - text may be faint")
        suggested_tools.append("enhance_clarity()")
        suggested_tools.append("stretch_contrast()")

    if edge_density < 5:
        artifacts.append("Few edges detected - image may be blurry")
        suggested_tools.append("apply_unsharp_mask()")
        suggested_tools.append("enhance_clarity()")

    return {
        "shadows": {
            "detected": has_shadows,
            "coverage_percentage": round(shadow_coverage, 1),
            "assessment": "Significant shadows present" if has_shadows else "Minimal shadows"
        },
        "edge_density": round(edge_density, 1),
        "artifacts": artifacts if artifacts else ["No major artifacts detected"],
        "suggested_tools": list(set(suggested_tools)) if suggested_tools else ["No enhancements needed"]
    }


def compare_images(
    image1_input: Union[str, Image.Image],
    image2_input: Union[str, Image.Image]
) -> Dict[str, Any]:
    """
    Compare two images to assess if preprocessing improved quality.

    Useful for A/B testing different enhancement strategies.

    Args:
        image1_input: Original image (file path or PIL Image)
        image2_input: Enhanced/processed image (file path or PIL Image)

    Returns:
        Dictionary with comparison metrics:
        - size_change: How dimensions changed
        - contrast_improvement: Contrast before/after
        - brightness_change: Brightness before/after
        - histogram_change: Overall distribution change
        - visual_similarity: Structural similarity metric
    """
    img1 = load_image(image1_input)
    img2 = load_image(image2_input)

    # Convert to grayscale for comparison
    arr1 = np.array(img1.convert('L'))
    arr2 = np.array(img2.convert('L'))

    # Size comparison
    size_change = {
        "original": img1.size,
        "processed": img2.size,
        "size_different": img1.size != img2.size
    }

    # Brightness comparison
    brightness_original = np.mean(arr1)
    brightness_processed = np.mean(arr2)

    # Contrast comparison
    contrast_original = np.std(arr1)
    contrast_processed = np.std(arr2)

    # Histogram difference (mean absolute difference)
    histogram_diff = np.mean(np.abs(arr1.astype(float) - arr2.astype(float)))

    # Structural similarity (simple version - edge correlation)
    edges1 = _sobel_edges(arr1)
    edges2 = _sobel_edges(arr2)
    edge_similarity = _correlation(edges1, edges2)

    return {
        "size": size_change,
        "brightness": {
            "original": round(float(brightness_original), 1),
            "processed": round(float(brightness_processed), 1),
            "change": round(float(brightness_processed - brightness_original), 1)
        },
        "contrast": {
            "original": round(float(contrast_original), 1),
            "processed": round(float(contrast_processed), 1),
            "improvement": round(float(contrast_processed - contrast_original), 1)
        },
        "overall_difference": round(float(histogram_diff), 1),
        "edge_similarity": round(edge_similarity, 2),
        "assessment": _assess_improvement(
            contrast_original, contrast_processed,
            brightness_original, brightness_processed,
            edge_similarity
        )
    }


# Helper functions

def _describe_aspect_ratio(ratio: float) -> str:
    """Describe aspect ratio in human terms."""
    if 0.8 < ratio < 1.2:
        return "Roughly square"
    elif ratio > 1.5:
        return "Landscape (wide)"
    elif ratio < 0.67:
        return "Portrait (tall)"
    else:
        return "Standard document"


def _infer_layout_type(aspect_ratio: float) -> str:
    """Infer page layout from aspect ratio."""
    if aspect_ratio > 1.3:
        return "Wide layout (likely single-column or side-by-side)"
    elif aspect_ratio < 0.8:
        return "Tall layout (likely multi-row content)"
    else:
        return "Standard layout"


def _describe_brightness_distribution(img_array: np.ndarray) -> str:
    """Describe the brightness distribution."""
    darker_half = np.sum(img_array < 128) / img_array.size
    if darker_half > 0.6:
        return "Darker overall"
    elif darker_half < 0.4:
        return "Lighter overall"
    else:
        return "Balanced dark/light"


def _sobel_edges(img_array: np.ndarray) -> np.ndarray:
    """Compute Sobel edges."""
    from scipy import ndimage
    sx = ndimage.sobel(img_array, axis=0)
    sy = ndimage.sobel(img_array, axis=1)
    return np.hypot(sx, sy)


def _correlation(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Compute correlation between two arrays."""
    # Flatten arrays
    flat1 = arr1.flatten().astype(float)
    flat2 = arr2.flatten().astype(float)

    # Normalize
    flat1 = (flat1 - np.mean(flat1)) / (np.std(flat1) + 1e-8)
    flat2 = (flat2 - np.mean(flat2)) / (np.std(flat2) + 1e-8)

    # Correlation
    correlation = np.mean(flat1 * flat2)
    return max(0.0, min(1.0, correlation))


def _assess_improvement(
    contrast_before: float,
    contrast_after: float,
    brightness_before: float,
    brightness_after: float,
    edge_similarity: float
) -> str:
    """Assess if processing improved the image."""
    improvements = []

    if contrast_after > contrast_before * 1.15:  # 15% improvement
        improvements.append("contrast improved")

    if edge_similarity > 0.7:
        improvements.append("structure preserved")

    if not improvements:
        return "No measurable improvement detected. Try a different enhancement."

    return f"Image improved: {', '.join(improvements)}"
