"""
Active Vision Module - Computer Vision Toolkit for Physics Problem Solving

This module provides assistive preprocessing tools to help LLMs better understand
physics problem images. The core philosophy is "Assistive Pre-processing" - fixing
artifacts (resolution, lighting, spatial confusion) that cause LLMs to fail, not
replacing their vision capabilities.

Tool Categories:
    1. SPATIAL SUITE - Navigation & Focus
       - crop_quadrant(): Fast pre-cropping into 25% sections
       - crop_region(): Precise cropping with pixel coordinates
       - apply_grid(): Overlay 10x10 grid with alphanumeric labels
       - crop_grid_square(): Zoom to specific grid cell (A0-J9)

    2. CLARITY SUITE - Enhancement & Restoration
       - binarize_image(): B&W conversion with adaptive thresholding
       - invert_colors(): Fix blackboard/dark-mode images
       - enhance_clarity(): Boost contrast and sharpen
       - stretch_contrast(): Full dynamic range enhancement
       - denoise_image(): Reduce noise while preserving edges
       - apply_unsharp_mask(): Targeted sharpening

    3. DEBUGGING SUITE - Metadata & Analysis
       - get_image_metadata(): Resolution, size assessment, guidance
       - analyze_image_contrast(): Brightness and contrast analysis
       - detect_shadows_and_artifacts(): Problem detection
       - compare_images(): A/B test preprocessing effectiveness

    4. CONTENT DETECTION - Automatic Region Finding
       - detect_content_regions(): Find text/diagram blocks
       - detect_text_regions(): Locate text-heavy areas
       - detect_diagram_regions(): Find visual content
       - highlight_content_regions(): Visualization overlay

Example Usage:
    >>> from tools.vision import (
    ...     apply_grid, crop_grid_square, binarize_image,
    ...     detect_content_regions, get_image_metadata
    ... )
    >>>
    >>> # Check if image is high-res enough for cropping
    >>> metadata = get_image_metadata("problem.jpg")
    >>> if not metadata['can_crop_effectively']:
    ...     print("Image too small:", metadata['cropping_recommendation'])
    >>>
    >>> # Detect interesting regions automatically
    >>> regions = detect_content_regions("problem.jpg")
    >>> for region in regions['regions'][:3]:
    ...     print(f"Found {region['type']}: {region['bounds']}")
    >>>
    >>> # Apply grid and zoom to specific cell
    >>> img, grid_meta = apply_grid("problem.jpg")
    >>> # Agent sees grid and identifies cell D5 has the equation
    >>> zoomed, crop_meta = crop_grid_square("problem.jpg", "D5")
    >>>
    >>> # Fix bad lighting
    >>> if regions['artifacts']:
    ...     enhanced = binarize_image("problem.jpg")

Chainability:
    All tools accept both file paths (str) and PIL Image objects, and return
    PIL Images. This enables chaining:

    >>> from tools.vision import crop_quadrant, binarize_image, enhance_clarity
    >>> img = crop_quadrant("problem.jpg", "top_right")
    >>> img = enhance_clarity(img, contrast=True, sharpen=True)
    >>> img = binarize_image(img)
    >>> img.save("processed.jpg")

Resolution Gain Guide:
    crop_quadrant():           ~2-4x magnification
    crop_region() [exact]:     Depends on selection (1-20x possible)
    crop_grid_square():        ~3-8x magnification (with margin)
    zoom deeply for text:      Use grid first to find interesting cells
"""

# Spatial Suite - Cropping and Grid Navigation
from .spatial import (
    crop_quadrant,
    crop_region,
    apply_grid,
    crop_grid_square,
)

# Clarity Suite - Image Enhancement and Restoration
from .clarity import (
    binarize_image,
    invert_colors,
    enhance_clarity,
    stretch_contrast,
    denoise_image,
    apply_unsharp_mask,
)

# Debugging Suite - Metadata and Analysis
from .debugging import (
    get_image_metadata,
    analyze_image_contrast,
    detect_shadows_and_artifacts,
    compare_images,
)

# Content Detection - Automatic Region Identification
from .content_detection import (
    detect_content_regions,
    detect_text_regions,
    detect_diagram_regions,
    highlight_content_regions,
)

# Utilities (for advanced use)
from . import utils

__all__ = [
    # Spatial Suite
    "crop_quadrant",
    "crop_region",
    "apply_grid",
    "crop_grid_square",
    # Clarity Suite
    "binarize_image",
    "invert_colors",
    "enhance_clarity",
    "stretch_contrast",
    "denoise_image",
    "apply_unsharp_mask",
    # Debugging Suite
    "get_image_metadata",
    "analyze_image_contrast",
    "detect_shadows_and_artifacts",
    "compare_images",
    # Content Detection
    "detect_content_regions",
    "detect_text_regions",
    "detect_diagram_regions",
    "highlight_content_regions",
    # Utilities
    "utils",
]

__version__ = "1.0.0"
__author__ = "Active Vision Research"
__description__ = "Assistive computer vision preprocessing for physics problem solving"
