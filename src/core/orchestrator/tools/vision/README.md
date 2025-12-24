# Active Vision Module - Computer Vision Toolkit

A specialized computer vision preprocessing toolkit for improving LLM performance on physics problem images. Based on "Assistive Pre-processing" - fixing artifacts (resolution, lighting, spatial confusion) that cause vision models to fail on technical content.

## Core Philosophy

**Don't replace LLM vision—enhance what it sees.**

Vision models (GPT-4o, Claude's vision) are already good at understanding context and semantics. This toolkit fixes specific technical failures:
- **Resolution Cliff**: Diagrams too small to read clearly
- **Lighting Artifacts**: Shadows, uneven lighting, reflections
- **Spatial Hallucination**: LLMs guess at coordinates—can't reliably point to pixel locations
- **Dark Mode Confusion**: Blackboard photos, dark screenshots

## Installation

Dependencies are in `requirements.txt`:
```bash
pip install Pillow>=10.0.0 opencv-python>=4.8.0 scipy>=1.11.0
```

## Quick Start

```python
from tools.vision import apply_grid, crop_grid_square, binarize_image

# 1. Check if image is viable for preprocessing
from tools.vision import get_image_metadata
info = get_image_metadata("problem.jpg")
if not info['can_crop_effectively']:
    print(info['cropping_recommendation'])  # "Image too small - recommend 300px+"

# 2. Apply grid overlay to identify regions
img, grid_meta = apply_grid("problem.jpg")
img.save("with_grid.jpg")
# Agent sees grid labels A0-J9 and can identify which cell has the equation

# 3. Zoom to specific cell
zoomed, crop_info = crop_grid_square("problem.jpg", "D5")
# crop_info['resolution_gain'] shows ~5.2x magnification

# 4. Fix image quality issues
enhanced = binarize_image("problem.jpg")  # Removes shadows, improves contrast
```

## Tool Categories

### 1. Spatial Suite - Navigation & Focus
**Problem**: "I see a diagram in the corner but the labels are too small"

| Tool | Purpose | Input | Output |
|------|---------|-------|--------|
| `crop_quadrant()` | Fast 25% cropping | Image + quadrant name | Cropped image |
| `crop_region()` | Precise pixel cropping | Image + (x1,y1,x2,y2) | Cropped image |
| `apply_grid()` | Overlay reference grid | Image | Image + grid metadata |
| `crop_grid_square()` | Zoom to grid cell | Image + "D5" | Zoomed image + metadata |

**Grid System**: 10×10 grid with alphanumeric labels (Rows: A-J top-to-bottom, Columns: 0-9 left-to-right)

```python
from tools.vision import apply_grid, crop_grid_square

# Step 1: Show grid
img, meta = apply_grid("problem.jpg")
img.save("with_grid.jpg")

# Step 2: Agent sees "equation is in cell D5" (row D, column 5)
# Step 3: Zoom to that cell
zoomed, meta = crop_grid_square("problem.jpg", "D5")
# meta['resolution_gain'] = 5.2  (5.2x magnification)
```

### 2. Clarity Suite - Enhancement & Restoration
**Problem**: "This scanned page has a shadow over the equation" or "The text is faint"

| Tool | Purpose | Key Parameters |
|------|---------|-----------------|
| `binarize_image()` | B&W conversion with adaptive thresholding | block_size=11 |
| `invert_colors()` | Fix blackboard/dark-mode images | (none) |
| `enhance_clarity()` | Boost contrast + sharpen | contrast_factor, sharpen_factor |
| `stretch_contrast()` | Full dynamic range | percentile_low, percentile_high |
| `denoise_image()` | Reduce noise (NLM denoising) | strength=10 |
| `apply_unsharp_mask()` | Targeted sharpening | radius, percent, threshold |

```python
from tools.vision import binarize_image, enhance_clarity, invert_colors

# Remove shadows from scanned document
clean = binarize_image("shadow_scan.jpg")

# Fix washed-out scan
enhanced = enhance_clarity("faded_scan.jpg", contrast=1.5, sharpen=2.0)

# Fix blackboard photo
whiteboard_style = invert_colors("chalk_on_black.jpg")
```

### 3. Debugging Suite - Metadata & Analysis
**Problem**: "Is this image high-res enough to process?" or "Did my preprocessing help?"

| Tool | Purpose | Returns |
|------|---------|---------|
| `get_image_metadata()` | Resolution, size assessment, guidance | Dict with size/aspect/recommendation |
| `analyze_image_contrast()` | Brightness and contrast metrics | Dict with brightness/contrast/distribution |
| `detect_shadows_and_artifacts()` | Identify problem areas | Dict with artifacts + suggested tools |
| `compare_images()` | A/B test preprocessing | Dict with improvement metrics |

```python
from tools.vision import get_image_metadata, detect_shadows_and_artifacts, compare_images

# Check if cropping will work
info = get_image_metadata("problem.jpg")
print(f"Size: {info['size_assessment']}")  # "MEDIUM" or "TINY"?
print(f"Can crop: {info['can_crop_effectively']}")
print(info['cropping_recommendation'])  # Guidance

# Detect issues
issues = detect_shadows_and_artifacts("problem.jpg")
print(issues['artifacts'])  # ["Significant shadow regions detected"]
print(issues['suggested_tools'])  # ["binarize_image()", "enhance_clarity()"]

# A/B test: is binarization helping?
original = load_image("problem.jpg")
processed = binarize_image("problem.jpg")
comparison = compare_images(original, processed)
print(comparison['assessment'])  # "Image improved: contrast improved, structure preserved"
```

### 4. Content Detection - Automatic Region Finding
**Problem**: "Where should I look? This image is cluttered"

| Tool | Purpose | Returns |
|------|---------|---------|
| `detect_content_regions()` | Find text/diagram blocks | List of regions + grid hotspots |
| `detect_text_regions()` | Locate text-heavy areas | List of text blocks |
| `detect_diagram_regions()` | Find visual content | List of diagram areas |
| `highlight_content_regions()` | Visualization | Image with boxes drawn |

```python
from tools.vision import detect_content_regions, highlight_content_regions

# Automatically find interesting regions
regions = detect_content_regions("problem.jpg")
print(f"Found {regions['total_regions']} regions")
for region in regions['regions'][:3]:
    print(f"{region['type']}: {region['bounds']}")
    print(f"  Density: {region['content_density']}")

# Recommended crops based on content
for crop in regions['recommended_crops']:
    print(f"Crop {crop['index']}: {crop['crop_bounds']} ({crop['resolution_gain']}x zoom)")

# Recommended grid squares
for spot in regions['grid_recommendations']:
    print(f"Hotspot: {spot['cell']} - try crop_grid_square(img, '{spot['cell']}')")

# Visualize where system detected content
viz = highlight_content_regions("problem.jpg")
viz.save("detected_regions.jpg")
```

## Resolution Gain Guide

How much will cropping help?

| Tool | Typical Magnification | When to Use |
|------|----------------------|-------------|
| `crop_quadrant()` | 2-4x | Quick pre-crop without analysis |
| `crop_region()` | 1-20x | After manual coordinate selection |
| `crop_grid_square()` | 3-8x | After grid analysis (recommended) |

**Rule of thumb**:
- **TINY** (<300px): Cropping won't help—insufficient pixels
- **SMALL** (300-600px): Limited benefit, text still may blur
- **MEDIUM** (600-1200px): Cropping recommended, good results
- **LARGE** (1200px+): Excellent for detailed crops

## Design Principles

### 1. Chainability
All tools accept both file paths and PIL Images, return PIL Images:

```python
from tools.vision import crop_quadrant, enhance_clarity, binarize_image

img = crop_quadrant("problem.jpg", "center")
img = enhance_clarity(img, contrast=1.5)
img = binarize_image(img)
img.save("result.jpg")
```

### 2. Sensible Defaults
Every tool works out-of-the-box without parameter tuning:

```python
# All of these just work:
crop_quadrant(img, "top_right")
enhance_clarity(img)  # Uses contrast=1.5, sharpen=2.0
apply_grid(img)  # Uses 10x10 grid, red lines, 50% alpha
```

### 3. Metadata-First Design
Tools return metadata alongside images to help agents make decisions:

```python
zoomed, crop_info = crop_grid_square(img, "D5")
resolution_gain = crop_info['resolution_gain']  # 5.2
gain_description = crop_info['gain_description']  # "5.2x magnification"
```

### 4. Assistive, Not Replacement
Tools fix specific technical failures. They don't:
- Replace OCR (use GPT-4o for text understanding)
- Do semantic understanding (use LLM for physics)
- Generate new information (only enhance existing)

## Grid Coordinate System

The 10×10 grid solves spatial hallucination:

```
     0 1 2 3 4 5 6 7 8 9
   +---+---+---+---+---+---+---+---+---+---+
 A | . | . | . | . | . | . | . | . | . | . |
   +---+---+---+---+---+---+---+---+---+---+
 B | . | . | . | . | . | . | . | . | . | . |
   +---+---+---+---+---+---+---+---+---+---+
 C | . | . | . | . | . | . | . | . | . | . |
   +---+---+---+---+---+---+---+---+---+---+
 D | . | . | . | . | . | . | . | . | . | . |  <- "The equation is here"
   +---+---+---+---+---+---+---+---+---+---+
 ...
 J | . | . | . | . | . | . | . | . | . | . |
   +---+---+---+---+---+---+---+---+---+---+
```

**Reference format**: "D5" = Row D, Column 5

**Why this works**:
- LLMs can identify "which grid square" (discrete choice)
- LLMs can't estimate pixel coordinates reliably
- Grid overlay gives visual context
- `crop_grid_square("D5")` zooms precisely

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Cropping didn't help - text still blurry" | Image is SMALL or TINY—try `detect_content_regions()` instead to find high-density areas |
| "Shadow still visible after binarize" | Adjust `block_size`: smaller for thin shadows, larger for broad shadows |
| "Image too dark for processing" | Try `invert_colors()` first, then other tools |
| "I don't know where to crop" | Use `detect_content_regions()` + `highlight_content_regions()` to visualize |
| "Processing made it worse" | Use `compare_images()` to verify, then try different `contrast_factor` or `sharpen_factor` |

## API Summary

**Import everything**:
```python
from tools.vision import *
```

**Or import specific categories**:
```python
from tools.vision import (
    # Spatial
    crop_quadrant, crop_region, apply_grid, crop_grid_square,
    # Clarity
    binarize_image, invert_colors, enhance_clarity,
    # Debugging
    get_image_metadata, detect_shadows_and_artifacts,
    # Content Detection
    detect_content_regions, highlight_content_regions,
)
```

**Or be specific**:
```python
from tools.vision.spatial import apply_grid
from tools.vision.clarity import binarize_image
from tools.vision.debugging import get_image_metadata
```

## Performance Notes

- **`apply_grid()`**: Fast (~50ms for typical image)
- **`binarize_image()`**: Medium (~200-500ms, depends on size)
- **`detect_content_regions()`**: Slow (~1-3s), worth it for automatic guidance
- **`crop_*` functions**: Fast (<50ms)
- **`enhance_clarity()`**: Medium (~200-800ms with both contrast and sharpen)

For real-time use, stick to fast tools: cropping, grid, contrast analysis.

## Future Enhancements

Potential additions (not implemented yet):
- `rotate_image()` - Correct skewed page scans
- `extract_text_bbox()` - Find individual text bounding boxes
- `deskew_image()` - Correct rotated document
- `remove_watermark()` - Remove document watermarks
- `perspective_correction()` - Fix photos taken at angles
