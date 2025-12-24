"""Clarity tools for enhancement and restoration - filters for image quality."""

from typing import Union
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
from .utils import load_image


def binarize_image(
    image_input: Union[str, Image.Image],
    block_size: int = 11
) -> Image.Image:
    """
    Convert image to strict black and white using adaptive thresholding.

    Removes shadows, paper texture, and makes faint marks stand out.
    Uses adaptive thresholding which calculates threshold locally for each region,
    solving the "global threshold problem" where shadows fool basic algorithms.

    Args:
        image_input: File path or PIL Image
        block_size: Size of local region for threshold calculation (must be odd, 3-21)

    Returns:
        Black and white PIL Image (1-bit color)

    Example:
        >>> img = binarize_image("photo_with_shadow.jpg")
        >>> # Returns crisp B&W image with shadow removed
    """
    img = load_image(image_input)

    # Validate block_size
    if block_size < 3 or block_size > 21 or block_size % 2 == 0:
        block_size = 11  # Default to reasonable value

    # Convert PIL to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    # Apply adaptive thresholding
    binary_cv = cv2.adaptiveThreshold(
        img_cv,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=2
    )

    # Convert back to PIL
    binary_pil = Image.fromarray(binary_cv, mode='L')

    return binary_pil.convert('RGB')


def invert_colors(
    image_input: Union[str, Image.Image]
) -> Image.Image:
    """
    Invert all pixel colors mathematically (255 - pixel_value).

    Fixes blackboard photos (white chalk on dark background) and dark mode screenshots.
    Vision models trained primarily on white documents struggle with dark backgrounds.
    Inverting to "document mode" (black text on white) significantly improves accuracy.

    Args:
        image_input: File path or PIL Image

    Returns:
        Color-inverted PIL Image

    Example:
        >>> img = invert_colors("blackboard_photo.jpg")
        >>> # White chalk now appears as black text on white background
    """
    img = load_image(image_input)

    # PIL's ImageOps.invert works on all modes
    from PIL import ImageOps
    inverted = ImageOps.invert(img.convert('RGB'))

    return inverted


def enhance_clarity(
    image_input: Union[str, Image.Image],
    contrast: bool = True,
    contrast_factor: float = 1.5,
    sharpen: bool = True,
    sharpen_factor: float = 2.0
) -> Image.Image:
    """
    Enhance image clarity through contrast stretching and sharpening.

    Improves washed-out scans and blurry photos by:
    1. Stretching histogram so darkest pixels become pure black, lightest become white
    2. Sharpening edges with high-pass filter to make blurry text crisp

    Args:
        image_input: File path or PIL Image
        contrast: Enable contrast enhancement (default True)
        contrast_factor: Contrast multiplier (1.0 = no change, 1.5 = moderate, 2.0+ = strong)
        sharpen: Enable sharpening (default True)
        sharpen_factor: Sharpening strength (1.0 = no change, 2.0 = moderate, 3.0+ = strong)

    Returns:
        Enhanced PIL Image

    Example:
        >>> img = enhance_clarity("washed_out_scan.jpg", contrast=True, sharpen=True)
        >>> # Returns crisp, high-contrast version
    """
    img = load_image(image_input)

    # Step 1: Contrast stretching (histogram equalization)
    if contrast and contrast_factor != 1.0:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

        # Histogram equalization
        equalized = cv2.equalizeHist(img_cv)

        # Stretch to full range
        min_val = np.min(equalized)
        max_val = np.max(equalized)

        if max_val > min_val:
            stretched = np.uint8(
                255 * (equalized - min_val) / (max_val - min_val)
            )
        else:
            stretched = equalized

        # Apply contrast factor
        contrast_enhanced = np.uint8(
            np.clip(stretched * contrast_factor / 1.5, 0, 255)
        )

        # Convert back to color
        img = Image.fromarray(
            cv2.cvtColor(contrast_enhanced, cv2.COLOR_GRAY2RGB)
        )

    # Step 2: Sharpening
    if sharpen and sharpen_factor != 1.0:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(sharpen_factor)

    return img


def stretch_contrast(
    image_input: Union[str, Image.Image],
    percentile_low: float = 0.5,
    percentile_high: float = 99.5
) -> Image.Image:
    """
    Stretch image contrast to use full dynamic range.

    Remaps pixel values so the lowest percentile becomes black (0) and
    highest percentile becomes white (255).

    Args:
        image_input: File path or PIL Image
        percentile_low: Lower percentile to map to black (default 0.5%)
        percentile_high: Upper percentile to map to white (default 99.5%)

    Returns:
        Contrast-stretched PIL Image

    Example:
        >>> img = stretch_contrast("faded_scan.jpg")
        >>> # Returns version with full black-to-white range
    """
    img = load_image(image_input)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    # Calculate percentiles
    low = np.percentile(img_cv, percentile_low)
    high = np.percentile(img_cv, percentile_high)

    # Stretch
    if high > low:
        stretched = np.uint8(
            255 * (img_cv - low) / (high - low)
        )
    else:
        stretched = img_cv

    # Convert back to RGB
    result = cv2.cvtColor(stretched, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(result)


def denoise_image(
    image_input: Union[str, Image.Image],
    strength: int = 10
) -> Image.Image:
    """
    Reduce image noise while preserving edges.

    Uses Non-Local Means denoising for effective noise reduction
    without blurring text and diagrams.

    Args:
        image_input: File path or PIL Image
        strength: Denoising strength (1-20, higher = more aggressive)

    Returns:
        Denoised PIL Image

    Example:
        >>> img = denoise_image("noisy_photo.jpg", strength=10)
        >>> # Returns cleaner image with text preserved
    """
    img = load_image(image_input)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Apply Non-Local Means denoising
    denoised = cv2.fastNlMeansDenoising(
        img_cv,
        h=strength,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # Convert back to PIL RGB
    result_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)

    return Image.fromarray(result_rgb)


def apply_unsharp_mask(
    image_input: Union[str, Image.Image],
    radius: float = 1.0,
    percent: float = 150,
    threshold: int = 3
) -> Image.Image:
    """
    Apply unsharp mask for targeted sharpening.

    Creates high-pass filtered version and blends with original.
    More precise than simple sharpening filter.

    Args:
        image_input: File path or PIL Image
        radius: Blur radius for high-pass filter (0.5-3.0)
        percent: Sharpening strength as percentage (50-300)
        threshold: Minimum contrast to apply sharpening (0-10)

    Returns:
        Sharpened PIL Image

    Example:
        >>> img = apply_unsharp_mask("blurry_scan.jpg", radius=1.5, percent=200)
        >>> # Returns sharpened version
    """
    img = load_image(image_input)

    # Convert to grayscale for processing
    if img.mode != 'L':
        gray = img.convert('L')
    else:
        gray = img

    # Apply unsharp mask
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=radius))

    # Create sharpened image
    sharpened_data = []
    for p, q in zip(gray.getdata(), blurred.getdata()):
        diff = p - q
        if abs(diff) > threshold:
            sp = int(p + diff * percent / 100)
            sp = max(0, min(255, sp))
        else:
            sp = p
        sharpened_data.append(sp)

    sharpened = Image.new('L', gray.size)
    sharpened.putdata(sharpened_data)

    # Convert back to RGB if needed
    if img.mode == 'RGB':
        sharpened = sharpened.convert('RGB')

    return sharpened
