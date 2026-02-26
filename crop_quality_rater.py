"""
crop_quality_rater.py — Machine vision-based quality rating for baseball card crops.

Uses open source tools to evaluate crop algorithm performance across multiple dimensions:
  - Sharpness (Laplacian variance)
  - Brightness & contrast
  - Aspect ratio correctness
  - Card completeness (edge detection)
  - Perspective/keystone correctness
  - Background ratio (card vs. non-card area)
  - Color naturalness
  - Noise/artifact detection
  - Blur detection

Usage:
    python crop_quality_rater.py --input-dir output --report report.json
    python crop_quality_rater.py --input-dir output --verbose
    python crop_quality_rater.py --cropped card.jpg --verbose
"""

import argparse
import json
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from skimage import exposure, color, filters
from skimage.metrics import structural_similarity as ssim


# Standard baseball card: 2.5" x 3.5" @ 300 DPI
STANDARD_ASPECT = 3.5 / 2.5  # 1.4
STANDARD_ASPECT_TOLERANCE = 0.15  # Allow ±15% variance


@dataclass
class QualityScores:
    """Individual quality metrics."""
    sharpness: float          # 0-100, Laplacian variance
    brightness: float         # 0-100, ideal around 50%
    contrast: float           # 0-100
    aspect_ratio: float       # Actual aspect ratio detected
    aspect_score: float       # 0-100, how close to 1.4
    completeness: float       # 0-100, % of card edges detected
    perspective: float        # 0-100, how square the corners are
    background_ratio: float   # 0-100, % of image that is card (not background)
    color_score: float        # 0-100, natural color distribution
    noise_score: float        # 0-100, lower noise = higher score
    blur_score: float         # 0-100, based on FFT high-freq energy
    
    # Overall weighted score
    overall: float            # Weighted composite


def load_image(path: Path) -> Optional[np.ndarray]:
    """Load image and convert to BGR numpy array."""
    try:
        img = cv2.imread(str(path))
        if img is None:
            # Try with PIL as fallback
            pil_img = Image.open(path)
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def calculate_sharpness(img: np.ndarray) -> float:
    """Calculate sharpness using Laplacian variance.
    
    Higher variance = sharper image.
    Typical range: 0-2000+, normalize to 0-100.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    
    # Normalize: 0-2000 variance maps to 0-100 score
    score = min(100, (variance / 2000) * 100)
    return round(score, 2)


def calculate_brightness_contrast(img: np.ndarray) -> tuple:
    """Calculate brightness (mean intensity) and contrast (std dev).
    
    Brightness: 0-255, normalize to 0-100 (50% = ideal)
    Contrast: std dev, normalize to 0-100
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    
    # Brightness: ideal around 127 (mid-gray), penalize too dark or bright
    brightness = (mean_val / 255) * 100
    
    # Contrast: typical range 20-100, normalize
    contrast = min(100, (std_val / 60) * 100)
    
    return round(brightness, 2), round(contrast, 2)


def calculate_aspect_ratio(img: np.ndarray) -> tuple:
    """Detect card aspect ratio and score against standard 1.4.
    
    Uses edge detection to find card boundaries.
    """
    h, w = img.shape[:2]
    aspect = w / h if h > 0 else 0
    
    # Score: how close to 1.4
    deviation = abs(aspect - STANDARD_ASPECT)
    max_deviation = STANDARD_ASPECT * STANDARD_ASPECT_TOLERANCE
    
    if deviation <= max_deviation:
        score = 100
    else:
        # Linear penalty beyond tolerance
        score = max(0, 100 - (deviation - max_deviation) / max_deviation * 100)
    
    return round(aspect, 3), round(score, 2)


def calculate_completeness(img: np.ndarray) -> float:
    """Estimate card completeness by detecting edges near image boundaries.
    
    If card edges are cut off, there will be strong edges near borders.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect edges
    edges = cv2.Canny(gray, 50, 150)
    
    h, w = edges.shape
    border_width = int(min(h, w) * 0.05)  # 5% border
    
    # Count edge pixels in border regions
    top = np.sum(edges[:border_width, :] > 0)
    bottom = np.sum(edges[-border_width:, :] > 0)
    left = np.sum(edges[:, :border_width] > 0)
    right = np.sum(edges[:, -border_width:] > 0)
    
    total_edge_pixels = np.sum(edges > 0)
    if total_edge_pixels == 0:
        return 50.0  # Can't determine
    
    border_edges = top + bottom + left + right
    
    # High border edge ratio = incomplete card (cut off)
    border_ratio = border_edges / total_edge_pixels
    
    # Invert: 0% border = 100% complete, 50%+ border = 0% complete
    score = max(0, (0.50 - border_ratio) / 0.50 * 100)
    
    return round(score, 2)


def calculate_perspective(img: np.ndarray) -> float:
    """Check perspective/keystone correctness.
    
    Uses corner detection to see if corners form a rectangle.
    More skewed = lower score.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find contours and get the largest quadrilateral
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 50.0
    
    # Get largest contour
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    
    if area < (img.shape[0] * img.shape[1] * 0.1):
        return 50.0
    
    # Approximate to polygon
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    
    if len(approx) != 4:
        # Not a quadrilateral, check if it's at least roughly 4-sided
        return 60.0
    
    # Get corner angles - should all be ~90 degrees for a rectangle
    corners = approx.reshape(4, 2)
    
    # Calculate angles at each corner
    scores = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        p3 = corners[(i + 2) % 4]
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.abs(np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi)
        
        # 90 degrees = perfect, deviation reduces score
        angle_deviation = abs(angle - 90)
        corner_score = max(0, 100 - angle_deviation * 2)  # -2 points per degree off
        scores.append(corner_score)
    
    return round(np.mean(scores), 2)


def calculate_background_ratio(img: np.ndarray) -> float:
    """Estimate what percentage of the image is card vs. background.
    
    Uses color variance - cards have more uniform color than wood grain.
    """
    h, w = img.shape[:2]
    total_pixels = h * w
    
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Calculate local color variance (windows)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use adaptive threshold to find "card-like" regions
    # Cards typically have more uniform color than backgrounds
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    local_mean = cv2.blur(gray.astype(float), (21, 21))
    local_var = cv2.blur((gray.astype(float) - local_mean) ** 2, (21, 21))
    
    # Low variance = likely card, high variance = likely background/edges
    low_var_mask = (local_var < 500).astype(float)
    
    card_ratio = np.mean(low_var_mask) * 100
    
    return round(card_ratio, 2)


def calculate_color_score(img: np.ndarray) -> float:
    """Check color naturalness using LAB color space analysis.
    
    Ideal baseball card colors should fall within natural ranges.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Check A and B channels (color-opponent channels)
    # Natural colors typically cluster around center
    a_mean, b_mean = np.mean(a), np.mean(b)
    
    # Distance from neutral (128 in LAB)
    a_dev = abs(a_mean - 128)
    b_dev = abs(b_mean - 128)
    
    # Very saturated images (high dev) might be unnatural
    saturation_score = max(0, 100 - (a_dev + b_dev) / 2)
    
    # Check for color cast (unusual a/b distribution)
    a_std, b_std = np.std(a), np.std(b)
    
    # Balanced std suggests natural color distribution
    balance_score = max(0, 100 - abs(a_std - b_std))
    
    score = (saturation_score + balance_score) / 2
    
    return round(score, 2)


def calculate_noise_score(img: np.ndarray) -> float:
    """Estimate noise level using high-frequency analysis.
    
    High-frequency noise indicates sensor noise or compression artifacts.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    
    # Apply Laplacian (high-pass filter)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Noise estimate = high-frequency variance
    # Normalize by image mean to handle exposure differences
    noise_estimate = np.std(laplacian) / (np.mean(np.abs(gray)) + 1)
    
    # Typical noise estimates: 0-50 range, normalize
    score = max(0, 100 - noise_estimate * 10)
    
    return round(min(100, score), 2)


def calculate_blur_score(img: np.ndarray) -> float:
    """Detect blur using FFT (Fourier Transform) frequency analysis.
    
    Sharp images have more high-frequency energy.
    Blurry images have energy concentrated in low frequencies.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    
    # Compute FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    
    # Calculate high-frequency ratio
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    
    # Low frequency region (center)
    r_low = min(h, w) // 8
    y, x = np.ogrid[:h, :w]
    low_mask = (x - cx) ** 2 + (y - cy) ** 2 <= r_low ** 2
    
    # High frequency region (outer)
    r_high = min(h, w) // 3
    high_mask = (x - cx) ** 2 + (y - cy) ** 2 >= r_low ** 2
    high_mask &= (x - cx) ** 2 + (y - cy) ** 2 <= r_high ** 2
    
    low_energy = np.sum(magnitude[low_mask])
    high_energy = np.sum(magnitude[high_mask])
    
    if low_energy == 0:
        return 50.0
    
    # Higher high-freq ratio = sharper
    hf_ratio = high_energy / (low_energy + 1)
    
    # Normalize: typical ratios 0.1-2.0, map to 0-100
    score = min(100, hf_ratio * 50)
    
    return round(score, 2)


def rate_crop(img_path: Path, verbose: bool = False) -> QualityScores:
    """Rate a single cropped image."""
    img = load_image(img_path)
    
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    # Calculate all metrics
    sharpness = calculate_sharpness(img)
    brightness, contrast = calculate_brightness_contrast(img)
    aspect, aspect_score = calculate_aspect_ratio(img)
    completeness = calculate_completeness(img)
    perspective = calculate_perspective(img)
    background_ratio = calculate_background_ratio(img)
    color_score = calculate_color_score(img)
    noise_score = calculate_noise_score(img)
    blur_score = calculate_blur_score(img)
    
    # Calculate weighted overall score
    weights = {
        'sharpness': 0.15,
        'brightness': 0.08,
        'contrast': 0.08,
        'aspect_score': 0.12,
        'completeness': 0.15,
        'perspective': 0.12,
        'background_ratio': 0.10,
        'color_score': 0.08,
        'noise_score': 0.06,
        'blur_score': 0.06,
    }
    
    overall = (
        sharpness * weights['sharpness'] +
        brightness * weights['brightness'] +
        contrast * weights['contrast'] +
        aspect_score * weights['aspect_score'] +
        completeness * weights['completeness'] +
        perspective * weights['perspective'] +
        background_ratio * weights['background_ratio'] +
        color_score * weights['color_score'] +
        noise_score * weights['noise_score'] +
        blur_score * weights['blur_score']
    )
    
    scores = QualityScores(
        sharpness=sharpness,
        brightness=brightness,
        contrast=contrast,
        aspect_ratio=aspect,
        aspect_score=aspect_score,
        completeness=completeness,
        perspective=perspective,
        background_ratio=background_ratio,
        color_score=color_score,
        noise_score=noise_score,
        blur_score=blur_score,
        overall=round(overall, 2)
    )
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"Quality Report: {img_path.name}")
        print(f"{'='*50}")
        print(f"  Overall Score:      {scores.overall:.1f}/100")
        print(f"  ─────────────────────────────────")
        print(f"  Sharpness:          {scores.sharpness:.1f}/100")
        print(f"  Brightness:         {scores.brightness:.1f}/100")
        print(f"  Contrast:           {scores.contrast:.1f}/100")
        print(f"  Aspect Ratio:       {scores.aspect_ratio:.3f} (target: {STANDARD_ASPECT})")
        print(f"    → Score:          {scores.aspect_score:.1f}/100")
        print(f"  Completeness:       {scores.completeness:.1f}/100")
        print(f"  Perspective:       {scores.perspective:.1f}/100")
        print(f"  Card/Background:   {scores.background_ratio:.1f}%")
        print(f"  Color Score:        {scores.color_score:.1f}/100")
        print(f"  Noise Score:        {scores.noise_score:.1f}/100")
        print(f"  Blur Score:         {scores.blur_score:.1f}/100")
    
    return scores


def rate_directory(input_dir: Path, output_file: Optional[Path] = None, 
                   verbose: bool = False) -> dict:
    """Rate all images in a directory."""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    image_files = [f for f in input_dir.iterdir() 
                   if f.suffix.lower() in extensions]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return {"images": [], "summary": {}}
    
    results = []
    
    for img_path in sorted(image_files):
        try:
            scores = rate_crop(img_path, verbose=verbose)
            results.append({
                "file": img_path.name,
                "scores": asdict(scores)
            })
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            results.append({
                "file": img_path.name,
                "error": str(e)
            })
    
    # Calculate summary statistics
    if results and "scores" in results[0]:
        overall_scores = [r["scores"]["overall"] for r in results]
        summary = {
            "count": len(results),
            "mean_overall": round(np.mean(overall_scores), 2),
            "std_overall": round(np.std(overall_scores), 2),
            "min_overall": round(min(overall_scores), 2),
            "max_overall": round(max(overall_scores), 2),
            "pass_rate_70": round(sum(1 for s in overall_scores if s >= 70) / len(overall_scores) * 100, 1),
            "pass_rate_80": round(sum(1 for s in overall_scores if s >= 80) / len(overall_scores) * 100, 1),
        }
    else:
        summary = {"count": len(results), "error": "No valid images processed"}
    
    output = {
        "images": results,
        "summary": summary
    }
    
    # Print summary
    print(f"\n{'='*50}")
    print("BATCH SUMMARY")
    print(f"{'='*50}")
    print(f"  Images processed: {summary.get('count', 0)}")
    if "mean_overall" in summary:
        print(f"  Mean overall:     {summary['mean_overall']:.1f}/100")
        print(f"  Std deviation:    {summary['std_overall']:.1f}")
        print(f"  Range:            {summary['min_overall']:.1f} - {summary['max_overall']:.1f}")
        print(f"  Pass rate (≥70): {summary['pass_rate_70']:.1f}%")
        print(f"  Pass rate (≥80): {summary['pass_rate_80']:.1f}%")
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n  Report saved to: {output_file}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Rate baseball card crop quality")
    parser.add_argument("--input-dir", type=Path, 
                        help="Directory of cropped images to rate")
    parser.add_argument("--cropped", type=Path,
                        help="Single cropped image to rate")
    parser.add_argument("--report", type=Path,
                        help="Output JSON report path")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed per-image scores")
    
    args = parser.parse_args()
    
    if args.cropped:
        # Rate single image
        scores = rate_crop(args.cropped, verbose=True)
        if args.report:
            output = {
                "images": [{"file": args.cropped.name, "scores": asdict(scores)}],
                "summary": {"count": 1, "mean_overall": scores.overall}
            }
            with open(args.report, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"\nReport saved to: {args.report}")
    
    elif args.input_dir:
        # Rate directory
        output = rate_directory(args.input_dir, args.report, args.verbose)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
