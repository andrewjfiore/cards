"""
auto_tuner.py — Automated tuning of card_crop.py parameters.

Runs card_crop with different configurations and evaluates based on:
1. No padding/background visible (card fills the crop)
2. Dimensions match baseball card (aspect ratio ~1.4)
3. Player's body is upright (portrait or landscape OK, but not rotated 90° off)
4. Text is upright (can be portrait or landscape)

Target: 75% correct threshold.

Usage:
    python auto_tuner.py --input-dir /path/to/photos --max-iterations 20
"""

import argparse
import json
import subprocess
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import itertools
import random

import cv2
import numpy as np
from PIL import Image


# Baseball card dimensions: 2.5" x 3.5" = 1.4 aspect ratio
TARGET_ASPECT = 1.4
ASPECT_TOLERANCE = 0.15  # ±15%


@dataclass
class CropResult:
    """Result of processing a single image."""
    source: str
    output_path: Path
    success: bool
    aspect_ratio: float
    width: int
    height: int
    has_background: bool  # True if non-card content detected
    orientation_correct: bool  # Player upright, text upright
    is_rectangle: bool  # Card is rectangular (not skewed)
    face_in_top_half: bool  # Face detected in top 50% of image
    is_correct: bool  # All criteria met


def is_card_oriented_correctly(img_path: Path) -> tuple:
    """
    Check if card orientation is correct:
    - Player's body should be upright (not sideways)
    - Text should be upright (readable, not rotated 90°)
    
    Returns: (is_correct, reason)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return False, "cannot_load"
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Check aspect ratio - baseball cards are ~1.4 (landscape) or ~0.71 (portrait)
    aspect = w / h if h > 0 else 0
    is_portrait = aspect < 1.0
    
    # For portrait cards (aspect < 1), we need to check if rotation is needed
    # Use text orientation detection
    if is_portrait:
        # Portrait orientation - check if text lines are horizontal
        # This would indicate a landscape card rotated 90° incorrectly
        # vs a properly captured portrait card
        pass
    
    # Use edge/line detection to find dominant text line orientation
    edges = cv2.Canny(gray, 50, 150)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=5)
    
    if lines is None or len(lines) < 3:
        # Not enough lines to determine - assume OK
        return True, "no_lines_detected"
    
    # Analyze line angles
    horizontal = 0
    vertical = 0
    diagonal = 0
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        
        if angle < 20 or angle > 160:
            horizontal += 1
        elif 70 <= angle <= 110:
            vertical += 1
        else:
            diagonal += 1
    
    total = horizontal + vertical + diagonal
    if total == 0:
        return True, "no_orientable_lines"
    
    # Dominant horizontal lines = text is upright
    # Dominant vertical lines = card rotated 90°
    if horizontal > vertical * 1.5 and horizontal > diagonal:
        return True, "text_upright"
    elif vertical > horizontal * 1.5 and vertical > diagonal:
        # Check if this might be a portrait card where vertical is expected
        if is_portrait and aspect < 0.85:
            return True, "portrait_card_vertical_text"
        return False, "text_rotated_90"
    else:
        # Mixed angles - could be mixed content (photo + text)
        # Check if we have both strong horizontal AND vertical
        if horizontal > total * 0.3 and vertical > total * 0.3:
            return False, "mixed_orientation"
        return True, "mixed_but_acceptable"


def check_for_background(img_path: Path) -> bool:
    """
    Check if there's visible background (non-card content) in the crop.
    Uses edge density at borders as a proxy.
    
    Returns: True if background detected, False if card fills the frame.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return True  # Assume background if can't load
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Edge detection
    edges = cv2.Canny(gray, 30, 100)
    
    # Check border regions for high edge density (indicates content at edges)
    border_pct = 0.03  # Check outer 3%
    bw = int(w * border_pct)
    bh = int(h * border_pct)
    
    top_edges = np.sum(edges[:bh, :] > 0)
    bottom_edges = np.sum(edges[-bh:, :] > 0)
    left_edges = np.sum(edges[:, :bw] > 0)
    right_edges = np.sum(edges[:, -bw:] > 0)
    
    total_edge_pixels = np.sum(edges > 0)
    if total_edge_pixels == 0:
        return True  # No edges = likely solid color = might be background
    
    border_ratio = (top_edges + bottom_edges + left_edges + right_edges) / total_edge_pixels
    
    # If more than 15% of edges are at borders, likely background visible
    return border_ratio > 0.15


def check_rectangle_squareness(img_path: Path) -> tuple:
    """
    Check if the card is rectangular (not skewed/rotated).
    Uses contour corner detection to measure how close corners are to 90 degrees.
    
    Returns: (is_rectangle, score) - score 0-100 where 100 is perfect rectangle
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return False, 0
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur and threshold
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False, 0
    
    # Get largest contour
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    
    # Must be significant portion of image
    h, w = gray.shape
    if area < h * w * 0.1:
        return False, 0
    
    # Approximate to polygon
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    
    if len(approx) != 4:
        # Not a quadrilateral - likely not a clean rectangle
        return False, 30
    
    # Reshape to corners
    corners = approx.reshape(4, 2).astype(float)
    
    # Calculate angles at each corner
    angles = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        p3 = corners[(i + 2) % 4]
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.abs(np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi)
        angles.append(angle)
    
    # Score based on how close each angle is to 90 degrees
    angle_scores = []
    for angle in angles:
        deviation = abs(angle - 90)
        # Score: 100 if exactly 90, decreases as deviation increases
        score = max(0, 100 - deviation * 2)
        angle_scores.append(score)
    
    avg_score = np.mean(angle_scores)
    
    # Also check for parallel opposite sides
    # Get side vectors
    sides = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        side = p2 - p1
        sides.append(side)
    
    # Opposite sides should be parallel (similar angles)
    # Side 0 vs Side 2, Side 1 vs Side 3
    def angle_between(v1, v2):
        ang1 = np.arctan2(v1[1], v1[0]) * 180 / np.pi
        ang2 = np.arctan2(v2[1], v2[0]) * 180 / np.pi
        diff = abs(ang1 - ang2)
        # Handle opposite directions (180 deg apart = parallel)
        return min(diff, 180 - diff)
    
    parallel_score = 100 - min(100, angle_between(sides[0], sides[2]) + angle_between(sides[1], sides[3]))
    
    # Combined score
    final_score = (avg_score * 0.6) + (parallel_score * 0.4)
    
    # >= 70 Threshold: score means reasonably rectangular
    is_rectangle = final_score >= 70
    
    return is_rectangle, round(final_score, 1)


def check_face_in_top_half(img_path: Path) -> tuple:
    """
    Check if a face is detected in the top 50% of the image.
    Uses OpenCV Haar cascade for face detection.
    
    Returns: (face_in_top_half, face_y_position) - face_y_position is normalized 0-1
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return False, -1
    
    h, w = img.shape[:2]
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try to use Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    if face_cascade.empty():
        # No cascade available - assume OK
        return True, 0.5
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )
    
    if len(faces) == 0:
        # No face detected - try alternate approach using skin color detection
        # This helps detect faces that Haar might miss
        return check_face_skin_color(img_path)
    
    # Check if any face is in top half
    for (x, y, fw, fh) in faces:
        face_center_y = y + fh / 2
        face_top_y = y
        normalized_y = face_center_y / h
        
        # Face is in top half if center is above 0.5 or top of face is above 0.5
        if normalized_y <= 0.5 or (face_top_y / h) <= 0.5:
            return True, normalized_y
    
    return False, normalized_y


def check_face_skin_color(img_path: Path) -> tuple:
    """
    Alternate face detection using skin color segmentation.
    Helps detect faces that Haar cascade might miss.
    
    Returns: (face_in_top_half, face_y_position)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return False, -1
    
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Skin color detection using RGB ranges
    # Loose skin color bounds
    lower_skin = np.array([80, 50, 50], dtype=np.uint8)
    upper_skin = np.array([255, 200, 200], dtype=np.uint8)
    
    mask = cv2.inRange(rgb, lower_skin, upper_skin)
    
    # Also try HSV skin detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
    upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # Combine masks
    combined = cv2.bitwise_or(mask, mask_hsv)
    
    # Clean up with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in skin regions
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False, -1
    
    # Find largest skin-colored region (likely face)
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    
    # Must be significant
    if area < h * w * 0.01:
        return False, -1
    
    # Get bounding box
    x, y, cw, ch = cv2.boundingRect(largest)
    
    # Check if in top half
    face_center_y = y + ch / 2
    normalized_y = face_center_y / h
    
    # Face should be roughly in upper portion and not too wide
    aspect_ratio = cw / ch if ch > 0 else 0
    
    # Likely face if: in top half and reasonable aspect ratio (0.5-2.0)
    if normalized_y <= 0.5 and 0.3 < aspect_ratio < 2.5:
        return True, normalized_y
    
    return False, normalized_y


def evaluate_crop(result: CropResult) -> CropResult:
    """Evaluate a single crop against correctness criteria."""
    img = cv2.imread(str(result.output_path))
    if img is None:
        result.is_correct = False
        return result
    
    h, w = img.shape[:2]
    result.width = w
    result.height = h
    result.aspect_ratio = w / h if h > 0 else 0
    
    # Check 1: Aspect ratio (must be ~1.4, allow 0.71 for portrait)
    aspect_ok = (
        abs(result.aspect_ratio - TARGET_ASPECT) <= TARGET_ASPECT * ASPECT_TOLERANCE or
        abs(result.aspect_ratio - (1/TARGET_ASPECT)) <= (1/TARGET_ASPECT) * ASPECT_TOLERANCE
    )
    
    # Check 2: No background visible
    result.has_background = check_for_background(result.output_path)
    
    # Check 3: Orientation correct
    orientation_ok, reason = is_card_oriented_correctly(result.output_path)
    result.orientation_correct = orientation_ok
    
    # Check 4: Rectangle/squareness (rotation and skew)
    is_rectangle, rect_score = check_rectangle_squareness(result.output_path)
    result.is_rectangle = is_rectangle
    
    # Check 5: Face in top half
    face_in_top, face_y = check_face_in_top_half(result.output_path)
    result.face_in_top_half = face_in_top
    
    # All criteria must pass
    result.is_correct = (
        aspect_ok and 
        not result.has_background and 
        orientation_ok and
        is_rectangle and
        face_in_top
    )
    
    return result


def run_card_crop(input_dir: Path, output_dir: Path, 
                  detector: str = "auto",
                  detector_conf: float = 0.35,
                  ocr_refine: bool = False,
                  ml_refine: bool = True,
                  ml_weight: float = 0.4,
                  padding: int = 0,
                  no_resize: bool = False,
                  debug: bool = False) -> dict:
    """Run card_crop.py with given parameters."""
    
    cmd = [
        "python3", "card_crop.py",
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--no-interactive",
        "--detector", detector,
        "--detector-conf", str(detector_conf),
    ]
    
    if ocr_refine:
        cmd.append("--ocr-refine")
    else:
        cmd.append("--no-ocr-refine")
    
    if ml_refine:
        cmd.append("--ml-refine")
    else:
        cmd.append("--no-ml-refine")
    
    cmd.extend(["--ml-weight", str(ml_weight)])
    
    if padding > 0:
        cmd.extend(["--padding", str(padding)])
    
    if no_resize:
        cmd.append("--no-resize")
    
    if debug:
        cmd.append("--debug")
    
    # Run the command
    result = subprocess.run(
        cmd,
        cwd="/home/andrew/Documents/GitHub/cards",
        capture_output=True,
        text=True,
        timeout=300
    )
    
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr
    }


def test_configuration(input_dir: Path, config: dict, sample_limit: int = 20) -> dict:
    """Test a single configuration and return results."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        
        # Get input files (limit sample)
        input_files = list(input_dir.glob("*.JPG")) + list(input_dir.glob("*.jpg"))
        input_files = input_files[:sample_limit]
        
        # Create temp input dir with limited files
        temp_input = Path(tmpdir) / "input"
        temp_input.mkdir()
        for f in input_files:
            shutil.copy(f, temp_input / f.name)
        
        # Run card_crop
        run_result = run_card_crop(
            temp_input, output_dir,
            detector=config.get("detector", "auto"),
            detector_conf=config.get("detector_conf", 0.35),
            ocr_refine=config.get("ocr_refine", False),
            ml_refine=config.get("ml_refine", True),
            ml_weight=config.get("ml_weight", 0.4),
            padding=config.get("padding", 0),
            no_resize=config.get("no_resize", False),
            debug=config.get("debug", False)
        )
        
        if run_result["returncode"] != 0:
            return {
                "config": config,
                "error": run_result["stderr"],
                "correct_count": 0,
                "total_count": 0,
                "correct_rate": 0.0
            }
        
        # Evaluate results
        results = []
        for out_file in output_dir.glob("*.jpg"):
            result = CropResult(
                source=out_file.stem,
                output_path=out_file,
                success=True,
                aspect_ratio=0,
                width=0,
                height=0,
                has_background=False,
                orientation_correct=False,
                is_rectangle=False,
                face_in_top_half=False,
                is_correct=False
            )
            result = evaluate_crop(result)
            results.append(result)
        
        correct_count = sum(1 for r in results if r.is_correct)
        total_count = len(results)
        
        return {
            "config": config,
            "results": results,
            "correct_count": correct_count,
            "total_count": total_count,
            "correct_rate": correct_count / total_count if total_count > 0 else 0,
            "details": {
                "aspect_failures": sum(1 for r in results if r.aspect_ratio > 0 and 
                                       abs(r.aspect_ratio - TARGET_ASPECT) > TARGET_ASPECT * ASPECT_TOLERANCE and
                                       abs(r.aspect_ratio - 1/TARGET_ASPECT) > (1/TARGET_ASPECT) * ASPECT_TOLERANCE),
                "background_failures": sum(1 for r in results if r.has_background),
                "orientation_failures": sum(1 for r in results if not r.orientation_correct),
                "rectangle_failures": sum(1 for r in results if not r.is_rectangle),
                "face_position_failures": sum(1 for r in results if not r.face_in_top_half),
            }
        }


def generate_configs() -> list:
    """Generate candidate configurations to test."""
    
    configs = []
    
    # Base configs to test
    detectors = ["auto", "rtdetr", "yolo", "contour"]
    detector_confs = [0.25, 0.35, 0.45]
    ocr_refines = [True, False]
    ml_refines = [True, False]
    paddings = [0, 5, 10]
    no_resizes = [False]  # Always resize to get proper aspect
    
    # Generate combinations
    for det in detectors:
        for conf in detector_confs:
            for ocr in ocr_refines:
                for ml in ml_refines:
                    for pad in paddings:
                        configs.append({
                            "detector": det,
                            "detector_conf": conf,
                            "ocr_refine": ocr,
                            "ml_refine": ml,
                            "ml_weight": 0.4,
                            "padding": pad,
                            "no_resize": False,
                            "debug": False
                        })
    
    return configs


def main():
    parser = argparse.ArgumentParser(description="Auto-tune card cropper")
    parser.add_argument("--input-dir", type=Path, 
                        default=Path("/home/andrew/Pictures/data"),
                        help="Input photos directory")
    parser.add_argument("--max-iterations", type=int, default=30,
                        help="Max configurations to test")
    parser.add_argument("--sample-size", type=int, default=20,
                        help="Number of images to sample per config")
    parser.add_argument("--target-rate", type=float, default=0.75,
                        help="Target correct rate (0.0-1.0)")
    parser.add_argument("--output", type=Path, 
                        default=Path("/home/andrew/Documents/GitHub/cards/tuner_results.json"),
                        help="Output results file")
    
    args = parser.parse_args()
    
    print(f"="*60)
    print(f"AUTO-TUNER: Finding optimal card_crop configuration")
    print(f"="*60)
    print(f"Input:       {args.input_dir}")
    print(f"Sample size: {args.sample_size} images per config")
    print(f"Target:      {args.target_rate*100:.0f}% correct")
    print(f"Max iters:   {args.max_iterations}")
    print()
    
    # Generate configurations
    all_configs = generate_configs()
    random.shuffle(all_configs)
    configs_to_test = all_configs[:args.max_iterations]
    
    print(f"Testing {len(configs_to_test)} configurations...")
    print()
    
    results = []
    best_result = None
    
    for i, config in enumerate(configs_to_test):
        config_str = f"{config['detector']}({config['detector_conf']}) OCR={config['ocr_refine']} ML={config['ml_refine']} pad={config['padding']}"
        print(f"[{i+1}/{len(configs_to_test)}] Testing: {config_str}")
        
        try:
            result = test_configuration(args.input_dir, config, args.sample_size)
            results.append(result)
            
            rate = result["correct_rate"]
            correct = result["correct_count"]
            total = result["total_count"]
            
            print(f"    → {correct}/{total} correct ({rate*100:.1f}%)")
            
            if result["details"]:
                details = result["details"]
                print(f"       Failures: aspect={details['aspect_failures']}, "
                      f"bg={details['background_failures']}, "
                      f"orient={details['orientation_failures']}, "
                      f"rect={details.get('rectangle_failures', 0)}, "
                      f"face={details.get('face_position_failures', 0)}")
            
            if best_result is None or rate > best_result["correct_rate"]:
                best_result = result
                print(f"    ★ NEW BEST!")
            
            # Check if we've hit target
            if rate >= args.target_rate:
                print(f"\n*** TARGET ACHIEVED! {rate*100:.0f}% >= {args.target_rate*100:.0f}% ***")
                break
                
        except Exception as e:
            print(f"    → ERROR: {e}")
            continue
    
    # Summary
    print()
    print(f"{'='*60}")
    print(f"TUNING COMPLETE")
    print(f"{'='*60}")
    
    if best_result:
        print(f"Best configuration:")
        print(f"  {best_result['config']}")
        print(f"  Correct rate: {best_result['correct_rate']*100:.1f}%")
        print(f"  ({best_result['correct_count']}/{best_result['total_count']})")
        
        if best_result.get("details"):
            d = best_result["details"]
            print(f"  Failures: aspect={d['aspect_failures']}, bg={d['background_failures']}, orient={d['orientation_failures']}, rect={d.get('rectangle_failures', 0)}, face={d.get('face_position_failures', 0)}")
    
    # Save results
    output_data = {
        "target_rate": args.target_rate,
        "configs_tested": len(results),
        "best_config": best_result["config"] if best_result else None,
        "best_rate": best_result["correct_rate"] if best_result else 0,
        "all_results": [
            {
                "config": r["config"],
                "correct_rate": r["correct_rate"],
                "correct_count": r["correct_count"],
                "total_count": r["total_count"],
                "details": r.get("details", {})
            }
            for r in results
        ]
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    
    return 0 if best_result and best_result["correct_rate"] >= args.target_rate else 1


if __name__ == "__main__":
    exit(main())
