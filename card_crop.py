"""
card_crop.py — Detect, crop, and straighten baseball cards from phone photos.

Optimized for:
  * Phone photos taken looking down at cards on a wood table
  * Wood grain background (not solid color)
  * Random card angles, partially visible objects (pans, etc.) in frame
  * Indoor, uneven lighting
  * Card occupies roughly 25-40% of the frame

Detection strategy:
  The wood grain background makes simple thresholding unreliable, so we
  use multiple complementary strategies and score the results:
    1. Bilateral filter + Canny edges (smooths grain, preserves card edges)
    2. Adaptive thresholding (handles uneven lighting)
    3. LAB L-channel + Otsu (perceptual lightness separates card from wood)
    4. Grayscale Otsu (simple fallback)
    5. HSV saturation + Otsu (dark-bordered cards are colorful vs brown wood)
    6. Median blur + Otsu (median filter kills periodic wood grain patterns)
  Each candidate contour is scored on rectangularity, aspect ratio,
  area, and circularity (to reject the round pan).

Requires:  pip install opencv-python numpy pillow

Usage:
    python card_crop.py                                    # process CWD -> ./output/
    python card_crop.py --input-dir photos --output-dir cropped
    python card_crop.py --debug                            # save contour overlay images
    python card_crop.py --padding 10                       # white border (default: 10px)
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Standard baseball card: 2.5" x 3.5" @ 300 DPI
CARD_W_PX = 750
CARD_H_PX = 1050
CARD_ASPECT = CARD_H_PX / CARD_W_PX  # 1.4

# Acceptable aspect-ratio window for a card candidate
ASPECT_LO = 1.15
ASPECT_HI = 1.65

# Contour area as a fraction of total image area
AREA_MIN = 0.04
AREA_MAX = 0.65

# Maximum image dimension during detection (speed vs accuracy)
DETECT_MAX_DIM = 1500


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Baseball card photo cropper")
    p.add_argument("--input-dir", default=".",
                   help="Folder of input images (default: .)")
    p.add_argument("--output-dir", default="output",
                   help="Folder for results (default: ./output)")
    p.add_argument("--ext",
                   default="jpg,jpeg,png,tiff,bmp,JPG,JPEG,PNG,TIFF,BMP",
                   help="Comma-separated file extensions to process")
    p.add_argument("--padding", default=10, type=int,
                   help="White border in px added after crop (default: 10)")
    p.add_argument("--no-resize", action="store_true",
                   help="Skip resize to standard card dimensions")
    p.add_argument("--debug", action="store_true",
                   help="Save debug images showing detected contour")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def order_points(pts):
    """Order 4 points: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],      # TL: smallest x+y
        pts[np.argmin(d)],      # TR: smallest y-x
        pts[np.argmax(s)],      # BR: largest  x+y
        pts[np.argmax(d)],      # BL: largest  y-x
    ], dtype=np.float32)


def four_point_transform(image, pts):
    """Perspective-warp a quadrilateral to a flat rectangle."""
    rect = order_points(pts)
    tl, tr, br, bl = rect

    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if w < 10 or h < 10:
        return None

    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (w, h))


# ---------------------------------------------------------------------------
# Contour scoring
# ---------------------------------------------------------------------------
def _score_contour(cnt, img_area):
    """
    Score how likely a contour is to be a baseball card.
    Returns (score, 4-point quad) or (-1, None) if rejected.
    """
    area = cv2.contourArea(cnt)
    if area < img_area * AREA_MIN or area > img_area * AREA_MAX:
        return -1, None

    peri = cv2.arcLength(cnt, True)
    if peri < 1:
        return -1, None

    # Reject circles (the pan).  Circularity of a perfect circle ~ 1.0
    circularity = 4 * np.pi * area / (peri * peri)
    if circularity > 0.85:
        return -1, None

    # Bounding rotated rectangle
    rect = cv2.minAreaRect(cnt)
    rw, rh = rect[1]
    if min(rw, rh) < 1:
        return -1, None

    aspect = max(rw, rh) / min(rw, rh)
    if aspect < ASPECT_LO or aspect > ASPECT_HI:
        return -1, None

    # Rectangularity: how much of the rotated rect the contour fills
    rect_area = rw * rh
    rectangularity = area / rect_area if rect_area > 0 else 0

    # Polygon approximation — prefer exactly 4 corners
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    has_four = (len(approx) == 4)

    # Aspect-ratio closeness to ideal 1.4
    aspect_score = max(0.0, 1.0 - abs(aspect - CARD_ASPECT) / 0.5)

    # Larger contours (within bounds) are more likely to be the card
    size_score = area / img_area

    score = (
        aspect_score    * 0.35
        + rectangularity * 0.30
        + size_score     * 0.20
        + (0.15 if has_four else 0.0)
    )

    # Build the 4-point output
    if has_four:
        quad = approx
    else:
        box = cv2.boxPoints(rect)
        quad = np.intp(box).reshape(4, 1, 2)

    return score, quad


def _collect_candidates(contours, img_area):
    """Score contours and return list of (score, quad)."""
    results = []
    for cnt in contours:
        sc, quad = _score_contour(cnt, img_area)
        if sc > 0:
            results.append((sc, quad))
    return results


# ---------------------------------------------------------------------------
# Multi-strategy card detection
# ---------------------------------------------------------------------------
def detect_card(img):
    """
    Try multiple detection strategies to find the card in a phone photo.
    Returns (4-point contour, strategy_name) or (None, None).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    img_area = h * w
    candidates = []   # (score, quad, strategy_name)

    # ------------------------------------------------------------------
    # Strategy 1 — Bilateral filter + Canny edges
    # Bilateral smooths wood grain texture while preserving the strong
    # edge at the card boundary.  Multiple Canny threshold pairs give
    # robustness across different lighting conditions.
    # ------------------------------------------------------------------
    bilateral = cv2.bilateralFilter(gray, 11, 75, 75)
    for lo, hi in [(30, 100), (50, 150), (20, 70)]:
        edges = cv2.Canny(bilateral, lo, hi)
        kern_d = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.dilate(edges, kern_d, iterations=3)
        closed = cv2.morphologyEx(
            closed, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        )
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        for sc, quad in _collect_candidates(cnts, img_area):
            candidates.append((sc, quad, f"canny({lo},{hi})"))

    # ------------------------------------------------------------------
    # Strategy 2 — Adaptive threshold
    # Handles uneven indoor lighting.  The card's light border becomes
    # foreground against the darker wood background.
    # ------------------------------------------------------------------
    bilateral2 = cv2.bilateralFilter(gray, 15, 80, 80)
    for bsz in [31, 51, 71]:
        for C in [-5, -10, -20]:
            binary = cv2.adaptiveThreshold(
                bilateral2, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                bsz, C
            )
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
            cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            for sc, quad in _collect_candidates(cnts, img_area):
                candidates.append((sc, quad, f"adapt(b={bsz},C={C})"))

    # ------------------------------------------------------------------
    # Strategy 3 — LAB L-channel + Otsu
    # The L channel in LAB represents perceptual lightness, giving good
    # card-vs-wood separation even with color variation in the wood.
    # ------------------------------------------------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0]
    l_blur = cv2.GaussianBlur(l_ch, (7, 7), 0)
    _, binary3 = cv2.threshold(l_blur, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    binary3 = cv2.morphologyEx(binary3, cv2.MORPH_CLOSE, k)
    binary3 = cv2.morphologyEx(binary3, cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(binary3, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    for sc, quad in _collect_candidates(cnts, img_area):
        candidates.append((sc, quad, "lab+otsu"))

    # ------------------------------------------------------------------
    # Strategy 4 — Simple grayscale Otsu (fallback)
    # ------------------------------------------------------------------
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, binary4 = cv2.threshold(blurred, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    binary4 = cv2.morphologyEx(binary4, cv2.MORPH_CLOSE, k)
    binary4 = cv2.morphologyEx(binary4, cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(binary4, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    for sc, quad in _collect_candidates(cnts, img_area):
        candidates.append((sc, quad, "otsu"))

    # ------------------------------------------------------------------
    # Strategy 5 — HSV saturation + Otsu
    # Card fronts with dark borders are hard to detect by brightness
    # alone, but they're far more colorful/saturated than the brown
    # wood table.  This separates vibrant card art from muted wood.
    # ------------------------------------------------------------------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s_ch = hsv[:, :, 1]
    s_blur = cv2.GaussianBlur(s_ch, (7, 7), 0)
    _, binary5 = cv2.threshold(s_blur, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    binary5 = cv2.morphologyEx(binary5, cv2.MORPH_CLOSE, k)
    binary5 = cv2.morphologyEx(binary5, cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(binary5, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    for sc, quad in _collect_candidates(cnts, img_area):
        candidates.append((sc, quad, "hsv_sat"))

    # ------------------------------------------------------------------
    # Strategy 6 — Median blur + Otsu
    # Median filter is especially good at wiping out periodic texture
    # patterns like wood grain while preserving the card's sharp edges.
    # ------------------------------------------------------------------
    median = cv2.medianBlur(gray, 15)
    _, binary6 = cv2.threshold(median, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    binary6 = cv2.morphologyEx(binary6, cv2.MORPH_CLOSE, k)
    binary6 = cv2.morphologyEx(binary6, cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(binary6, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    for sc, quad in _collect_candidates(cnts, img_area):
        candidates.append((sc, quad, "median+otsu"))

    if not candidates:
        return None, None

    # Pick the highest-scoring candidate
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_quad, strategy = candidates[0]
    return best_quad, strategy


# ---------------------------------------------------------------------------
# Orientation helpers
# ---------------------------------------------------------------------------
def orient_portrait(img):
    """Rotate to portrait if the image came out landscape."""
    h, w = img.shape[:2]
    if w > h:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def fix_upside_down(img):
    """
    Heuristic: baseball cards usually have a player photo on top and a
    lighter stat bar / name plate on the bottom third.  If the bottom
    third is significantly brighter than the top third, assume the card
    is upside-down and rotate 180 degrees.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h = gray.shape[0]
    top_mean = gray[:h // 3, :].mean()
    bot_mean = gray[2 * h // 3:, :].mean()
    if bot_mean > top_mean + 25:
        img = cv2.rotate(img, cv2.ROTATE_180)
    return img


# ---------------------------------------------------------------------------
# Image processing pipeline
# ---------------------------------------------------------------------------
def process_image(src, dst, args, debug_dir=None):
    """Process a single image: detect card, warp, orient, resize, save."""
    img = cv2.imread(str(src))
    if img is None:
        return False, "Could not read image"

    oh, ow = img.shape[:2]

    # Downscale for faster detection on large phone photos
    if max(oh, ow) > DETECT_MAX_DIM:
        scale = DETECT_MAX_DIM / max(oh, ow)
        small = cv2.resize(img, (int(ow * scale), int(oh * scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        small = img
        scale = 1.0

    contour, strategy = detect_card(small)
    if contour is None:
        # Save failed debug image if requested
        if debug_dir:
            debug_path = debug_dir / f"FAIL_{src.stem}.jpg"
            _save_debug_fail(img, debug_path)
        return False, "No card detected"

    # Scale contour back to original resolution
    if scale != 1.0:
        contour = (contour.astype(np.float64) / scale).astype(np.float32)

    # Perspective warp on full-resolution image
    warped = four_point_transform(img, contour)
    if warped is None:
        return False, "Warp failed (degenerate quad)"

    warped = orient_portrait(warped)
    warped = fix_upside_down(warped)

    # Add white padding
    if args.padding > 0:
        p = args.padding
        warped = cv2.copyMakeBorder(warped, p, p, p, p,
                                    cv2.BORDER_CONSTANT,
                                    value=(255, 255, 255))

    # Resize to standard card dimensions
    if not args.no_resize:
        warped = cv2.resize(warped, (CARD_W_PX, CARD_H_PX),
                            interpolation=cv2.INTER_LANCZOS4)

    cv2.imwrite(str(dst), warped, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Debug overlay — draw detected contour on the original image
    if debug_dir:
        _save_debug_overlay(img, contour, strategy, debug_dir, src.stem)

    return True, f"OK ({strategy})"


def _save_debug_overlay(img, contour, strategy, debug_dir, stem):
    """Save an annotated copy of the original with the detected quad."""
    debug_img = img.copy()
    oh = img.shape[0]
    pts = contour.reshape(4, 2).astype(np.int32)

    # Green quad outline
    thickness = max(3, oh // 300)
    cv2.polylines(debug_img, [pts], True, (0, 255, 0), thickness)

    # Red corner dots
    radius = max(8, oh // 200)
    for pt in pts:
        cv2.circle(debug_img, tuple(pt), radius, (0, 0, 255), -1)

    # Strategy label
    font_scale = max(0.8, oh / 2000)
    cv2.putText(debug_img, strategy, (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

    # Shrink for reasonable file size
    if max(debug_img.shape[:2]) > 2000:
        ds = 2000 / max(debug_img.shape[:2])
        debug_img = cv2.resize(debug_img, None, fx=ds, fy=ds)

    debug_path = debug_dir / f"debug_{stem}.jpg"
    cv2.imwrite(str(debug_path), debug_img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def _save_debug_fail(img, debug_path):
    """Save a reduced copy of an image that failed detection."""
    if max(img.shape[:2]) > 2000:
        ds = 2000 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=ds, fy=ds)
    cv2.imwrite(str(debug_path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------
def collect_images(input_dir, extensions):
    images = []
    for ext in extensions:
        images.extend(input_dir.glob(f"*.{ext}"))
    return sorted(set(images))


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    extensions = [e.strip().lstrip(".") for e in args.ext.split(",")]

    if not input_dir.is_dir():
        print(f"[ERROR] Input directory not found: {input_dir}")
        sys.exit(1)

    images = collect_images(input_dir, extensions)
    if not images:
        print(f"[INFO] No images found in {input_dir}")
        sys.exit(0)

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = None
    if args.debug and not args.dry_run:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(exist_ok=True)

    print(f"[INFO] Input  : {input_dir}")
    print(f"[INFO] Output : {output_dir}")
    print(f"[INFO] Images : {len(images)}")
    print(f"[INFO] Padding: {args.padding}px")
    if args.debug:
        print(f"[INFO] Debug  : ON")
    print()

    ok = fail = 0
    for img_path in images:
        out_name = img_path.stem + ".jpg"
        out_path = output_dir / out_name
        tag = img_path.name
        print(f"  {tag:40s}", end="  ")

        if args.dry_run:
            print("[DRY RUN]")
            continue

        success, msg = process_image(img_path, out_path, args, debug_dir)
        if success:
            ok += 1
            print(f"[OK] {msg}")
        else:
            fail += 1
            print(f"[FAIL] {msg}")

    if not args.dry_run:
        print(f"\n[DONE] {ok} succeeded, {fail} failed -> {output_dir}")
        if args.debug and debug_dir:
            print(f"[DEBUG] Overlays saved in -> {debug_dir}")


if __name__ == "__main__":
    main()
