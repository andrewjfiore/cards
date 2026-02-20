"""
card_crop.py
------------
Batch crop and straighten baseball card scans.
Replaces scan-buddy entirely — no external repo needed.

Requires:  pip install opencv-python numpy pillow

Usage:
    python card_crop.py                         # process CWD, output -> ./output/
    python card_crop.py --input-dir scans --output-dir cropped
    python card_crop.py --debug                 # saves debug images showing detected contour
    python card_crop.py --padding 20            # add N px white border around card
    python card_crop.py --bg dark               # scanner background is dark (default: auto)

How it works:
    1. Convert to grayscale, blur, threshold to isolate card from background
    2. Find the largest 4-corner contour (the card rectangle)
    3. Perspective-warp that quad to a flat rectangle
    4. Resize to standard baseball card output resolution (750x1050 px = 2.5x3.5" @ 300dpi)
    5. Save to output dir

Tuning:
    If cards aren't detected well, try --bg dark or --bg light to force
    background detection mode, or --thresh to adjust the binary threshold value.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Standard baseball card: 2.5" x 3.5" @ 300 DPI
CARD_W_PX = 750
CARD_H_PX = 1050
CARD_ASPECT = CARD_H_PX / CARD_W_PX  # 1.4


def parse_args():
    p = argparse.ArgumentParser(description="Batch baseball card scan cropper")
    p.add_argument("--input-dir",  default=".",      help="Folder of input images (default: .)")
    p.add_argument("--output-dir", default="output", help="Folder for results (default: ./output)")
    p.add_argument("--ext",        default="jpg,jpeg,png,tiff,bmp,JPG,JPEG,PNG,TIFF,BMP")
    p.add_argument("--bg",         default="auto",   choices=["auto","dark","light"],
                                                     help="Scanner background color hint")
    p.add_argument("--thresh",     default=None, type=int,
                                                     help="Manual binary threshold 0-255 (skips auto)")
    p.add_argument("--padding",    default=0,    type=int,
                                                     help="Extra white border in pixels added after crop")
    p.add_argument("--no-resize",  action="store_true",
                                                     help="Skip resize to standard card dimensions")
    p.add_argument("--debug",      action="store_true",
                                                     help="Save debug contour images alongside outputs")
    p.add_argument("--dry-run",    action="store_true")
    return p.parse_args()


def order_points(pts):
    """Order 4 points: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],    # top-left
        pts[np.argmin(diff)], # top-right
        pts[np.argmax(s)],    # bottom-right
        pts[np.argmax(diff)], # bottom-left
    ], dtype=np.float32)


def four_point_transform(image, pts):
    """Perspective warp a quadrilateral to a rectangle."""
    rect = order_points(pts)
    tl, tr, br, bl = rect

    width  = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    height = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    width, height = int(width), int(height)

    dst = np.array([
        [0,         0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0,         height - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (width, height))


def detect_card_contour(gray, bg_hint="auto", manual_thresh=None):
    """
    Return the 4-point contour of the card, or None if not found.
    Tries multiple strategies to handle different scan conditions.
    """
    h, w = gray.shape

    # Determine background: scanners are usually very dark or very white
    if bg_hint == "auto":
        # Sample corners to judge background brightness
        corners = [gray[0:50, 0:50], gray[0:50, w-50:w],
                   gray[h-50:h, 0:50], gray[h-50:h, w-50:w]]
        bg_mean = np.mean([c.mean() for c in corners])
        bg_hint = "dark" if bg_mean < 128 else "light"

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    if manual_thresh is not None:
        if bg_hint == "dark":
            _, binary = cv2.threshold(blurred, manual_thresh, 255, cv2.THRESH_BINARY)
        else:
            _, binary = cv2.threshold(blurred, manual_thresh, 255, cv2.THRESH_BINARY_INV)
    else:
        if bg_hint == "dark":
            # Card is brighter than the dark scanner bed
            _, binary = cv2.threshold(blurred, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # Card is darker than white background — invert
            _, binary = cv2.threshold(blurred, 0, 255,
                                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, binary

    # Sort by area descending — card should be the largest contour
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    img_area = h * w
    for cnt in contours[:5]:  # check top 5 largest
        area = cv2.contourArea(cnt)
        if area < img_area * 0.05:  # skip tiny blobs
            continue

        # Try to approximate to 4 corners
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            return approx, binary

        # If not 4 corners, use the bounding rotated rect as fallback
        rect = cv2.minAreaRect(cnt)
        box  = cv2.boxPoints(rect)
        box  = np.intp(box)
        return box.reshape(4, 1, 2), binary

    return None, binary


def orient_to_portrait(img):
    """Flip to portrait if image came out landscape."""
    h, w = img.shape[:2]
    if w > h:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def fix_upside_down(img):
    """
    Baseball cards: detect if the image is upside-down by checking
    whether more bright content is in the top or bottom half.
    Cards typically have the photo in the upper 60% and the stat box below —
    the stat box is usually lighter (white/cream).
    
    Simple heuristic: if bottom half is significantly brighter than top, flip.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h = gray.shape[0]
    top_mean    = gray[:h//2, :].mean()
    bottom_mean = gray[h//2:, :].mean()
    if bottom_mean > top_mean + 15:
        # Bottom is brighter → card is upside down
        img = cv2.rotate(img, cv2.ROTATE_180)
    return img


def process_image(src: Path, dst: Path, args, debug_dir=None):
    img = cv2.imread(str(src))
    if img is None:
        return False, "Could not read image"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contour, binary = detect_card_contour(gray, args.bg, args.thresh)

    if contour is None:
        return False, "No card contour found"

    # Perspective warp
    warped = four_point_transform(img, contour)

    # Orient to portrait
    warped = orient_to_portrait(warped)

    # Try to fix upside-down cards
    warped = fix_upside_down(warped)

    # Add padding
    if args.padding > 0:
        p = args.padding
        warped = cv2.copyMakeBorder(warped, p, p, p, p,
                                    cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # Resize to standard card dimensions
    if not args.no_resize:
        h, w = warped.shape[:2]
        # Decide orientation of output based on warped content
        if h >= w:
            out_w, out_h = CARD_W_PX, CARD_H_PX
        else:
            out_w, out_h = CARD_H_PX, CARD_W_PX
        warped = cv2.resize(warped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

    cv2.imwrite(str(dst), warped, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Debug output
    if debug_dir:
        debug_img = img.copy()
        cv2.drawContours(debug_img, [contour], -1, (0, 255, 0), 8)
        debug_path = debug_dir / f"debug_{src.name}"
        cv2.imwrite(str(debug_path), debug_img)

    return True, "OK"


def collect_images(input_dir: Path, extensions: list[str]) -> list[Path]:
    images = []
    for ext in extensions:
        images.extend(input_dir.glob(f"*.{ext}"))
    return sorted(set(images))


def main():
    args = parse_args()
    input_dir  = Path(args.input_dir).resolve()
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
    print(f"[INFO] BG mode: {args.bg}")
    print()

    ok = fail = 0
    for img_path in images:
        out_path = output_dir / img_path.name
        print(f"  {img_path.name}  ->  {out_path.name}", end="  ")

        if args.dry_run:
            print("[DRY RUN]")
            continue

        success, msg = process_image(img_path, out_path, args, debug_dir)
        if success:
            ok += 1
            print(f"[OK]")
        else:
            fail += 1
            print(f"[FAILED] {msg}")

    if not args.dry_run:
        print(f"\n[DONE] {ok} ok, {fail} failed — results in: {output_dir}")
        if args.debug:
            print(f"[DEBUG] Contour overlays saved in: {debug_dir}")


if __name__ == "__main__":
    main()
