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
    7. Scharr gradient magnitude (strong card-border edges)
    8. LAB distance-from-wood background model (table-color rejection)
    9. GrabCut foreground proposals (document-scanner style fallback)
  Each candidate contour is scored on rectangularity, aspect ratio,
  area, center proximity, and circularity (to reject the round pan).

Requires:  pip install opencv-python numpy pillow

Usage:
    python card_crop.py                                    # process CWD -> ./output/
    python card_crop.py --input-dir photos --output-dir cropped
    python card_crop.py --debug                            # save contour overlay images
    python card_crop.py --padding 10                       # white border (default: 10px)
    python card_crop.py --ocr-refine                       # expand crop to include OCR text
"""

import argparse
import csv
import importlib.util
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
ASPECT_LO = 1.05
ASPECT_HI = 1.85

# Contour area as a fraction of total image area
AREA_MIN = 0.01
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
    p.add_argument("--ocr-refine", action="store_true",
                   help="Use GPU-accelerated OCR (EasyOCR when available) to expand crop bounds so detected text is inside")
    p.add_argument("--ocr-min-conf", type=float, default=0.25,
                   help="Minimum OCR confidence for text boxes used in crop refinement")
    p.add_argument("--ocr-max-dim", type=int, default=1800,
                   help="Max image dimension for OCR pass (for speed)")
    p.add_argument("--ocr-text-margin", type=float, default=0.03,
                   help="Extra margin around OCR text boxes as fraction of card short side")
    p.add_argument("--ocr-csv", default="",
                   help="Optional CSV path to save OCR text per output file (filename + inline text string)")
    p.add_argument("--ml-refine", action="store_true",
                   help="Use CLIP image-text recognition (GPU when available) to improve candidate ranking")
    p.add_argument("--ml-model", default="openai/clip-vit-base-patch32",
                   help="HuggingFace CLIP model id used for semantic card scoring")
    p.add_argument("--ml-device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="Device for ML scorer (default: auto)")
    p.add_argument("--ml-weight", type=float, default=0.40,
                   help="Blend weight [0..1] for CLIP semantic score in final ranking")
    p.add_argument("--ml-required", action="store_true",
                   help="Fail fast if --ml-refine is requested but CLIP could not be loaded")
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
# OCR-based crop refinement (EasyOCR with GPU when available)
# ---------------------------------------------------------------------------
_OCR_READER_CACHE = None


def _get_easyocr_reader():
    """Lazily load EasyOCR reader; prefer GPU when torch+CUDA is available."""
    global _OCR_READER_CACHE
    if _OCR_READER_CACHE is not None:
        return _OCR_READER_CACHE

    try:
        import easyocr
    except Exception:
        _OCR_READER_CACHE = False
        return None

    use_gpu = False
    try:
        import torch
        use_gpu = bool(torch.cuda.is_available())
    except Exception:
        use_gpu = False

    try:
        _OCR_READER_CACHE = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
        return _OCR_READER_CACHE
    except Exception:
        _OCR_READER_CACHE = False
        return None


def _detect_text_entries(img, min_conf=0.25, max_dim=1800):
    """Return OCR entries with box/text/conf in original-image coordinates."""
    reader = _get_easyocr_reader()
    if reader is None:
        return []

    oh, ow = img.shape[:2]
    scale = 1.0
    proc = img
    if max(oh, ow) > max_dim:
        scale = max_dim / max(oh, ow)
        proc = cv2.resize(img, (int(ow * scale), int(oh * scale)), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
    try:
        results = reader.readtext(rgb, detail=1, paragraph=False)
    except Exception:
        return []

    entries = []
    inv = 1.0 / scale
    for item in results:
        if len(item) != 3:
            continue
        pts, txt, conf = item
        if conf is None or conf < min_conf:
            continue
        arr = np.array(pts, dtype=np.float32).reshape(4, 2) * inv
        entries.append({"box": arr, "text": str(txt).strip(), "conf": float(conf)})
    return entries


def _format_ocr_inline(entries):
    """Serialize OCR entries as a compact inline string for CSV output."""
    chunks = []
    for e in entries:
        txt = (e.get("text") or "").replace("|", " ").replace("\n", " ").strip()
        if not txt:
            continue
        chunks.append(f"{txt}({e.get('conf', 0.0):.2f})")
    return " | ".join(chunks)


def _refine_quad_with_ocr(img, quad, min_conf=0.25, max_dim=1800, text_margin_frac=0.03, entries=None):
    """Expand card quad so OCR-detected text near the card remains inside bounds."""
    if entries is None:
        entries = _detect_text_entries(img, min_conf=min_conf, max_dim=max_dim)
    boxes = [e["box"] for e in entries]
    if not boxes:
        return quad, False

    rect = order_points(quad).astype(np.float32)
    tl, tr, br, bl = rect
    w = float(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = float(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if w < 10 or h < 10:
        return quad, False

    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    Minv = np.linalg.inv(M)

    loose = 0.25 * max(w, h)
    picked = []
    for box in boxes:
        pts = cv2.perspectiveTransform(box.reshape(1, 4, 2), M).reshape(4, 2)
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)

        intersects = not (xmax < -loose or ymax < -loose or xmin > (w - 1 + loose) or ymin > (h - 1 + loose))
        if intersects:
            picked.append(pts)

    if not picked:
        return quad, False

    all_pts = np.concatenate(picked, axis=0)
    xmin, ymin = all_pts.min(axis=0)
    xmax, ymax = all_pts.max(axis=0)

    margin = text_margin_frac * min(w, h)
    left_pad = max(0.0, -xmin) + margin
    top_pad = max(0.0, -ymin) + margin
    right_pad = max(0.0, xmax - (w - 1)) + margin
    bot_pad = max(0.0, ymax - (h - 1)) + margin

    if max(left_pad, top_pad, right_pad, bot_pad) < 1.0:
        return quad, False

    expanded = np.array([
        [-left_pad, -top_pad],
        [w - 1 + right_pad, -top_pad],
        [w - 1 + right_pad, h - 1 + bot_pad],
        [-left_pad, h - 1 + bot_pad],
    ], dtype=np.float32)

    refined = cv2.perspectiveTransform(expanded.reshape(1, 4, 2), Minv).reshape(4, 2)

    ih, iw = img.shape[:2]
    refined[:, 0] = np.clip(refined[:, 0], 0, iw - 1)
    refined[:, 1] = np.clip(refined[:, 1], 0, ih - 1)
    return refined.reshape(4, 1, 2).astype(np.float32), True


# ---------------------------------------------------------------------------
# ML semantic scorer (CLIP; GPU-capable)
# ---------------------------------------------------------------------------
_CLIP_SCORER_CACHE = {}
_CLIP_SCORER_STATUS = {}


def _get_clip_card_scorer(model_id="openai/clip-vit-base-patch32", device="auto"):
    """Return a callable that scores how likely an image patch is a baseball card."""
    key = (model_id, device)
    if key in _CLIP_SCORER_CACHE:
        return _CLIP_SCORER_CACHE[key]

    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
        _CLIP_SCORER_CACHE[key] = None
        return None

    import torch
    from transformers import CLIPModel, CLIPProcessor

    if device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = device
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "cpu"

    try:
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id)
    except Exception as e:
        status["reason"] = f"Could not load model '{model_id}': {e}"
        _CLIP_SCORER_CACHE[key] = None
        _CLIP_SCORER_STATUS[key] = status
        return None, status

    model.eval()
    model.to(dev)
    status["enabled"] = True
    status["runtime_device"] = dev
    if not status["reason"]:
        status["reason"] = "loaded"

    prompts = [
        "a close-up photo of a baseball trading card",
        "a sports card on a table",
        "a printed baseball card with text and player image",
        "a wooden tabletop texture",
        "a plain wooden surface",
        "wood grain background",
    ]

    with torch.no_grad():
        text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(dev)
        text_feats = model.get_text_features(**text_inputs)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    pos_idx = [0, 1, 2]
    neg_idx = [3, 4, 5]

    def _score_patch(bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        with torch.no_grad():
            img_inputs = processor(images=pil, return_tensors="pt").to(dev)
            img_feats = model.get_image_features(**img_inputs)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

            sims = (img_feats @ text_feats.T).squeeze(0)
            pos = float(sims[pos_idx].mean().item())
            neg = float(sims[neg_idx].mean().item())
            margin = pos - neg
            return float(1.0 / (1.0 + np.exp(-6.0 * margin)))

    _CLIP_SCORER_CACHE[key] = _score_patch
    _CLIP_SCORER_STATUS[key] = status
    return _score_patch, status


# ---------------------------------------------------------------------------
# Contour scoring
# ---------------------------------------------------------------------------
def _score_contour(cnt, img_area, img_shape):
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

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    # Aspect-ratio closeness to ideal 1.4
    aspect_score = max(0.0, 1.0 - abs(aspect - CARD_ASPECT) / 0.55)

    # Prefer contours in the typical card-area range; penalise very
    # large contours (likely background from inverse masks).
    area_frac = area / img_area
    if area_frac < 0.12:
        size_score = area_frac / 0.12
    elif area_frac <= 0.45:
        size_score = 1.0
    else:
        size_score = max(0.15, 1.0 - (area_frac - 0.45) / 0.20)

    ih, iw = img_shape
    M = cv2.moments(cnt)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = iw / 2, ih / 2
    dc = np.hypot(cx - iw / 2, cy - ih / 2)
    dmax = np.hypot(iw / 2, ih / 2)
    center_score = max(0.0, 1.0 - dc / (dmax * 0.9))

    x, y, bw, bh = cv2.boundingRect(cnt)
    border_pad = max(4, int(min(iw, ih) * 0.01))
    touches_border = (
        x <= border_pad or y <= border_pad
        or (x + bw) >= (iw - border_pad)
        or (y + bh) >= (ih - border_pad)
    )
    # Border contact can happen for valid close-up shots; only penalize small border-touching regions.
    border_penalty = 0.05 if (touches_border and size_score < 0.65) else 0.0

    score = (
        aspect_score    * 0.30
        + rectangularity * 0.25
        + size_score     * 0.10
        + center_score   * 0.10
        + solidity       * 0.05
        + (0.15 if has_four else 0.0)
        - border_penalty
    )

    # Build the 4-point output
    if has_four:
        quad = approx
    else:
        box = cv2.boxPoints(rect)
        quad = np.intp(box).reshape(4, 1, 2)

    return score, quad


def _collect_candidates(contours, img_area, img_shape):
    """Score contours and return list of (score, quad)."""
    results = []
    if img_shape is None and contours:
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        img_shape = (max(1, y + h), max(1, x + w))
    elif img_shape is None:
        img_shape = (1, 1)

    for cnt in contours:
        sc, quad = _score_contour(cnt, img_area, img_shape)
        if sc > 0:
            results.append((sc, quad))
    return results


def _add_mask_candidates(mask, img_area, candidates, strategy_name, img_shape=None):
    """Extract contour candidates from a binary mask and append scored quads."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for sc, quad in _collect_candidates(cnts, img_area, img_shape):
        candidates.append((sc, quad, strategy_name))


def _morph_variants(mask, close_sizes=(11, 15, 21), open_sizes=(7, 11)):
    """Yield morphology variants to bridge weak edges across lighting changes."""
    variants = [mask]
    for csz in close_sizes:
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (csz, csz))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        variants.append(closed)
        for osz in open_sizes:
            k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (osz, osz))
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open)
            variants.append(opened)
    return variants


def _card_content_score(img, quad):
    """
    Score how card-like the warped quad content looks.

    Geometric scoring can still select wood-table patches that happen to be
    roughly rectangular and card-sized. This secondary score prefers regions
    with printed-card characteristics: fine detail/edges plus some color
    richness.
    """
    warped = four_point_transform(img, quad.reshape(4, 2).astype(np.float32))
    if warped is None:
        return 0.0

    h, w = warped.shape[:2]
    if h < 24 or w < 24:
        return 0.0

    # Normalize scale so texture metrics are comparable across candidates.
    short_side = min(h, w)
    scale = 220.0 / short_side
    norm = cv2.resize(
        warped,
        (max(24, int(round(w * scale))), max(24, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )

    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(norm, cv2.COLOR_BGR2HSV)

    # Fine printed text/art yields stronger local gradients than wood grain.
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    lap_score = float(np.clip(np.log1p(lap_var) / 6.0, 0.0, 1.0))

    # Edge density in a card is usually moderate to high because of text,
    # player contours, logos, stat blocks, etc.
    edges = cv2.Canny(gray, 70, 160)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)
    edge_score = float(np.clip(edge_density / 0.11, 0.0, 1.0))

    # Cards are often more saturated than wood, but keep this weight lower
    # so low-saturation vintage cards are not rejected.
    sat_mean = float(hsv[:, :, 1].mean()) / 255.0
    sat_score = float(np.clip((sat_mean - 0.08) / 0.24, 0.0, 1.0))

    return 0.45 * lap_score + 0.35 * edge_score + 0.20 * sat_score


# ---------------------------------------------------------------------------
# Multi-strategy card detection
# ---------------------------------------------------------------------------
def detect_card(img, ml_scorer=None, ml_weight=0.40):
    """
    Try multiple detection strategies to find the card in a phone photo.
    Returns (4-point contour, strategy_name) or (None, None).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    img_shape = (h, w)
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
        _add_mask_candidates(closed, img_area, candidates, f"canny({lo},{hi})", img_shape)

        # Inverse canny mask can isolate lighter cards on darker woods
        inv = cv2.bitwise_not(closed)
        _add_mask_candidates(inv, img_area, candidates, f"canny_inv({lo},{hi})", img_shape)

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
            for idx, variant in enumerate(_morph_variants(binary)):
                _add_mask_candidates(
                    variant, img_area, candidates,
                    f"adapt(b={bsz},C={C})#{idx}", img_shape
                )

            inv_binary = cv2.bitwise_not(binary)
            for idx, variant in enumerate(_morph_variants(inv_binary)):
                _add_mask_candidates(
                    variant, img_area, candidates,
                    f"adapt_inv(b={bsz},C={C})#{idx}", img_shape
                )

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
    for idx, variant in enumerate(_morph_variants(binary3)):
        _add_mask_candidates(variant, img_area, candidates, f"lab+otsu#{idx}", img_shape)
    inv3 = cv2.bitwise_not(binary3)
    for idx, variant in enumerate(_morph_variants(inv3)):
        _add_mask_candidates(variant, img_area, candidates, f"lab+otsu_inv#{idx}", img_shape)

    # ------------------------------------------------------------------
    # Strategy 4 — Simple grayscale Otsu (fallback)
    # ------------------------------------------------------------------
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, binary4 = cv2.threshold(blurred, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for idx, variant in enumerate(_morph_variants(binary4)):
        _add_mask_candidates(variant, img_area, candidates, f"otsu#{idx}", img_shape)
    inv4 = cv2.bitwise_not(binary4)
    for idx, variant in enumerate(_morph_variants(inv4)):
        _add_mask_candidates(variant, img_area, candidates, f"otsu_inv#{idx}", img_shape)

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
    for idx, variant in enumerate(_morph_variants(binary5)):
        _add_mask_candidates(variant, img_area, candidates, f"hsv_sat#{idx}", img_shape)
    inv5 = cv2.bitwise_not(binary5)
    for idx, variant in enumerate(_morph_variants(inv5)):
        _add_mask_candidates(variant, img_area, candidates, f"hsv_sat_inv#{idx}", img_shape)

    # ------------------------------------------------------------------
    # Strategy 6 — Median blur + Otsu
    # Median filter is especially good at wiping out periodic texture
    # patterns like wood grain while preserving the card's sharp edges.
    # ------------------------------------------------------------------
    median = cv2.medianBlur(gray, 15)
    _, binary6 = cv2.threshold(median, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for idx, variant in enumerate(_morph_variants(binary6)):
        _add_mask_candidates(variant, img_area, candidates, f"median+otsu#{idx}", img_shape)
    inv6 = cv2.bitwise_not(binary6)
    for idx, variant in enumerate(_morph_variants(inv6)):
        _add_mask_candidates(variant, img_area, candidates, f"median+otsu_inv#{idx}", img_shape)

    # ------------------------------------------------------------------
    # Strategy 7 — Scharr gradient magnitude
    # Captures strong border edges where thresholding fails due to gloss.
    # ------------------------------------------------------------------
    schx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    schy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(schx, schy)
    mag_u8 = cv2.convertScaleAbs(mag)
    mag_blur = cv2.GaussianBlur(mag_u8, (5, 5), 0)
    _, binary7 = cv2.threshold(mag_blur, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for idx, variant in enumerate(_morph_variants(binary7, close_sizes=(9, 13, 17),
                                                  open_sizes=(5, 9))):
        _add_mask_candidates(variant, img_area, candidates, f"scharr#{idx}", img_shape)

    # ------------------------------------------------------------------
    # Full-frame candidate — if the image aspect ratio matches a card,
    # inject the image boundary as a moderate-score candidate.  In
    # normal photos real card contours outscore it; in re-crops (where
    # only internal features are detected) it wins as a tiebreaker.
    # ------------------------------------------------------------------
    frame_aspect = max(h, w) / max(1, min(h, w))
    if ASPECT_LO <= frame_aspect <= ASPECT_HI:
        frame_aspect_score = max(0.0, 1.0 - abs(frame_aspect - CARD_ASPECT) / 0.55)
        # Moderate score: beats weak internal features, loses to real
        # card contours in card-on-table photos.
        frame_score = 0.35 + 0.20 * frame_aspect_score
        m = 4
        frame_quad = np.array([
            [[m, m]], [[w - 1 - m, m]],
            [[w - 1 - m, h - 1 - m]], [[m, h - 1 - m]]
        ], dtype=np.int32)
        candidates.append((frame_score, frame_quad, "full_frame"))

    if not candidates:
        return None, None

    # ------------------------------------------------------------------
    # Candidate selection with geometry + content scoring.
    # Geometry can still pick table rectangles; add a card-content score
    # from the warped candidate and combine both signals.
    # ------------------------------------------------------------------
    ranked = []
    ml_weight = float(np.clip(ml_weight, 0.0, 1.0)) if ml_scorer is not None else 0.0

    # Limit expensive ML scoring to strongest geometric candidates.
    geom_sorted = sorted(candidates, key=lambda x: x[0], reverse=True)
    ml_limit = min(32, len(geom_sorted))

    for idx, (geom_score, quad, strat) in enumerate(geom_sorted):
        area_frac = cv2.contourArea(
            quad.reshape(-1, 1, 2).astype(np.float32)
        ) / img_area

        content_score = _card_content_score(img, quad)
        base_score = 0.72 * geom_score + 0.28 * content_score

        ml_score = base_score
        if ml_scorer is not None and idx < ml_limit:
            patch = four_point_transform(img, quad.reshape(4, 2).astype(np.float32))
            if patch is not None and min(patch.shape[:2]) >= 24:
                try:
                    ml_score = ml_scorer(patch)
                except Exception:
                    ml_score = base_score

        final_score = (1.0 - ml_weight) * base_score + ml_weight * ml_score

        # Large non-frame regions are often background leakage from inverse
        # masks. Keep them available but with a clear penalty.
        if area_frac > 0.45 and strat != "full_frame":
            final_score -= 0.12

        ranked.append((final_score, geom_score, content_score, ml_score, quad, strat, area_frac))

    ranked.sort(key=lambda x: x[0], reverse=True)
    _, _, _, _, best_quad, strategy, _ = ranked[0]
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
def process_image(src, dst, args, debug_dir=None, ml_scorer=None):
    """Process a single image: detect card, warp, orient, resize, save."""
    img = cv2.imread(str(src))
    if img is None:
        return False, "Could not read image", ""

    oh, ow = img.shape[:2]

    # OCR pass on the full image before any crop/warp so refinement and CSV
    # can use all available text context.
    full_ocr_entries = []
    if args.ocr_refine or args.ocr_csv:
        full_ocr_entries = _detect_text_entries(
            img,
            min_conf=args.ocr_min_conf,
            max_dim=args.ocr_max_dim,
        )

    # Downscale for faster detection on large phone photos
    if max(oh, ow) > DETECT_MAX_DIM:
        scale = DETECT_MAX_DIM / max(oh, ow)
        small = cv2.resize(img, (int(ow * scale), int(oh * scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        small = img
        scale = 1.0

    contour, strategy = detect_card(small, ml_scorer=ml_scorer, ml_weight=args.ml_weight)
    if contour is None:
        # Save failed debug image if requested
        if debug_dir:
            debug_path = debug_dir / f"FAIL_{src.stem}.jpg"
            _save_debug_fail(img, debug_path)
        return False, "No card detected", ""

    # Scale contour back to original resolution
    if scale != 1.0:
        contour = (contour.astype(np.float64) / scale).astype(np.float32)

    ocr_used = False
    if args.ocr_refine:
        contour, ocr_used = _refine_quad_with_ocr(
            img,
            contour,
            min_conf=args.ocr_min_conf,
            max_dim=args.ocr_max_dim,
            text_margin_frac=args.ocr_text_margin,
            entries=full_ocr_entries,
        )

    # Perspective warp on full-resolution image
    warped = four_point_transform(img, contour)
    if warped is None:
        return False, "Warp failed (degenerate quad)", ""

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

    ocr_tag = "+ocr" if ocr_used else ""
    ml_tag = "+ml" if ml_scorer is not None else ""
    return True, f"OK ({strategy}{ocr_tag}{ml_tag})", _format_ocr_inline(ocr_entries)


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
    if args.ml_refine:
        print(f"[INFO] ML     : requested (weight={args.ml_weight:.2f})")
    print()

    ml_scorer = None
    if args.ml_refine:
        ml_scorer = _get_clip_card_scorer(args.ml_model, args.ml_device)
        if ml_scorer is None:
            print("[WARN] ML refine requested but CLIP dependencies/model are unavailable; continuing without ML")
        else:
            print(f"[INFO] ML scorer: ON ({args.ml_model} on {args.ml_device})")

    ok = fail = 0
    ocr_rows = []
    for img_path in images:
        out_name = img_path.stem + ".jpg"
        out_path = output_dir / out_name
        tag = img_path.name
        print(f"  {tag:40s}", end="  ")

        if args.dry_run:
            print("[DRY RUN]")
            continue

        success, msg, ocr_inline = process_image(img_path, out_path, args, debug_dir, ml_scorer=ml_scorer)
        if success:
            ok += 1
            print(f"[OK] {msg}")
        else:
            fail += 1
            print(f"[FAIL] {msg}")

        if args.ocr_csv and not args.dry_run:
            ocr_rows.append({"filename": out_name, "ocr_text": ocr_inline if success else ""})

    if not args.dry_run:
        print(f"\n[DONE] {ok} succeeded, {fail} failed -> {output_dir}")
        if args.debug and debug_dir:
            print(f"[DEBUG] Overlays saved in -> {debug_dir}")

        if args.ocr_csv:
            csv_path = Path(args.ocr_csv)
            if not csv_path.is_absolute():
                csv_path = output_dir / csv_path
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["filename", "ocr_text"])
                w.writeheader()
                w.writerows(ocr_rows)
            print(f"[OCR] CSV saved -> {csv_path}")


if __name__ == "__main__":
    main()
