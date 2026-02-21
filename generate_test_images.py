"""Generate 10 realistic test images of baseball cards on a wood table.

Produces images that exercise the card_crop.py detector under conditions
observed in real phone-camera photos:
  - Realistic wood-grain texture with knots and color variation
  - Indoor lighting with soft shadows and vignetting
  - Cards at varying angles and positions (not always centered)
  - JPEG-like noise and slight blur
  - Dark-bordered and light-bordered card styles
  - Glare / specular highlights
  - Slight perspective (not perfectly top-down)
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def _perlin_like_noise(h, w, scale, rng):
    """Quick low-freq noise via upscaled random grid."""
    sh, sw = max(2, h // scale), max(2, w // scale)
    small = rng.random((sh, sw)).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def wood_background(h, w, rng):
    """Generate a realistic wood-grain background."""
    # Base brown color with spatial gradient
    y_coord = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x_coord = np.linspace(0, 1, w, dtype=np.float32)[None, :]

    # Randomize the base wood tone
    r_base = rng.uniform(100, 160)
    g_base = rng.uniform(70, 120)
    b_base = rng.uniform(40, 80)

    bg = np.zeros((h, w, 3), np.float32)
    bg[..., 2] = r_base + rng.uniform(-15, 15) * y_coord + rng.uniform(-10, 10) * x_coord
    bg[..., 1] = g_base + rng.uniform(-12, 12) * y_coord + rng.uniform(-8, 8) * x_coord
    bg[..., 0] = b_base + rng.uniform(-10, 10) * y_coord + rng.uniform(-6, 6) * x_coord

    # Wood grain — horizontal streaks at varying angles
    n_grains = rng.integers(15, 35)
    for _ in range(n_grains):
        gy = rng.uniform(0, h)
        thickness = rng.uniform(1.5, 6.0)
        intensity = rng.uniform(-25, 25)
        angle = rng.uniform(-0.03, 0.03)
        for row in range(h):
            offset = int(angle * (row - h / 2))
            y_dist = abs(row - gy)
            if y_dist < thickness * 4:
                falloff = np.exp(-0.5 * (y_dist / thickness) ** 2)
                col_start = max(0, offset)
                col_end = min(w, w + offset)
                bg[row, col_start:col_end] += intensity * falloff

    # Low-frequency color variation (knots, stains)
    lf = _perlin_like_noise(h, w, 80, rng)
    bg += (lf[..., None] - 0.5) * 40

    # High-frequency noise
    noise = rng.normal(0, 4, size=bg.shape).astype(np.float32)
    bg += noise

    return np.clip(bg, 0, 255).astype(np.uint8)


def card_patch(rng, style="light"):
    """Generate a realistic baseball card image."""
    card_h, card_w = 700, 500

    if style == "dark":
        # Dark-bordered card (like many Topps designs)
        card = np.full((card_h, card_w, 3), 30, np.uint8)
        border_color = tuple(int(v) for v in rng.integers(10, 60, size=3))
        inner_bg = tuple(int(v) for v in rng.integers(200, 250, size=3))
    else:
        # Light/white bordered card
        card = np.full((card_h, card_w, 3), 245, np.uint8)
        border_color = tuple(int(v) for v in rng.integers(200, 255, size=3))
        inner_bg = (250, 250, 250)

    # Outer border
    cv2.rectangle(card, (0, 0), (card_w - 1, card_h - 1), border_color, thickness=12)

    # Inner frame
    cv2.rectangle(card, (18, 18), (card_w - 18, card_h - 18), inner_bg, thickness=-1)

    # Player photo area — random colorful region
    art_top = 60
    art_bot = int(card_h * 0.68)
    art_left = 30
    art_right = card_w - 30
    art_h = art_bot - art_top
    art_w = art_right - art_left

    # Simulate player photo with blended random colors
    art = rng.integers(30, 230, size=(art_h, art_w, 3), dtype=np.uint8)
    art = cv2.GaussianBlur(art, (15, 15), 0)

    # Add a "sky" gradient at top
    sky = np.zeros((art_h, art_w, 3), np.float32)
    for row in range(min(art_h // 3, art_h)):
        t = row / max(1, art_h // 3)
        sky[row, :] = [200 + 40 * (1 - t), 180 + 50 * (1 - t), 120 + 80 * (1 - t)]
    mask = np.clip(sky, 0, 255).astype(np.uint8)
    alpha = 0.4
    art[:art_h // 3] = cv2.addWeighted(art[:art_h // 3], 1 - alpha, mask[:art_h // 3], alpha, 0)

    card[art_top:art_bot, art_left:art_right] = art

    # Name plate
    name_top = int(card_h * 0.04)
    cv2.rectangle(card, (25, name_top), (card_w - 25, name_top + 45),
                  (230, 230, 230) if style != "dark" else (50, 50, 50), -1)
    text_color = (20, 20, 120) if style != "dark" else (220, 220, 50)
    names = ["MIKE TROUT", "AARON JUDGE", "SHOHEI OHTANI", "MOOKIE BETTS",
             "BRYCE HARPER", "JUAN SOTO", "TATIS JR", "VLADDY JR",
             "ACUNA JR", "YORDAN ALVAREZ"]
    name = names[rng.integers(0, len(names))]
    cv2.putText(card, name, (35, name_top + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

    # Stats area at bottom
    stats_top = int(card_h * 0.72)
    cv2.rectangle(card, (25, stats_top), (card_w - 25, card_h - 25),
                  (240, 240, 240) if style != "dark" else (45, 45, 45), -1)
    for i in range(5):
        y = stats_top + 20 + i * 25
        cv2.line(card, (35, y), (card_w - 35, y), (100, 100, 100), 1)
        # Fake stat text
        cv2.putText(card, f".{rng.integers(200,350)}", (40, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)

    # Team logo circle
    logo_center = (card_w - 60, 45)
    cv2.circle(card, logo_center, 22,
               tuple(int(v) for v in rng.integers(50, 200, size=3)), -1)

    return card


def composite_card(bg, card, rng, center_frac=None, angle=None, area_frac=None):
    """Warp card onto background with perspective."""
    h, w = bg.shape[:2]
    ch, cw = card.shape[:2]

    if area_frac is None:
        area_frac = rng.uniform(0.15, 0.40)
    target_area = h * w * area_frac
    card_aspect = ch / cw
    target_w = np.sqrt(target_area / card_aspect)
    target_h = target_w * card_aspect

    if center_frac is None:
        cx = rng.uniform(w * 0.25, w * 0.75)
        cy = rng.uniform(h * 0.25, h * 0.75)
    else:
        cx = w * center_frac[0]
        cy = h * center_frac[1]

    if angle is None:
        angle = rng.uniform(-25, 25)

    # Build rotated rectangle corners
    cos_a = np.cos(np.radians(angle))
    sin_a = np.sin(np.radians(angle))
    hw, hh = target_w / 2, target_h / 2
    corners = np.array([
        [-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]
    ], dtype=np.float32)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    rotated = corners @ rot.T
    rotated[:, 0] += cx
    rotated[:, 1] += cy

    # Add slight perspective distortion (not perfectly top-down)
    persp_strength = rng.uniform(0.0, 0.06)
    persp_jitter = rng.uniform(-1, 1, size=(4, 2)).astype(np.float32)
    persp_jitter *= np.array([target_w, target_h]) * persp_strength
    dst_pts = np.clip(rotated + persp_jitter, [0, 0], [w - 1, h - 1]).astype(np.float32)

    src_pts = np.array([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped = cv2.warpPerspective(card, M, (w, h))
    mask = cv2.warpPerspective(np.full((ch, cw), 255, np.uint8), M, (w, h))

    # Composite with slight edge anti-aliasing
    mask_blur = cv2.GaussianBlur(mask, (3, 3), 0)
    alpha = mask_blur.astype(np.float32) / 255.0
    out = bg.astype(np.float32)
    for c in range(3):
        out[..., c] = out[..., c] * (1 - alpha) + warped[..., c].astype(np.float32) * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)

    # Slight shadow under card
    shadow_mask = cv2.GaussianBlur(mask, (21, 21), 0)
    shadow_alpha = (shadow_mask.astype(np.float32) / 255.0) * 0.2
    for c in range(3):
        shifted = np.roll(np.roll(out[..., c].astype(np.float32), 4, axis=0), 3, axis=1)
        out[..., c] = np.clip(
            out[..., c].astype(np.float32) * (1 - shadow_alpha * 0.5) + shifted * shadow_alpha * 0.1,
            0, 255
        ).astype(np.uint8)

    return out, dst_pts


def add_lighting(img, rng):
    """Add realistic indoor lighting: vignette + uneven illumination."""
    h, w = img.shape[:2]
    out = img.astype(np.float32)

    # Vignette
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2 + rng.uniform(-w * 0.1, w * 0.1), h / 2 + rng.uniform(-h * 0.1, h * 0.1)
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    vignette = 1.0 - 0.3 * (dist / max_dist) ** 2
    out *= vignette[..., None]

    # Uneven illumination gradient
    angle = rng.uniform(0, 2 * np.pi)
    grad = np.cos(angle) * (X - w / 2) / w + np.sin(angle) * (Y - h / 2) / h
    out *= (1.0 + grad * rng.uniform(0.05, 0.15))[..., None]

    return np.clip(out, 0, 255).astype(np.uint8)


def add_glare(img, rng):
    """Add specular highlight / glare spot."""
    h, w = img.shape[:2]
    overlay = img.copy()
    cx = int(rng.uniform(w * 0.2, w * 0.8))
    cy = int(rng.uniform(h * 0.2, h * 0.8))
    axes = (int(rng.uniform(w * 0.04, w * 0.15)), int(rng.uniform(h * 0.03, h * 0.1)))
    angle = rng.uniform(0, 180)
    cv2.ellipse(overlay, (cx, cy), axes, angle, 0, 360, (255, 255, 255), -1)
    alpha = rng.uniform(0.1, 0.3)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def add_noise(img, rng, sigma=6):
    """Add camera sensor noise."""
    noise = rng.normal(0, sigma, size=img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# --- Scene configurations for 10 diverse test images ---
SCENES = [
    # (description, card_style, angle, center, area_frac, add_glare, extra_objects)
    {"desc": "DSC00032", "style": "light", "angle": 5, "center": (0.50, 0.48), "area": 0.30, "glare": False},
    {"desc": "DSC00033", "style": "dark",  "angle": -12, "center": (0.52, 0.50), "area": 0.28, "glare": True},
    {"desc": "DSC00034", "style": "light", "angle": 0,  "center": (0.48, 0.52), "area": 0.35, "glare": False},
    {"desc": "DSC00035", "style": "dark",  "angle": 18, "center": (0.55, 0.45), "area": 0.22, "glare": False},
    {"desc": "DSC00036", "style": "light", "angle": -8, "center": (0.45, 0.55), "area": 0.32, "glare": True},
    {"desc": "DSC00037", "style": "light", "angle": 22, "center": (0.50, 0.50), "area": 0.38, "glare": False},
    {"desc": "DSC00038", "style": "dark",  "angle": -3, "center": (0.47, 0.48), "area": 0.25, "glare": False},
    {"desc": "DSC00039", "style": "light", "angle": 15, "center": (0.60, 0.42), "area": 0.20, "glare": True},
    {"desc": "DSC00040", "style": "dark",  "angle": -20, "center": (0.42, 0.55), "area": 0.33, "glare": False},
    {"desc": "DSC00041", "style": "light", "angle": 10, "center": (0.50, 0.50), "area": 0.36, "glare": False},
]


def generate_scene(scene, rng, width=1200, height=900):
    """Generate one realistic scene image."""
    bg = wood_background(height, width, rng)
    card = card_patch(rng, style=scene["style"])

    img, quad = composite_card(
        bg, card, rng,
        center_frac=(scene["center"][0], scene["center"][1]),
        angle=scene["angle"],
        area_frac=scene["area"],
    )

    img = add_lighting(img, rng)

    if scene["glare"]:
        img = add_glare(img, rng)

    img = add_noise(img, rng, sigma=rng.uniform(3, 8))

    # Slight overall blur to simulate phone camera
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


def main():
    p = argparse.ArgumentParser(description="Generate realistic baseball card test images")
    p.add_argument("--output-dir", default="input", help="Output directory (default: input)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=1200)
    p.add_argument("--height", type=int, default=900)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    for scene in SCENES:
        img = generate_scene(scene, rng, width=args.width, height=args.height)
        path = out_dir / f"{scene['desc']}.JPG"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  {path}")

    print(f"\nGenerated {len(SCENES)} test images in {out_dir}/")


if __name__ == "__main__":
    main()
