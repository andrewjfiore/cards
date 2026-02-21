"""Synthetic benchmark for card_crop.py.

Generates photo-like synthetic scenes (wood table + card + distractors),
runs detect_card, and scores localization IoU against ground truth.

Also performs cross-card consistency checks:
- Re-crop stability: re-running detect_card on a crop should not change it much.
- Aspect-ratio consistency: detected crops should have low stdev in long/short ratio.
- Area-scale consistency: predicted quad area vs GT quad area should be stable.

Usage:
  python synthetic_self_test.py --samples 300 --seed 7
  python synthetic_self_test.py --samples 500 --min-pass-rate 0.9 --save-failures failures
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from card_crop import detect_card, four_point_transform


@dataclass
class SampleMeta:
    glare: bool
    small: bool
    high_skew: bool


def parse_args():
    p = argparse.ArgumentParser(description="Synthetic self-test for card detector")
    p.add_argument("--samples", type=int, default=200, help="Number of synthetic scenes")
    p.add_argument("--seed", type=int, default=7, help="RNG seed")
    p.add_argument("--width", type=int, default=1600)
    p.add_argument("--height", type=int, default=1200)
    p.add_argument("--iou-threshold", type=float, default=0.7)
    p.add_argument("--min-pass-rate", type=float, default=0.85)
    p.add_argument("--min-recrop-stability", type=float, default=0.30,
                   help="Minimum fraction of successful detections with stable re-crops")
    p.add_argument("--max-recrop-change", type=float, default=0.30,
                   help="Allowed change (1-IoU) when re-cropping a crop")
    p.add_argument("--max-aspect-stdev", type=float, default=0.15,
                   help="Maximum stdev of detected crop long/short aspect ratio")
    p.add_argument("--max-area-scale-stdev", type=float, default=1.50,
                   help="Maximum stdev of predicted/GT area ratio")
    p.add_argument("--save-failures", type=str, default="", help="Optional directory to save failed cases")
    return p.parse_args()


def _wood_background(h, w, rng):
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]

    # Randomise the base wood tone for variety (light pine → dark walnut)
    r_base = rng.uniform(80, 160)
    g_base = r_base * rng.uniform(0.65, 0.85)
    b_base = r_base * rng.uniform(0.35, 0.60)

    base = np.zeros((h, w, 3), np.float32)
    base[..., 2] = r_base + rng.uniform(20, 50) * y + rng.uniform(5, 25) * x
    base[..., 1] = g_base + rng.uniform(15, 40) * y + rng.uniform(5, 18) * x
    base[..., 0] = b_base + rng.uniform(10, 28) * y + rng.uniform(3, 14) * x

    # Multiple grain streaks at varying angles for realism
    n_grains = rng.integers(3, 7)
    for _ in range(n_grains):
        phase = rng.uniform(0, 2 * np.pi)
        freq = rng.uniform(6.0, 20.0)
        angle = rng.uniform(-0.05, 0.05)
        grain = (np.sin((x * freq + y * (1.5 + angle * 10)) * np.pi + phase)
                 * rng.uniform(10, 22)).astype(np.float32)
        base += grain[..., None]

    # Low-frequency colour splotches (stain / knot simulation)
    for _ in range(rng.integers(0, 3)):
        cx = rng.uniform(0.1, 0.9)
        cy = rng.uniform(0.1, 0.9)
        sigma = rng.uniform(0.08, 0.25)
        blob = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
        base += (blob * rng.uniform(-30, 30))[..., None]

    noise = rng.normal(0, rng.uniform(4, 8), size=base.shape).astype(np.float32)
    base += noise

    return np.clip(base, 0, 255).astype(np.uint8)


def _card_patch(rng, card_h=700, card_w=500):
    # Randomly pick a card style: light-bordered (70%) or dark-bordered (30%)
    dark_style = bool(rng.random() < 0.30)

    if dark_style:
        bg_val = rng.integers(20, 60)
        card = np.full((card_h, card_w, 3), bg_val, np.uint8)
        border_color = tuple(int(v) for v in rng.integers(10, 70, size=3))
        inner_bg = tuple(int(v) for v in rng.integers(200, 250, size=3))
        text_color = tuple(int(v) for v in rng.integers(180, 255, size=3))
    else:
        card = np.full((card_h, card_w, 3), 240, np.uint8)
        border_color = tuple(int(v) for v in rng.integers(170, 255, size=3))
        inner_bg = (250, 250, 250)
        text_color = (30, 30, 180)

    cv2.rectangle(card, (8, 8), (card_w - 8, card_h - 8), border_color, thickness=10)
    cv2.rectangle(card, (24, 24), (card_w - 24, card_h - 24), inner_bg, thickness=-1)

    art_tl = (40, 120)
    art_br = (card_w - 40, int(card_h * 0.72))
    art = rng.integers(20, 240, size=(art_br[1] - art_tl[1], art_br[0] - art_tl[0], 3), dtype=np.uint8)
    art = cv2.GaussianBlur(art, (7, 7), 0)
    card[art_tl[1]:art_br[1], art_tl[0]:art_br[0]] = art

    name_bg = (230, 230, 230) if not dark_style else (50, 50, 50)
    cv2.rectangle(card, (35, 35), (card_w - 35, 95), name_bg, thickness=-1)
    stat_bg = (235, 235, 235) if not dark_style else (45, 45, 45)
    cv2.rectangle(card, (35, int(card_h * 0.74)), (card_w - 35, card_h - 35), stat_bg, thickness=-1)

    cv2.putText(card, "PLAYER", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, text_color, 3)
    for i in range(6):
        y = int(card_h * 0.78) + i * 28
        cv2.line(card, (45, y), (card_w - 45, y), (80, 80, 80), 2)

    return card


def _random_quad(h, w, rng):
    card_aspect = 1.4
    area_frac = float(rng.uniform(0.06, 0.35))
    card_area = h * w * area_frac
    short_side = np.sqrt(card_area / card_aspect)
    long_side = short_side * card_aspect

    rw = short_side
    rh = long_side
    if rng.random() < 0.2:
        rw, rh = rh, rw

    cx = rng.uniform(w * 0.18, w * 0.82)
    cy = rng.uniform(h * 0.18, h * 0.82)
    angle = rng.uniform(-65, 65)

    rect = ((cx, cy), (rw, rh), angle)
    quad = cv2.boxPoints(rect).astype(np.float32)

    strength = rng.uniform(0.0, 0.22)
    jitter = np.column_stack([
        rng.uniform(-w * strength * 0.12, w * strength * 0.12, size=4),
        rng.uniform(-h * strength * 0.12, h * strength * 0.12, size=4),
    ]).astype(np.float32)
    quad += jitter

    return np.clip(quad, [0, 0], [w - 1, h - 1])


def _composite_card(bg, quad, rng):
    h, w = bg.shape[:2]
    card = _card_patch(rng)
    ch, cw = card.shape[:2]

    src = np.array([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, quad.astype(np.float32))

    warped_card = cv2.warpPerspective(card, M, (w, h))
    mask = cv2.warpPerspective(np.full((ch, cw), 255, np.uint8), M, (w, h))

    out = bg.copy()
    fg = mask > 0
    out[fg] = warped_card[fg]

    # Vignette (simulates phone-camera lens falloff)
    if rng.random() < 0.60:
        Y, X = np.ogrid[:h, :w]
        cx_v = w / 2 + rng.uniform(-w * 0.08, w * 0.08)
        cy_v = h / 2 + rng.uniform(-h * 0.08, h * 0.08)
        dist = np.sqrt((X - cx_v) ** 2 + (Y - cy_v) ** 2).astype(np.float32)
        max_dist = np.sqrt(cx_v ** 2 + cy_v ** 2)
        vig = (1.0 - float(rng.uniform(0.15, 0.35)) * (dist / max_dist) ** 2)
        out = np.clip(out.astype(np.float32) * vig[..., None], 0, 255).astype(np.uint8)

    # Card shadow (offset dark duplicate)
    if rng.random() < 0.40:
        shadow = cv2.GaussianBlur(mask, (21, 21), 0)
        s_alpha = (shadow.astype(np.float32) / 255.0) * float(rng.uniform(0.10, 0.25))
        out = np.clip(
            out.astype(np.float32) * (1.0 - s_alpha[..., None]),
            0, 255
        ).astype(np.uint8)

    glare = bool(rng.random() < 0.35)
    if glare:
        gc = (int(rng.uniform(w * 0.2, w * 0.8)), int(rng.uniform(h * 0.2, h * 0.8)))
        axes = (int(rng.uniform(w * 0.08, w * 0.2)), int(rng.uniform(h * 0.05, h * 0.14)))
        overlay = out.copy()
        cv2.ellipse(overlay, gc, axes, rng.uniform(0, 180), 0, 360, (255, 255, 255), -1)
        alpha = float(rng.uniform(0.08, 0.22))
        out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)

    if rng.random() < 0.55:
        center = (int(rng.uniform(w * 0.75, w * 1.03)), int(rng.uniform(h * 0.45, h * 1.03)))
        radius = int(rng.uniform(min(h, w) * 0.15, min(h, w) * 0.3))
        cv2.circle(out, center, radius, (35, 35, 35), thickness=-1)

    return out, glare


def _iou_quad(pred_quad, gt_quad, h, w):
    pm = np.zeros((h, w), np.uint8)
    gm = np.zeros((h, w), np.uint8)
    cv2.fillPoly(pm, [np.int32(pred_quad.reshape(4, 2))], 255)
    cv2.fillPoly(gm, [np.int32(gt_quad.reshape(4, 2))], 255)
    inter = np.logical_and(pm > 0, gm > 0).sum()
    union = np.logical_or(pm > 0, gm > 0).sum()
    return 0.0 if union == 0 else float(inter / union)


def _quad_area(quad):
    return abs(cv2.contourArea(np.float32(quad).reshape(4, 1, 2)))


def _long_short_aspect_from_crop(crop):
    h, w = crop.shape[:2]
    if min(h, w) == 0:
        return 0.0
    return max(h, w) / min(h, w)


def _recrop_change_score(crop):
    """Return change score for crop->detect->crop cycle. Ideal is 0.0."""
    pred2, _ = detect_card(crop)
    if pred2 is None:
        return 1.0

    h, w = crop.shape[:2]
    full = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    iou = _iou_quad(np.float32(pred2).reshape(4, 2), full, h, w)
    return 1.0 - iou


def _generate_sample(width, height, rng):
    bg = _wood_background(height, width, rng)
    quad = _random_quad(height, width, rng)
    image, glare = _composite_card(bg, quad, rng)

    box = cv2.minAreaRect(quad.astype(np.float32))
    rw, rh = box[1]
    area_frac = (rw * rh) / (width * height)
    small = area_frac < 0.08

    sides = [
        np.linalg.norm(quad[0] - quad[1]),
        np.linalg.norm(quad[1] - quad[2]),
        np.linalg.norm(quad[2] - quad[3]),
        np.linalg.norm(quad[3] - quad[0]),
    ]
    high_skew = abs(sides[0] - sides[2]) > 40 or abs(sides[1] - sides[3]) > 40

    return image, quad, SampleMeta(glare=glare, small=small, high_skew=high_skew)


def _bucket_scores(records, field):
    yes = [r for r in records if getattr(r["meta"], field)]
    no = [r for r in records if not getattr(r["meta"], field)]

    def rate(arr):
        return 0.0 if not arr else sum(r["pass"] for r in arr) / len(arr)

    return rate(yes), len(yes), rate(no), len(no)


def _safe_stats(values):
    if not values:
        return 0.0, 0.0
    arr = np.array(values, dtype=np.float32)
    return float(arr.mean()), float(arr.std())


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    failure_dir = Path(args.save_failures) if args.save_failures else None
    if failure_dir:
        failure_dir.mkdir(parents=True, exist_ok=True)

    records = []
    aspect_values = []
    area_scale_values = []
    recrop_changes = []

    for i in range(args.samples):
        image, gt_quad, meta = _generate_sample(args.width, args.height, rng)
        pred_quad, strategy = detect_card(image)

        recrop_change = 1.0
        aspect_ratio = 0.0
        area_scale = 0.0

        if pred_quad is None:
            iou = 0.0
            passed = False
        else:
            pred4 = np.float32(pred_quad).reshape(4, 2)
            iou = _iou_quad(pred4, gt_quad, args.height, args.width)
            passed = iou >= args.iou_threshold

            pred_area = max(1e-6, _quad_area(pred4))
            gt_area = max(1e-6, _quad_area(gt_quad))
            area_scale = pred_area / gt_area
            area_scale_values.append(area_scale)

            crop = four_point_transform(image, np.float32(pred_quad))
            if crop is not None and crop.size > 0:
                aspect_ratio = _long_short_aspect_from_crop(crop)
                aspect_values.append(aspect_ratio)
                recrop_change = _recrop_change_score(crop)
                recrop_changes.append(recrop_change)

        records.append({
            "pass": passed,
            "iou": iou,
            "meta": meta,
            "strategy": strategy or "none",
            "recrop_change": recrop_change,
            "aspect": aspect_ratio,
            "area_scale": area_scale,
        })

        if failure_dir and not passed:
            vis = image.copy()
            cv2.polylines(vis, [np.int32(gt_quad)], True, (0, 255, 0), 4)
            if pred_quad is not None:
                cv2.polylines(vis, [np.int32(np.float32(pred_quad).reshape(4, 2))], True, (0, 0, 255), 4)
            cv2.putText(vis, f"iou={iou:.3f} recrop={recrop_change:.3f}", (25, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            cv2.imwrite(str(failure_dir / f"fail_{i:04d}.jpg"), vis)

    pass_rate = sum(r["pass"] for r in records) / len(records)
    mean_iou = sum(r["iou"] for r in records) / len(records)

    glare_yes, glare_n, glare_no, no_glare_n = _bucket_scores(records, "glare")
    small_yes, small_n, small_no, large_n = _bucket_scores(records, "small")
    skew_yes, skew_n, skew_no, low_skew_n = _bucket_scores(records, "high_skew")

    aspect_mean, aspect_stdev = _safe_stats(aspect_values)
    area_mean, area_stdev = _safe_stats(area_scale_values)
    recrop_mean, recrop_stdev = _safe_stats(recrop_changes)
    recrop_stable_rate = 0.0 if not recrop_changes else (
        sum(v <= args.max_recrop_change for v in recrop_changes) / len(recrop_changes)
    )

    print(f"samples={len(records)} iou_threshold={args.iou_threshold:.2f}")
    print(f"pass_rate={pass_rate:.3f} mean_iou={mean_iou:.3f}")
    print(f"glare    : {glare_yes:.3f} ({glare_n}) | no-glare : {glare_no:.3f} ({no_glare_n})")
    print(f"small    : {small_yes:.3f} ({small_n}) | larger   : {small_no:.3f} ({large_n})")
    print(f"high-skew: {skew_yes:.3f} ({skew_n}) | low-skew : {skew_no:.3f} ({low_skew_n})")
    print(f"aspect_ratio mean={aspect_mean:.3f} stdev={aspect_stdev:.3f} n={len(aspect_values)}")
    print(f"area_scale  mean={area_mean:.3f} stdev={area_stdev:.3f} n={len(area_scale_values)}")
    print(
        "recrop_change mean={:.3f} stdev={:.3f} stable_rate={:.3f} n={}".format(
            recrop_mean, recrop_stdev, recrop_stable_rate, len(recrop_changes)
        )
    )

    strat_counts = {}
    for r in records:
        strat_counts[r["strategy"]] = strat_counts.get(r["strategy"], 0) + 1
    top = sorted(strat_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    print("top_strategies=" + ", ".join(f"{k}:{v}" for k, v in top))

    failed = False
    if pass_rate < args.min_pass_rate:
        print(f"FAIL: pass_rate {pass_rate:.3f} < min_pass_rate {args.min_pass_rate:.3f}")
        failed = True
    if aspect_stdev > args.max_aspect_stdev:
        print(f"FAIL: aspect_stdev {aspect_stdev:.3f} > max_aspect_stdev {args.max_aspect_stdev:.3f}")
        failed = True
    if area_stdev > args.max_area_scale_stdev:
        print(f"FAIL: area_scale_stdev {area_stdev:.3f} > max_area_scale_stdev {args.max_area_scale_stdev:.3f}")
        failed = True
    if recrop_stable_rate < args.min_recrop_stability:
        print(
            f"FAIL: recrop_stable_rate {recrop_stable_rate:.3f} "
            f"< min_recrop_stability {args.min_recrop_stability:.3f}"
        )
        failed = True

    if failed:
        raise SystemExit(1)

    print("PASS")


if __name__ == "__main__":
    main()
