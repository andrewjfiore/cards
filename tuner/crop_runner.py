"""
crop_runner.py — Run card_crop in-process for the tuner.

Models (CLIP, RT-DETR/YOLO, EasyOCR) are loaded once on first use and
cached across all images / batches for the lifetime of the server process.
This avoids the enormous overhead of spawning a fresh Python subprocess
(and re-importing torch + re-loading weights) for every single image.
"""

import argparse
import concurrent.futures
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `import card_crop` works.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import card_crop  # noqa: E402  (must come after path manipulation)

# ── ML scorer cache ────────────────────────────────────────────────────────
# Kept here (rather than relying only on card_crop's own cache) so we can
# detect when the tuner config changes the model/device and reload.
_ml_scorer = None
_ml_scorer_key = None  # (model_id, device)


# ── Model warmup ──────────────────────────────────────────────────────────

def warmup_models(timeout: int = 90):
    """Pre-load detector models with a timeout so a stalled download
    doesn't hang the server forever.  Models that fail to load within
    *timeout* seconds are cached as None (unavailable) in card_crop's
    global cache so later calls skip them instantly.

    Call this once at server startup.
    """
    detectors = [
        ("rtdetr", card_crop._DEFAULT_MODELS["rtdetr"]),
        ("yolo", card_crop._DEFAULT_MODELS["yolo"]),
    ]

    for det_type, model_id in detectors:
        key = (det_type, model_id)
        if key in card_crop._DETECTOR_MODEL_CACHE:
            continue  # already loaded from a previous run

        print(f"[WARMUP] Loading {det_type} ({model_id}) …", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(card_crop._load_detector, det_type, model_id)
            try:
                model = fut.result(timeout=timeout)
                if model is not None:
                    print(f"[WARMUP] {det_type} ready.", flush=True)
                else:
                    print(f"[WARMUP] {det_type} unavailable (load returned None).", flush=True)
            except concurrent.futures.TimeoutError:
                print(f"[WARMUP] {det_type} download timed out after {timeout}s — "
                      "marking unavailable. Configs needing this detector "
                      "will fall back to contour.", flush=True)
                # Cache as None so card_crop never retries.
                card_crop._DETECTOR_MODEL_CACHE[key] = None


def _cfg_to_namespace(cfg: dict) -> argparse.Namespace:
    """Convert a tuner config dict into the Namespace card_crop expects."""
    return argparse.Namespace(
        detector=cfg.get("detector", "auto"),
        detector_model=cfg.get("detector_model", ""),
        detector_conf=cfg.get("detector_conf", 0.35),
        ocr_refine=cfg.get("ocr_refine", True),
        ocr_min_conf=cfg.get("ocr_min_conf", 0.25),
        ocr_max_dim=cfg.get("ocr_max_dim", 1800),
        ocr_text_margin=cfg.get("ocr_text_margin", 0.03),
        ocr_csv="",
        ml_refine=cfg.get("ml_refine", True),
        ml_model=cfg.get("ml_model", "openai/clip-vit-base-patch32"),
        ml_device=cfg.get("ml_device", "auto"),
        ml_weight=cfg.get("ml_weight", 0.40),
        ml_required=False,
        padding=cfg.get("padding", 0),
        no_resize=cfg.get("no_resize", False),
        debug=cfg.get("debug", False),
    )


def _ensure_ml_scorer(cfg: dict):
    """Load or reuse the CLIP scorer.  Returns the scorer callable or None."""
    global _ml_scorer, _ml_scorer_key

    if not cfg.get("ml_refine", True):
        return None

    model_id = cfg.get("ml_model", "openai/clip-vit-base-patch32")
    device = cfg.get("ml_device", "auto")
    key = (model_id, device)

    if key == _ml_scorer_key:
        return _ml_scorer

    _ml_scorer = card_crop._get_clip_card_scorer(model_id, device)
    _ml_scorer_key = key
    return _ml_scorer


def run_single_image(
    image_path: str,
    output_dir: str,
    cfg: dict,
    timeout: int = 300,  # kept for API compat; not used in-process
) -> dict:
    """
    Run card_crop.process_image() in-process on a single image.

    Returns dict with keys:
      ok, status, strategy, output_path, debug_path
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)

    cropped_dir = output_dir / "cropped"
    cropped_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = None
    if cfg.get("debug", False):
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

    dst = cropped_dir / f"{image_path.stem}.jpg"
    args = _cfg_to_namespace(cfg)
    ml_scorer = _ensure_ml_scorer(cfg)

    result = {
        "ok": False,
        "status": "FAIL",
        "strategy": "",
        "output_path": "",
        "debug_path": "",
    }

    try:
        success, msg, _ocr = card_crop.process_image(
            image_path, dst, args, debug_dir=debug_dir, ml_scorer=ml_scorer,
        )

        if success:
            result["ok"] = True
            result["status"] = "OK"
            result["output_path"] = str(dst)
            result["strategy"] = msg
        else:
            result["status"] = f"FAIL: {msg}"

        # Locate debug overlays
        for candidate in [
            debug_dir and (debug_dir / f"debug_{image_path.stem}.jpg"),
            debug_dir and (debug_dir / f"FAIL_{image_path.stem}.jpg"),
            cropped_dir / "debug" / f"debug_{image_path.stem}.jpg",
            cropped_dir / "debug" / f"FAIL_{image_path.stem}.jpg",
        ]:
            if candidate and candidate.exists():
                result["debug_path"] = str(candidate)
                break

    except Exception as e:
        result["status"] = f"FAIL: {str(e)[:200]}"

    return result


def run_batch(
    image_paths: list[str],
    output_root: str,
    config_id: int,
    cfg: dict,
    on_progress=None,
) -> list[dict]:
    """
    Run the cropper on a list of images with the given config.

    on_progress(index, total, result) is called after each image.
    """
    output_dir = Path(output_root) / f"config_{config_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, img_path in enumerate(image_paths):
        r = run_single_image(img_path, str(output_dir), cfg)
        results.append(r)
        if on_progress:
            on_progress(i, len(image_paths), r)

    return results
