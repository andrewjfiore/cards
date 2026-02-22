"""
crop_runner.py — Invoke card_crop.py as a subprocess for a batch of images.

Builds CLI flags from a config dict, runs the cropper per image, and
captures status / strategy / output paths.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Resolve the card_crop.py script location (one directory up from this file).
CARD_CROP_SCRIPT = Path(__file__).resolve().parent.parent / "card_crop.py"


def config_to_cli_flags(cfg: dict) -> list[str]:
    """Convert a config dict to CLI flags for card_crop.py."""
    flags: list[str] = []

    detector = cfg.get("detector", "auto")
    flags += ["--detector", detector]

    detector_model = cfg.get("detector_model", "")
    if detector_model:
        flags += ["--detector-model", detector_model]

    flags += ["--detector-conf", str(cfg.get("detector_conf", 0.35))]

    if cfg.get("ocr_refine", True):
        flags.append("--ocr-refine")
    else:
        flags.append("--no-ocr-refine")

    flags += ["--ocr-min-conf", str(cfg.get("ocr_min_conf", 0.25))]
    flags += ["--ocr-max-dim", str(cfg.get("ocr_max_dim", 1800))]
    flags += ["--ocr-text-margin", str(cfg.get("ocr_text_margin", 0.03))]

    if cfg.get("ml_refine", True):
        flags.append("--ml-refine")
    else:
        flags.append("--no-ml-refine")

    ml_model = cfg.get("ml_model", "openai/clip-vit-base-patch32")
    flags += ["--ml-model", ml_model]
    flags += ["--ml-device", cfg.get("ml_device", "auto")]
    flags += ["--ml-weight", str(cfg.get("ml_weight", 0.40))]

    flags += ["--padding", str(cfg.get("padding", 0))]

    if cfg.get("no_resize", False):
        flags.append("--no-resize")

    if cfg.get("debug", False):
        flags.append("--debug")

    # Always disable interactive mode when running from tuner
    flags.append("--no-interactive")

    return flags


def run_single_image(
    image_path: str,
    output_dir: str,
    cfg: dict,
    timeout: int = 300,
) -> dict:
    """
    Run card_crop.py on a single image.

    Returns dict with keys:
      ok: bool, status: str, strategy: str, output_path: str, debug_path: str
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a temporary single-image input directory via a symlink trick:
    # we point --input-dir at the parent and filter by exact filename via --ext.
    # But card_crop processes all matching files, so instead we create a temp
    # dir with a symlink to the single image.
    import tempfile
    tmp_input = Path(tempfile.mkdtemp(prefix="tuner_in_"))
    link = tmp_input / image_path.name
    try:
        link.symlink_to(image_path.resolve())
    except OSError:
        # Symlink may fail on some systems; copy instead
        import shutil
        shutil.copy2(str(image_path), str(link))

    cropped_dir = output_dir / "cropped"
    debug_dir = output_dir / "debug"
    cropped_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    cli_flags = config_to_cli_flags(cfg)
    cmd = [
        sys.executable, str(CARD_CROP_SCRIPT),
        "--input-dir", str(tmp_input),
        "--output-dir", str(cropped_dir),
    ] + cli_flags

    result = {
        "ok": False,
        "status": "FAIL",
        "strategy": "",
        "output_path": "",
        "debug_path": "",
    }

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        combined = stdout + "\n" + stderr

        # Parse stdout for OK / FAIL status
        stem = image_path.stem
        out_file = cropped_dir / f"{stem}.jpg"

        if out_file.exists():
            result["ok"] = True
            result["status"] = "OK"
            result["output_path"] = str(out_file)

            # Try to extract strategy from stdout like "[OK] OK (rtdetr(book@0.87)+ocr)"
            m = re.search(r'\[OK\]\s*(.*)', combined)
            if m:
                result["strategy"] = m.group(1).strip()
        else:
            # Check for failure message
            m = re.search(r'\[FAIL\]\s*(.*)', combined)
            if m:
                result["status"] = f"FAIL: {m.group(1).strip()}"
            elif proc.returncode != 0:
                result["status"] = f"FAIL: exit code {proc.returncode}"
                if stderr.strip():
                    # Grab last meaningful line
                    lines = [l for l in stderr.strip().split('\n') if l.strip()]
                    if lines:
                        result["status"] += f" — {lines[-1][:200]}"

        # Check for debug overlay
        debug_file = debug_dir / f"debug_{stem}.jpg"
        # card_crop.py puts debug files in output_dir/debug/ when --debug is set
        # but we set --output-dir to cropped_dir, so debug goes to cropped_dir/debug/
        debug_sub = cropped_dir / "debug" / f"debug_{stem}.jpg"
        for df in [debug_file, debug_sub]:
            if df.exists():
                result["debug_path"] = str(df)
                break

        # Also check FAIL debug
        fail_debug = cropped_dir / "debug" / f"FAIL_{stem}.jpg"
        if not result["debug_path"] and fail_debug.exists():
            result["debug_path"] = str(fail_debug)

    except subprocess.TimeoutExpired:
        result["status"] = "FAIL: timeout"
    except Exception as e:
        result["status"] = f"FAIL: {str(e)[:200]}"
    finally:
        # Clean up temp input dir
        import shutil
        shutil.rmtree(tmp_input, ignore_errors=True)

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
    Returns list of result dicts.
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
