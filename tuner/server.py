"""
server.py — FastAPI backend for the card-crop tuner.

Endpoints:
  POST /api/datasets/select       – set input/output dirs
  GET  /api/configs               – list configs with arm stats
  POST /api/configs               – add a custom config
  POST /api/batches/start         – start a new batch (10 images)
  GET  /api/batches/{id}          – batch status + items
  POST /api/batches/{id}/process  – run cropper (SSE progress stream)
  POST /api/votes                 – submit a vote
  POST /api/batches/{id}/finalize – mark batch done
  GET  /api/exports/leaderboard.json
  GET  /api/exports/votes.csv
  POST /api/reset                 – reset learning state
  GET  /api/settings              – get exploration slider, etc.
  POST /api/settings              – set exploration slider, etc.
"""

import csv
import io
import json
import math
import random
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import db
import crop_runner

app = FastAPI(title="Card Crop Tuner")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Global state ─────────────────────────────────────────────────────────────

_state = {
    "input_dir": "",
    "output_root": "",
    "glob_filter": "*.jpg,*.jpeg,*.png,*.JPG,*.JPEG,*.PNG",
    "exploration": 1.0,   # 0..3  (multiplied into alpha0/beta0 priors)
    "epsilon": 0.0,       # epsilon-greedy mix  0..1
}


# ── Seed configs ─────────────────────────────────────────────────────────────

SEED_CONFIGS = [
    # --- RT-DETR variants ---
    {"detector": "rtdetr", "detector_model": "", "detector_conf": 0.25,
     "ocr_refine": True,  "ocr_min_conf": 0.25, "ocr_max_dim": 1800, "ocr_text_margin": 0.03,
     "ml_refine": True,   "ml_model": "openai/clip-vit-base-patch32", "ml_device": "auto", "ml_weight": 0.40,
     "padding": 0, "no_resize": False, "debug": True},

    {"detector": "rtdetr", "detector_model": "", "detector_conf": 0.35,
     "ocr_refine": True,  "ocr_min_conf": 0.25, "ocr_max_dim": 1800, "ocr_text_margin": 0.03,
     "ml_refine": True,   "ml_model": "openai/clip-vit-base-patch32", "ml_device": "auto", "ml_weight": 0.40,
     "padding": 0, "no_resize": False, "debug": True},

    {"detector": "rtdetr", "detector_model": "", "detector_conf": 0.45,
     "ocr_refine": True,  "ocr_min_conf": 0.25, "ocr_max_dim": 1800, "ocr_text_margin": 0.03,
     "ml_refine": True,   "ml_model": "openai/clip-vit-base-patch32", "ml_device": "auto", "ml_weight": 0.40,
     "padding": 0, "no_resize": False, "debug": True},

    {"detector": "rtdetr", "detector_model": "", "detector_conf": 0.35,
     "ocr_refine": False, "ocr_min_conf": 0.25, "ocr_max_dim": 1800, "ocr_text_margin": 0.03,
     "ml_refine": True,   "ml_model": "openai/clip-vit-base-patch32", "ml_device": "auto", "ml_weight": 0.40,
     "padding": 0, "no_resize": False, "debug": True},

    {"detector": "rtdetr", "detector_model": "", "detector_conf": 0.35,
     "ocr_refine": True,  "ocr_min_conf": 0.25, "ocr_max_dim": 1800, "ocr_text_margin": 0.03,
     "ml_refine": False,  "ml_model": "openai/clip-vit-base-patch32", "ml_device": "auto", "ml_weight": 0.40,
     "padding": 0, "no_resize": False, "debug": True},

    {"detector": "rtdetr", "detector_model": "", "detector_conf": 0.35,
     "ocr_refine": False, "ocr_min_conf": 0.25, "ocr_max_dim": 1800, "ocr_text_margin": 0.03,
     "ml_refine": False,  "ml_model": "openai/clip-vit-base-patch32", "ml_device": "auto", "ml_weight": 0.40,
     "padding": 0, "no_resize": False, "debug": True},

    {"detector": "rtdetr", "detector_model": "", "detector_conf": 0.30,
     "ocr_refine": True,  "ocr_min_conf": 0.20, "ocr_max_dim": 1800, "ocr_text_margin": 0.05,
     "ml_refine": True,   "ml_model": "openai/clip-vit-base-patch32", "ml_device": "auto", "ml_weight": 0.30,
     "padding": 5, "no_resize": False, "debug": True},

    {"detector": "rtdetr", "detector_model": "", "detector_conf": 0.35,
     "ocr_refine": True,  "ocr_min_conf": 0.30, "ocr_max_dim": 1800, "ocr_text_margin": 0.02,
     "ml_refine": True,   "ml_model": "openai/clip-vit-base-patch32", "ml_device": "auto", "ml_weight": 0.50,
     "padding": 10, "no_resize": False, "debug": True},

    # --- YOLO variants ---
    {"detector": "yolo", "detector_model": "", "detector_conf": 0.25,
     "ocr_refine": True,  "ocr_min_conf": 0.25, "ocr_max_dim": 1800, "ocr_text_margin": 0.03,
     "ml_refine": True,   "ml_model": "openai/clip-vit-base-patch32", "ml_device": "auto", "ml_weight": 0.40,
     "padding": 0, "no_resize": False, "debug": True},

    {"detector": "yolo", "detector_model": "", "detector_conf": 0.35,
     "ocr_refine": True,  "ocr_min_conf": 0.25, "ocr_max_dim": 1800, "ocr_text_margin": 0.03,
     "ml_refine": True,   "ml_model": "openai/clip-vit-base-patch32", "ml_device": "auto", "ml_weight": 0.40,
     "padding": 0, "no_resize": False, "debug": True},

    {"detector": "yolo", "detector_model": "", "detector_conf": 0.45,
     "ocr_refine": True,  "ocr_min_conf": 0.25, "ocr_max_dim": 1800, "ocr_text_margin": 0.03,
     "ml_refine": True,   "ml_model": "openai/clip-vit-base-patch32", "ml_device": "auto", "ml_weight": 0.40,
     "padding": 0, "no_resize": False, "debug": True},

    {"detector": "yolo", "detector_model": "", "detector_conf": 0.35,
     "ocr_refine": False, "ocr_min_conf": 0.25, "ocr_max_dim": 1800, "ocr_text_margin": 0.03,
     "ml_refine": True,   "ml_model": "openai/clip-vit-base-patch32", "ml_device": "auto", "ml_weight": 0.40,
     "padding": 0, "no_resize": False, "debug": True},

    {"detector": "yolo", "detector_model": "", "detector_conf": 0.35,
     "ocr_refine": True,  "ocr_min_conf": 0.25, "ocr_max_dim": 1800, "ocr_text_margin": 0.03,
     "ml_refine": False,  "ml_model": "openai/clip-vit-base-patch32", "ml_device": "auto", "ml_weight": 0.40,
     "padding": 0, "no_resize": False, "debug": True},

    {"detector": "yolo", "detector_model": "", "detector_conf": 0.35,
     "ocr_refine": False, "ocr_min_conf": 0.25, "ocr_max_dim": 1800, "ocr_text_margin": 0.03,
     "ml_refine": False,  "ml_model": "openai/clip-vit-base-patch32", "ml_device": "auto", "ml_weight": 0.40,
     "padding": 0, "no_resize": False, "debug": True},

    {"detector": "yolo", "detector_model": "", "detector_conf": 0.30,
     "ocr_refine": True,  "ocr_min_conf": 0.20, "ocr_max_dim": 1800, "ocr_text_margin": 0.05,
     "ml_refine": True,   "ml_model": "openai/clip-vit-base-patch32", "ml_device": "auto", "ml_weight": 0.30,
     "padding": 5, "no_resize": False, "debug": True},

    # --- Auto detector variants ---
    {"detector": "auto", "detector_model": "", "detector_conf": 0.25,
     "ocr_refine": True,  "ocr_min_conf": 0.25, "ocr_max_dim": 1800, "ocr_text_margin": 0.03,
     "ml_refine": True,   "ml_model": "openai/clip-vit-base-patch32", "ml_device": "auto", "ml_weight": 0.40,
     "padding": 0, "no_resize": False, "debug": True},

    {"detector": "auto", "detector_model": "", "detector_conf": 0.35,
     "ocr_refine": True,  "ocr_min_conf": 0.25, "ocr_max_dim": 1800, "ocr_text_margin": 0.03,
     "ml_refine": True,   "ml_model": "openai/clip-vit-base-patch32", "ml_device": "auto", "ml_weight": 0.40,
     "padding": 0, "no_resize": False, "debug": True},

    {"detector": "auto", "detector_model": "", "detector_conf": 0.35,
     "ocr_refine": True,  "ocr_min_conf": 0.25, "ocr_max_dim": 1800, "ocr_text_margin": 0.03,
     "ml_refine": False,  "ml_model": "openai/clip-vit-base-patch32", "ml_device": "auto", "ml_weight": 0.40,
     "padding": 0, "no_resize": False, "debug": True},

    {"detector": "auto", "detector_model": "", "detector_conf": 0.35,
     "ocr_refine": False, "ocr_min_conf": 0.25, "ocr_max_dim": 1800, "ocr_text_margin": 0.03,
     "ml_refine": True,   "ml_model": "openai/clip-vit-base-patch32", "ml_device": "auto", "ml_weight": 0.40,
     "padding": 0, "no_resize": False, "debug": True},

    # --- Contour-only variants ---
    {"detector": "contour", "detector_model": "", "detector_conf": 0.35,
     "ocr_refine": True,  "ocr_min_conf": 0.25, "ocr_max_dim": 1800, "ocr_text_margin": 0.03,
     "ml_refine": True,   "ml_model": "openai/clip-vit-base-patch32", "ml_device": "auto", "ml_weight": 0.40,
     "padding": 0, "no_resize": False, "debug": True},

    {"detector": "contour", "detector_model": "", "detector_conf": 0.35,
     "ocr_refine": True,  "ocr_min_conf": 0.25, "ocr_max_dim": 1800, "ocr_text_margin": 0.03,
     "ml_refine": True,   "ml_model": "openai/clip-vit-base-patch32", "ml_device": "auto", "ml_weight": 0.60,
     "padding": 0, "no_resize": False, "debug": True},

    {"detector": "contour", "detector_model": "", "detector_conf": 0.35,
     "ocr_refine": False, "ocr_min_conf": 0.25, "ocr_max_dim": 1800, "ocr_text_margin": 0.03,
     "ml_refine": True,   "ml_model": "openai/clip-vit-base-patch32", "ml_device": "auto", "ml_weight": 0.40,
     "padding": 0, "no_resize": False, "debug": True},

    {"detector": "contour", "detector_model": "", "detector_conf": 0.35,
     "ocr_refine": True,  "ocr_min_conf": 0.25, "ocr_max_dim": 1800, "ocr_text_margin": 0.03,
     "ml_refine": False,  "ml_model": "openai/clip-vit-base-patch32", "ml_device": "auto", "ml_weight": 0.40,
     "padding": 0, "no_resize": False, "debug": True},

    {"detector": "contour", "detector_model": "", "detector_conf": 0.35,
     "ocr_refine": False, "ocr_min_conf": 0.25, "ocr_max_dim": 1800, "ocr_text_margin": 0.03,
     "ml_refine": False,  "ml_model": "openai/clip-vit-base-patch32", "ml_device": "auto", "ml_weight": 0.40,
     "padding": 0, "no_resize": False, "debug": True},

    # --- Padding / no-resize variants ---
    {"detector": "auto", "detector_model": "", "detector_conf": 0.35,
     "ocr_refine": True,  "ocr_min_conf": 0.25, "ocr_max_dim": 1800, "ocr_text_margin": 0.03,
     "ml_refine": True,   "ml_model": "openai/clip-vit-base-patch32", "ml_device": "auto", "ml_weight": 0.40,
     "padding": 10, "no_resize": False, "debug": True},

    {"detector": "auto", "detector_model": "", "detector_conf": 0.35,
     "ocr_refine": True,  "ocr_min_conf": 0.25, "ocr_max_dim": 1800, "ocr_text_margin": 0.03,
     "ml_refine": True,   "ml_model": "openai/clip-vit-base-patch32", "ml_device": "auto", "ml_weight": 0.40,
     "padding": 0, "no_resize": True, "debug": True},
]


def seed_configs():
    """Insert seed configurations if none exist."""
    existing = db.get_configs()
    if existing:
        return
    for cfg in SEED_CONFIGS:
        if not db.config_exists(cfg):
            db.insert_config(cfg)


# ── Thompson sampling ────────────────────────────────────────────────────────

def thompson_select(exclude_config_id: int = None) -> int:
    """
    Select a config_id using Thompson sampling from Beta(alpha, beta).

    Diversity: don't repeat the last config unless it's 5% better.
    Exploration slider multiplies the prior (alpha0, beta0).
    Epsilon-greedy mixes in uniform random with probability epsilon.
    """
    arms = db.get_all_arms()
    if not arms:
        raise ValueError("No arms available")

    exploration = _state.get("exploration", 1.0)
    epsilon = _state.get("epsilon", 0.0)

    # Epsilon-greedy: with probability epsilon, pick uniformly at random
    if epsilon > 0 and random.random() < epsilon:
        return random.choice(arms)["config_id"]

    # Thompson sample from Beta(alpha * exploration, beta * exploration)
    # (higher exploration = more prior weight = more exploration)
    samples = []
    for arm in arms:
        a = arm["alpha"] * exploration
        b = arm["beta"] * exploration
        # Clamp to avoid degenerate distributions
        a = max(a, 0.01)
        b = max(b, 0.01)
        sample = random.betavariate(a, b)
        samples.append((sample, arm["config_id"], arm["alpha"], arm["beta"]))

    samples.sort(key=lambda x: x[0], reverse=True)

    best_sample, best_id, best_a, best_b = samples[0]

    # Diversity constraint: don't repeat unless best by 5% margin
    if exclude_config_id is not None and best_id == exclude_config_id and len(samples) > 1:
        second_sample = samples[1][0]
        best_mean = best_a / (best_a + best_b)
        second_a, second_b = samples[1][2], samples[1][3]
        second_mean = second_a / (second_a + second_b)

        margin = best_mean - second_mean
        if margin < 0.05:
            # Pick the second-best instead
            return samples[1][1]

    return best_id


def compute_vote_deltas(vote: str, confidence: str) -> tuple[float, float]:
    """Return (d_alpha, d_beta) for a vote + confidence."""
    conf_weight = {"sure": 1.0, "maybe": 0.5, "unsure": 0.25}.get(confidence, 1.0)

    if vote == "up":
        return (1.0 * conf_weight, 0.0)
    elif vote == "down":
        return (0.0, 1.0 * conf_weight)
    elif vote == "uncertain":
        return (0.25 * conf_weight, 0.25 * conf_weight)
    elif vote == "failure":
        return (0.0, 2.0 * conf_weight)
    elif vote == "skip":
        return (0.0, 0.0)
    else:
        return (0.0, 0.0)


# ── Image listing ────────────────────────────────────────────────────────────

def list_images(input_dir: str, glob_filter: str = "") -> list[str]:
    """List image files in input_dir matching the glob filter."""
    d = Path(input_dir)
    if not d.is_dir():
        return []

    patterns = [p.strip() for p in (glob_filter or "*.jpg,*.jpeg,*.png").split(",")]
    files = set()
    for pat in patterns:
        files.update(d.glob(pat))
    return sorted(str(f) for f in files if f.is_file())


def sample_images(input_dir: str, n: int = 10, glob_filter: str = "") -> list[str]:
    """Sample n images with replacement."""
    all_imgs = list_images(input_dir, glob_filter)
    if not all_imgs:
        return []
    return [random.choice(all_imgs) for _ in range(n)]


# ── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    db.init_db()
    seed_configs()


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/api/datasets/select")
async def select_dataset(req: Request):
    body = await req.json()
    input_dir = body.get("input_dir", "").strip()
    output_root = body.get("output_root", "").strip()
    glob_filter = body.get("glob_filter", "").strip()

    if not input_dir:
        raise HTTPException(400, "input_dir is required")
    if not Path(input_dir).is_dir():
        raise HTTPException(400, f"input_dir does not exist: {input_dir}")

    _state["input_dir"] = input_dir
    if output_root:
        _state["output_root"] = output_root
    else:
        _state["output_root"] = str(Path(input_dir).parent / "tuner_output")
    if glob_filter:
        _state["glob_filter"] = glob_filter

    imgs = list_images(_state["input_dir"], _state["glob_filter"])
    return {"ok": True, "input_dir": _state["input_dir"],
            "output_root": _state["output_root"],
            "image_count": len(imgs)}


@app.get("/api/datasets/info")
async def dataset_info():
    if not _state["input_dir"]:
        return {"configured": False}
    imgs = list_images(_state["input_dir"], _state["glob_filter"])
    return {
        "configured": True,
        "input_dir": _state["input_dir"],
        "output_root": _state["output_root"],
        "glob_filter": _state["glob_filter"],
        "image_count": len(imgs),
    }


@app.get("/api/configs")
async def get_configs():
    configs = db.get_configs()
    vote_stats = {s["config_id"]: s for s in db.get_vote_stats()}

    result = []
    for c in configs:
        cfg = json.loads(c["json_cfg"])
        cid = c["id"]
        stats = vote_stats.get(cid, {})
        alpha = c["alpha"]
        beta = c["beta"]
        mean = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5

        result.append({
            "id": cid,
            "config": cfg,
            "alpha": alpha,
            "beta": beta,
            "mean": round(mean, 4),
            "total_votes": stats.get("total_votes", 0),
            "ups": stats.get("ups", 0),
            "downs": stats.get("downs", 0),
            "failures": stats.get("failures", 0),
            "uncertains": stats.get("uncertains", 0),
            "skips": stats.get("skips", 0),
            "last_tested_at": c["last_tested_at"],
            "created_at": c["created_at"],
        })

    result.sort(key=lambda x: x["mean"], reverse=True)
    return result


@app.post("/api/configs")
async def add_config(req: Request):
    body = await req.json()
    cfg = body.get("config")
    if not cfg:
        raise HTTPException(400, "config is required")
    if db.config_exists(cfg):
        raise HTTPException(409, "config already exists")
    cid = db.insert_config(cfg)
    return {"ok": True, "config_id": cid}


@app.post("/api/batches/start")
async def start_batch(req: Request):
    body = await req.json()
    mode = body.get("mode", "single")
    force_config = body.get("config_id")

    if not _state["input_dir"]:
        raise HTTPException(400, "No dataset selected — call /api/datasets/select first")

    last_config = db.get_last_batch_config_id()

    if force_config:
        config_id = force_config
    else:
        config_id = thompson_select(exclude_config_id=last_config)

    config_id_b = None
    if mode == "pairwise":
        config_id_b = thompson_select(exclude_config_id=config_id)

    images = sample_images(_state["input_dir"], 10, _state["glob_filter"])
    if not images:
        raise HTTPException(400, "No images found in input directory")

    batch_num = len(db.get_batches()) + 1
    output_root = str(Path(_state["output_root"]) / f"batch_{batch_num:04d}")

    batch_id = db.create_batch(
        config_id, _state["input_dir"], output_root,
        config_id_b=config_id_b, mode=mode,
    )

    cfg = json.loads(db.get_config(config_id)["json_cfg"])
    cfg_b = None
    if config_id_b:
        cfg_b = json.loads(db.get_config(config_id_b)["json_cfg"])

    # Create batch items
    items = []
    for img_path in images:
        item_id = db.add_batch_item(batch_id, img_path)
        items.append({"id": item_id, "filename": img_path})

    return {
        "batch_id": batch_id,
        "config_id": config_id,
        "config": cfg,
        "config_id_b": config_id_b,
        "config_b": cfg_b,
        "mode": mode,
        "images": items,
        "output_root": output_root,
    }


@app.post("/api/batches/{batch_id}/process")
async def process_batch(batch_id: int):
    """Run the cropper on all images in the batch. Returns SSE stream."""
    batch = db.get_batch(batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")

    items = db.get_batch_items(batch_id)
    config_id = batch["config_id"]
    config_id_b = batch["config_id_b"]
    cfg = json.loads(db.get_config(config_id)["json_cfg"])
    cfg_b = None
    if config_id_b:
        cfg_b = json.loads(db.get_config(config_id_b)["json_cfg"])

    async def generate():
        for i, item in enumerate(items):
            img_path = item["filename"]
            output_root = batch["output_root"]

            # Run config A
            result_a = crop_runner.run_single_image(
                img_path,
                str(Path(output_root) / f"config_{config_id}"),
                cfg,
            )

            update_a = {
                "output_path": result_a["output_path"],
                "debug_path": result_a["debug_path"],
                "status": result_a["status"],
                "strategy": result_a["strategy"],
            }

            # Run config B if pairwise
            result_b = None
            if cfg_b and config_id_b:
                result_b = crop_runner.run_single_image(
                    img_path,
                    str(Path(output_root) / f"config_{config_id_b}"),
                    cfg_b,
                )
                update_a["output_path_b"] = result_b["output_path"]
                update_a["debug_path_b"] = result_b["debug_path"]
                update_a["status_b"] = result_b["status"]
                update_a["strategy_b"] = result_b["strategy"]

            db.update_batch_item(item["id"], **update_a)

            event_data = {
                "index": i,
                "total": len(items),
                "item_id": item["id"],
                "filename": Path(img_path).name,
                "result_a": result_a,
            }
            if result_b:
                event_data["result_b"] = result_b

            yield f"data: {json.dumps(event_data)}\n\n"

        yield "data: {\"done\": true}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/batches/{batch_id}")
async def get_batch(batch_id: int):
    batch = db.get_batch(batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")

    items = db.get_batch_items(batch_id)
    votes = db.get_votes_for_batch(batch_id)
    vote_map = {}
    for v in votes:
        vote_map.setdefault(v["batch_item_id"], []).append(dict(v))

    cfg = json.loads(db.get_config(batch["config_id"])["json_cfg"])
    cfg_b = None
    if batch["config_id_b"]:
        cfg_b = json.loads(db.get_config(batch["config_id_b"])["json_cfg"])

    enriched_items = []
    for item in items:
        d = dict(item)
        d["votes"] = vote_map.get(item["id"], [])
        enriched_items.append(d)

    return {
        **dict(batch),
        "config": cfg,
        "config_b": cfg_b,
        "items": enriched_items,
    }


@app.get("/api/batches")
async def get_batches():
    batches = db.get_batches()
    result = []
    for b in batches:
        cfg = json.loads(db.get_config(b["config_id"])["json_cfg"])
        items = db.get_batch_items(b["id"])
        votes = db.get_votes_for_batch(b["id"])

        ups = sum(1 for v in votes if v["vote"] == "up")
        downs = sum(1 for v in votes if v["vote"] == "down")
        failures = sum(1 for v in votes if v["vote"] == "failure")
        total = len(votes)

        result.append({
            **dict(b),
            "config": cfg,
            "image_count": len(items),
            "vote_summary": {
                "total": total, "ups": ups, "downs": downs, "failures": failures,
            },
        })
    return result


@app.post("/api/votes")
async def submit_vote(req: Request):
    body = await req.json()
    batch_item_id = body.get("batch_item_id")
    vote = body.get("vote")  # up, down, uncertain, failure, skip
    confidence = body.get("confidence", "sure")
    reason_tags = body.get("reason_tags", [])
    pairwise_winner = body.get("pairwise_winner")  # "a", "b", or None

    if not batch_item_id or not vote:
        raise HTTPException(400, "batch_item_id and vote are required")

    if vote not in ("up", "down", "uncertain", "failure", "skip"):
        raise HTTPException(400, f"Invalid vote: {vote}")

    # Get the batch for this item to find config_id
    items = db._conn().execute(
        "SELECT bi.*, b.config_id, b.config_id_b FROM batch_items bi "
        "JOIN batches b ON bi.batch_id = b.id WHERE bi.id = ?",
        (batch_item_id,),
    ).fetchone()
    if not items:
        raise HTTPException(404, "Batch item not found")

    config_id = items["config_id"]
    config_id_b = items["config_id_b"]

    vote_id = db.add_vote(batch_item_id, vote, confidence, reason_tags, pairwise_winner)

    # Online bandit update
    if pairwise_winner and config_id_b:
        # Pairwise mode: winner gets alpha, loser gets beta
        conf_weight = {"sure": 1.0, "maybe": 0.5, "unsure": 0.25}.get(confidence, 1.0)
        if pairwise_winner == "a":
            db.update_arm(config_id, 1.0 * conf_weight, 0.0)
            db.update_arm(config_id_b, 0.0, 1.0 * conf_weight)
        elif pairwise_winner == "b":
            db.update_arm(config_id_b, 1.0 * conf_weight, 0.0)
            db.update_arm(config_id, 0.0, 1.0 * conf_weight)
        elif pairwise_winner == "tie":
            db.update_arm(config_id, 0.25 * conf_weight, 0.25 * conf_weight)
            db.update_arm(config_id_b, 0.25 * conf_weight, 0.25 * conf_weight)
    else:
        # Single mode
        d_alpha, d_beta = compute_vote_deltas(vote, confidence)
        db.update_arm(config_id, d_alpha, d_beta)

    return {"ok": True, "vote_id": vote_id}


@app.post("/api/batches/{batch_id}/finalize")
async def finalize_batch(batch_id: int):
    batch = db.get_batch(batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")

    db.finish_batch(batch_id)

    votes = db.get_votes_for_batch(batch_id)
    ups = sum(1 for v in votes if v["vote"] == "up")
    downs = sum(1 for v in votes if v["vote"] == "down")
    failures = sum(1 for v in votes if v["vote"] == "failure")
    uncertains = sum(1 for v in votes if v["vote"] == "uncertain")

    return {
        "ok": True,
        "batch_id": batch_id,
        "summary": {
            "total_votes": len(votes),
            "ups": ups, "downs": downs,
            "failures": failures, "uncertains": uncertains,
        },
    }


@app.get("/api/exports/leaderboard.json")
async def export_leaderboard():
    configs = db.get_configs()
    vote_stats = {s["config_id"]: s for s in db.get_vote_stats()}

    result = []
    for c in configs:
        cfg = json.loads(c["json_cfg"])
        cid = c["id"]
        stats = vote_stats.get(cid, {})
        alpha = c["alpha"]
        beta = c["beta"]
        mean = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
        total = stats.get("total_votes", 0)
        failures = stats.get("failures", 0)
        failure_rate = failures / total if total > 0 else 0.0

        result.append({
            "config_id": cid,
            "config": cfg,
            "alpha": alpha,
            "beta": beta,
            "posterior_mean": round(mean, 4),
            "total_votes": total,
            "failure_rate": round(failure_rate, 4),
            "last_tested_at": c["last_tested_at"],
        })

    result.sort(key=lambda x: x["posterior_mean"], reverse=True)
    return JSONResponse(result)


@app.get("/api/exports/votes.csv")
async def export_votes_csv():
    votes = db.get_all_votes()

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "vote_id", "batch_item_id", "batch_id", "filename",
        "config_id", "vote", "confidence", "reason_tags",
        "pairwise_winner", "created_at",
    ])
    writer.writeheader()
    for v in votes:
        writer.writerow({
            "vote_id": v["id"],
            "batch_item_id": v["batch_item_id"],
            "batch_id": v["batch_id"],
            "filename": v["filename"],
            "config_id": v["config_id"],
            "vote": v["vote"],
            "confidence": v["confidence"],
            "reason_tags": v["reason_tags"],
            "pairwise_winner": v.get("pairwise_winner", ""),
            "created_at": v["created_at"],
        })

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=votes.csv"},
    )


@app.post("/api/reset")
async def reset_learning():
    db.reset_arms()
    return {"ok": True, "message": "Arms and votes reset. Configs preserved."}


@app.get("/api/settings")
async def get_settings():
    return {
        "exploration": _state["exploration"],
        "epsilon": _state["epsilon"],
    }


@app.post("/api/settings")
async def update_settings(req: Request):
    body = await req.json()
    if "exploration" in body:
        _state["exploration"] = max(0.1, min(5.0, float(body["exploration"])))
    if "epsilon" in body:
        _state["epsilon"] = max(0.0, min(1.0, float(body["epsilon"])))
    return {"ok": True, **{k: _state[k] for k in ("exploration", "epsilon")}}


# ── Serve images from output dirs ───────────────────────────────────────────

@app.get("/api/image")
async def serve_image(path: str):
    """Serve any image file by absolute path (for before/after display)."""
    from fastapi.responses import FileResponse
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "Image not found")
    media_types = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".bmp": "image/bmp",
        ".tiff": "image/tiff", ".tif": "image/tiff",
    }
    mt = media_types.get(p.suffix.lower(), "image/jpeg")
    return FileResponse(str(p), media_type=mt)


# ── Static files (must be last) ─────────────────────────────────────────────

static_dir = Path(__file__).resolve().parent / "static"
if static_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8377, reload=True)
