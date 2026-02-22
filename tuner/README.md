# Card Crop Tuner

Local webapp that helps you converge on the best `card_crop.py` parameter
configuration through multi-armed bandit experimentation.

## Quick Start

```powershell
# 1. From the repo root, install the main card_crop dependencies (if not already)
pip install -r requirements.txt

# 2. Install the tuner's own dependencies (still from repo root)
pip install -r tuner/requirements.txt

# 3. Move into the tuner directory and start the server
cd tuner
python server.py

# 4. Open in your browser
#    http://localhost:8377
```

> **Windows note:** Use `cd tuner` then `python server.py` as separate
> commands (PowerShell doesn't support `&&` in older versions).
> Browse to `http://localhost:8377` — not `0.0.0.0`.

## How It Works

1. **Setup** — Enter the absolute path to your folder of phone photos and
   (optionally) an output folder. The tuner scans for jpg/jpeg/png images.

2. **Start Batch** — The system picks a configuration via Thompson sampling
   from a pool of 27 seed configs spanning RT-DETR, YOLO, contour-only,
   OCR on/off, CLIP on/off, and various confidence/weight parameters.
   It then runs `card_crop.py` on 10 randomly-sampled images (with
   replacement) using that config.

3. **Review** — For each image you see Before/After side-by-side (plus
   debug overlay when available). Rate each image thumbs-up/down,
   uncertain, failure, or skip. Add confidence (sure/maybe/unsure) and
   defect tags for thumbs-down.

4. **Finalize** — Marks the batch done. The bandit updates arms online
   after each vote, so the next batch already reflects your feedback.

5. **Iterate** — After 5-15 batches the Leaderboard will clearly show
   which configs work best for your photo style.

## Pairwise Mode

Toggle "Pairwise comparison mode" to run two configs per batch and pick
which result is better for each image. Good for fine-tuning between
top contenders.

## Exploration Controls

- **Exploration slider** (0.1 – 5.0): Scales the Beta prior. Higher =
  more exploration (wider posterior samples).
- **Epsilon slider** (0 – 1.0): Fraction of batches that pick a config
  uniformly at random instead of Thompson sampling.

## Exports

- **Leaderboard JSON** — `/api/exports/leaderboard.json`
- **Votes CSV** — `/api/exports/votes.csv`

## Reset

"Reset Learning State" clears all arm parameters and votes but keeps the
config pool intact.
