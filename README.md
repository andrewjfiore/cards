# cards

## Synthetic self-test benchmark

You can benchmark detector robustness without real photos by generating synthetic wood-table scenes and scoring card localization IoU:

```bash
python synthetic_self_test.py --samples 300 --seed 7
```

Useful options:
- `--iou-threshold 0.70` success threshold for each sample
- `--min-pass-rate 0.85` required aggregate pass rate (non-zero exit code on failure)
- `--min-recrop-stability 0.90` minimum rate where re-cropping a crop is stable
- `--max-aspect-stdev 0.12` cross-card consistency gate for detected crop aspect ratio
- `--max-area-scale-stdev 0.30` cross-card consistency gate for predicted/GT area scale
- `--save-failures failures/` writes visual overlays with GT (green) and prediction (red)

The benchmark now includes cross-card checks beyond IoU: batch aspect-ratio variance, area-scale variance, and re-crop stability (detecting drift when `detect_card` is run again on an already-cropped card).

This is designed as a repeatable tuning loop: change `card_crop.py`, run the synthetic benchmark, inspect failure overlays, and iterate until quality gates pass.

## OCR-aware crop expansion (GPU-capable)

`card_crop.py` can optionally run OCR and expand card bounds so detected text is not clipped.

```bash
python card_crop.py --input-dir photos --output-dir output --ocr-refine
```

Options:
- `--ocr-min-conf` minimum OCR confidence to consider a text box
- `--ocr-max-dim` OCR resize cap for speed
- `--ocr-text-margin` extra border around OCR text boxes
- `--ocr-csv ocr_results.csv` save per-file OCR text as inline string in CSV (`filename`, `ocr_text`)

Implementation uses EasyOCR when installed and automatically enables GPU when `torch.cuda.is_available()` is true. If OCR dependencies are missing (or a card has no detectable text), cropping continues with geometric detection only.

## ML semantic re-ranking (GPU-capable)

You can optionally enable CLIP-based semantic scoring to bias detection toward card-like regions and away from wood-table false positives:

```bash
python card_crop.py --input-dir photos --output-dir output --ml-refine
```

Useful options:
- `--ml-model` HuggingFace CLIP model id (default: `openai/clip-vit-base-patch32`)
- `--ml-device` `auto|cpu|cuda`
- `--ml-weight` blending weight for ML semantic score in final candidate ranking

Notes:
- This is optional and falls back to geometry/content heuristics if `torch`/`transformers`/model weights are unavailable.
- When CUDA is available, CLIP inference runs on GPU.
