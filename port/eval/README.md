# Detection eval harness

Tune the detection config (model x confidence x scale schedule) for the AMD
DirectML path by measuring what each config catches, instead of eyeballing one
image. Privacy-critical metric: **face + plate recall** (a missed face is the
failure mode).

## Why relative, not absolute

Certifying absolute ground truth on dense 360 street scenes needs human review
at full resolution. So the harness scores **recall relative to a generous
reference** (`yolo11l`, all scales, conf 0.15) — i.e. "what does a cheaper /
faster config lose vs the best-effort config?". Pruning false positives out of
the reference sharpens the numbers over time.

## Workflow

```
.venv/Scripts/python.exe port/eval/select_sample.py    # 1. pick ~14 diverse frames -> sample.txt
.venv/Scripts/python.exe port/eval/gen_candidates.py   # 2. over-detect -> annotated/ + candidates.json
# 3. review annotated/ at full res; prune false positives / note misses (see below)
.venv/Scripts/python.exe port/eval/score.py            # 4. sweep configs vs the reference
```

## Sharpening the reference (`candidates.json`)

`candidates.json` is the reference recall ceiling. Open the images in
`annotated/` at full resolution and, per frame:

- **False positive** (box on a non-face/non-plate): delete that entry from the
  frame's `dets` list in `candidates.json`.
- **Missed** face/plate the model never boxed: add `["face", 1.0, x1, y1, x2, y2]`
  (or `"plate"`) with approximate full-image pixel coords.

Re-run `score.py` — recalls now reflect the corrected reference. Only face/plate
matter for privacy; signs are reported for completeness.

## Files

- `select_sample.py` — evenly-spaced frame picker across staging sequences
- `gen_candidates.py` — generous detection -> annotated images + candidates JSON
- `score.py` — config sweep, recall-vs-reference + speed
- generated (git-ignored): `sample.txt`, `candidates.json`, `annotated/`, `score_out.txt`
