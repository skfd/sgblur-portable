# Porting plan (draft)

No code yet. Ordered so the make-or-break question is answered first.

## Phase 0 — Feasibility gate — PASSED (2026-07-01)

- **Weights: available & permissive.** HF `Panoramax/detect_face_plate_sign`:
  `yolo11l_panoramax.pt` (51.6 MB, YOLOv11-l, imgsz 2048) + `yolov8s_panoramax.pt`
  (23 MB). License **etalab-2.0**. Standard Ultralytics `.pt`.
- Repo `github.com/cquest/sgblur` (MIT) has `models/ scripts/ src/ tests/`,
  `pyproject.toml`, Dockerfile. Weights NOT committed — pulled from HF.

Still to confirm in Phase 1 (cheap, do with code): exact device-selection point in
`src/`, and that the JPEG-MCU blur step is CPU-only (expected yes).

Gate result: **GO.**

## Phase 1 — Get detection running non-CUDA (smallest test)

1. Export weights to ONNX: `yolo export model=<weights>.pt format=onnx`.
2. Run inference on ONE 5760x2880 test image (from ../mapillary-export staging) via:
   - ONNX Runtime + **DirectML** provider on the AMD 890M (Windows), and/or
   - PyTorch **MPS** if a Mac is ever in play.
3. Verify detections match the CUDA baseline (same boxes on the same image).

## Phase 2 — Wire blur + measure

1. Feed detections into SGBlur's JPEG-MCU blur; confirm output visually correct on a
   360 equirect (watch pole/seam distortion — the custom model should handle it).
2. Benchmark throughput on the 890M -> extrapolate to 70k. Decide if the wall-clock is
   acceptable vs. just renting an Nvidia box.

## Phase 3 — Batch mode

1. Wrap as a local batch CLI: folder in -> blurred JPEGs out (skip already-done).
2. Run the full 70k as the private pre-processing step; upload blurred images to the
   Panoramax instance with server-side blurring DISABLED.

## Open questions

- Weights license/availability (Phase 0 gate).
- DirectML op coverage for YOLOv11 (some ops may fall back to CPU -> slower).
- Actual 890M throughput (only measurable in Phase 2).
- Do we even proceed, or default to OSM-FR endpoint / rented GPU? Decide after Phase 1.
