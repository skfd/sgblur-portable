# sgblur-portable

Port Panoramax's **SGBlur** face/plate blurring off its Nvidia-CUDA-only default so
it runs on **AMD** (Radeon 890M iGPU, via DirectML/ONNX) and/or **Apple Metal**
(Mac Mini, via PyTorch MPS). Goal: **private, local blurring** of a fixed archive
before upload, using hardware already owned — no GPU rental, no shared cloud endpoint.

Status: **planning only (text stubs, no code yet).**

## Why this exists

Sibling project to `mapillary-export` (../mapillary-export). That project republishes
~70k equirectangular 360 photos (5760x2880, ~16 MP each) to a self-hosted, public,
federated Panoramax instance + KartaView. Panoramax expects faces/plates blurred.
SGBlur is the tool, but it GPU-accelerates only on Nvidia CUDA. This repo explores
making it run on non-Nvidia hardware we already have.

## Upstream facts (verified 2026-07-01)

- Source: `gitlab.com/panoramax/server/sgblur` and mirror `github.com/cquest/sgblur`.
- **License: MIT** — free to fork, modify, redistribute.
- Two independent parts:
  1. **Detection** = Ultralytics **YOLOv11** (PyTorch), *custom-trained* model for
     faces + plates. This is the ONLY GPU-accelerated step.
  2. **Blur** = low-level **JPEG MCU manipulation** (no decompress/recompress). Pure
     CPU, already portable, nothing to change.
- So "porting" = getting the YOLOv11 detection onto a non-CUDA backend. The blur
  step runs anywhere as-is.

## Port targets

| Target | Backend | Notes |
|---|---|---|
| AMD Radeon 890M (Windows) | ONNX Runtime + **DirectML**, or `torch-directml` | Uses hardware we own. iGPU ~5-15x slower than desktop Nvidia (estimate), still >> CPU-only. Preferred path. |
| Apple Metal (Mac Mini) | PyTorch **MPS** (`device='mps'`) | ~one-line change, but requires buying a Mac; its GPU still trails a cheap Nvidia card. Not recommended as a purchase. |
| Universal | Export `.pt` -> **ONNX**, run via ONNX Runtime | Cleanest; provider picks CUDA/DirectML/CoreML/CPU. |

## GATING QUESTION: RESOLVED — port is FEASIBLE (2026-07-01)

Weights are **public and permissively licensed**. On Hugging Face,
`Panoramax/detect_face_plate_sign`:
- `yolo11l_panoramax.pt` (51.6 MB) — YOLOv11-large, trained at **imgsz 2048**, 300 epochs.
- `yolov8s_panoramax.pt` (23 MB) — lighter/faster fallback.
- License **etalab-2.0** (permissive open, reuse incl. commercial w/ attribution).
- Standard Ultralytics `.pt` -> exports to ONNX via `yolo export format=onnx`.

QA flags: faces mAP50 only **0.657** (plates/signs higher) -> some faces missed;
spot-check a blurred series before trusting it. Inference at **2048px** is heavy;
`yolov8s` is the speed lever on the 890M.

## Reality check

For a one-time 70k batch, the free OSM-FR blur endpoint (zero hardware) or a rented
Nvidia spot box (~$5, done in hours) both beat spending time on this port. This port
is only worth it if we want **permanent, private, $0-hardware** blurring we control.
See PORTING-PLAN.md.
