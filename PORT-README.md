# sgblur-portable

Port of Panoramax's **SGBlur** face/plate blurring off its Nvidia-CUDA-only
default so it runs on **AMD** (Radeon 890M iGPU, via ONNX Runtime + **DirectML**).
Goal: **private, local blurring** of a fixed archive before upload, on hardware
already owned — no GPU rental, no shared cloud endpoint, **no native binaries**.

Status: **working end-to-end.** Detection → lossless blur → resumable batch, all
on the iGPU, no CUDA. See `PORTING-PLAN.md` for phase history.

## Why this exists

Sibling to `mapillary-export` (../mapillary-export), which republishes ~70k
equirectangular 360 photos (5760×2880, ~16 MP) to a self-hosted Panoramax
instance + KartaView. Panoramax expects faces/plates blurred. Upstream SGBlur
GPU-accelerates only on Nvidia CUDA; this repo makes it run on non-Nvidia
hardware we own.

## What's here (`port/`)

| File | Role |
|---|---|
| `detect_dml.py` | Multi-scale (S 1024, L 2048, XL split-halves @4096 for 360s) YOLOv11 detection on DirectML |
| `blur_dct.py` | Pure-Python **lossless** face/plate blur via JPEG DCT-coefficient editing (`jpeglib`) |
| `blur_batch.py` | Resumable folder-in → folder-out batch runner |
| `eval/` | Recall-vs-speed tuning harness (see `eval/README.md`) |
| `detect_test.py`, `blur_test.py` | Early probes / validation (superseded by the above) |
| `requirements.txt` | Deps + the DirectML install gotchas |

## Usage

```sh
py -3.12 -m venv .venv                                      # 3.14 has no torch/ort-dml wheels
.venv/Scripts/python.exe -m pip install -r port/requirements.txt
# one-time: export weights to ONNX (see requirements.txt gotcha -- export re-installs
# vanilla onnxruntime and shadows DirectML; reinstall onnxruntime-directml after)
.venv/Scripts/python.exe -c "from ultralytics import YOLO; YOLO('models/yolo11l_panoramax.pt').export(format='onnx', dynamic=True, opset=17)"

.venv/Scripts/python.exe port/blur_batch.py <input_dir> <output_dir>
```

Output mirrors the input tree, preserves EXIF/GPS, and **skips already-blurred
files** so a multi-day run is stop/restart-safe.

## How it works — and two things that surprised us

1. **Ultralytics' ONNX loader never selects DirectML** (only CUDA/CoreML/CPU), so
   loading a `.onnx` through `YOLO()` silently runs on CPU. We drive
   `onnxruntime.InferenceSession(..., providers=['DmlExecutionProvider', ...])`
   directly and do letterbox + NMS ourselves.
2. **The blur step was *not* "already portable."** Upstream's MCU-lossless blur
   shells out to native `jpegtran`/`djpeg`/`cjpeg` + a turbojpeg DLL. Rather than
   vendor those on Windows, we reimplemented the blur **in the DCT domain in pure
   Python**: zero the AC coefficients of each 8×8 block a face/plate box touches
   (keep DC = block-average pixelation), leaving every other block bit-exact. No
   subprocess, no native deps. Verified: 0 pixels change outside the boxes; output
   is not bloated; EXIF/GPS preserved.

## Recall vs speed (14-frame eval, iGPU)

Best recall is the full config; every speedup costs real faces/plates — **no free
lunch** (fp16, smaller model, and dropping the 4096 passes were all tested).

| Config | face recall | plate recall | speed | 70k est. |
|---|---|---|---|---|
| **yolo11l, all scales, conf 0.15** | 1.00 (ref) | 1.00 (ref) | ~12–14 s | **~11 days** |
| yolo11l, all scales, conf 0.30 | 0.78 | 0.86 | ~12 s | ~10 days |
| yolo11s, all scales, conf 0.15 | 0.56 | 0.71 | ~6 s | ~5 days |
| yolo11l, no XL passes | 0.67 | 0.43 | ~1 s | ~21 h |

(Recall is relative to the generous reference; QA note: faces mAP50 is only ~0.657
upstream, so spot-check output before trusting a full run.)

## Reality check

For a **one-time** 70k batch, a rented Nvidia spot box (~$5, hours) or the free
OSM-FR endpoint beat ~11 days of iGPU grinding. This port earns its keep only if
you want **permanent, private, $0-marginal, repeatable** blurring you control —
which now exists and works.

## Upstream facts

- Source: `gitlab.com/panoramax/server/sgblur`, mirror `github.com/cquest/sgblur`. **MIT.**
- Weights: HF `Panoramax/detect_face_plate_sign` (`yolo11{n,s,m,l}_panoramax.pt`),
  license **etalab-2.0**. Standard Ultralytics `.pt` → ONNX.
- Two independent parts: **detection** (YOLOv11, the only GPU step) and **blur**
  (JPEG block manipulation, CPU).
