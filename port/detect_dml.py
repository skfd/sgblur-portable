"""Multi-scale + 360-split detection on the AMD 890M via ONNX Runtime + DirectML.

Ports the detection *quality* of upstream src/detect/detect.py onto the DirectML
path. A single 2048px pass under-detects on 5760px equirect 360 images (small
faces/plates vanish when downsampled ~2.8x), so detect.py runs several passes and
merges them. This reproduces that:

    S  : full image @ 1024   (large close-up objects)
    L  : full image @ 2048   (standard)
    XL : for 360s, left/right halves cropped to the middle horizontal band,
         each @ min(width rounded to /32, 4096)   (small distant objects)

Detections from all passes are mapped back to full-image coords and merged with
IoU-based dedup (highest-resolution pass wins), matching detect.py's overlap
removal. Requires the dynamic-axes ONNX export (handles 1024/2048/4096).

Deferred to Phase 2 (blur wiring): MCU-aligned crop rects; feeding the JPEG-MCU
blur step. This module only proves detection quality on the iGPU.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from detect_test import letterbox, postprocess, NAMES, COLORS, MODEL, DEFAULT_IMG


def iou(a, b):
    """IoU of two xyxy boxes."""
    ax1 = max(a[0], b[0]); ay1 = max(a[1], b[1])
    ax2 = min(a[2], b[2]); ay2 = min(a[3], b[3])
    if ax2 <= ax1 or ay2 <= ay1:
        return 0.0
    inter = (ax2 - ax1) * (ay2 - ay1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def make_session(model_path):
    return ort.InferenceSession(
        str(model_path), providers=["DmlExecutionProvider", "CPUExecutionProvider"]
    )


def infer(session, bgr_crop, imgsz, offset):
    """Run one pass on a (crop of an) image at a given square size. Returns
    detections as (name, conf, x1, y1, x2, y2) in FULL-image coords via offset."""
    padded, r, left, top = letterbox(bgr_crop, new_shape=imgsz)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    x = np.ascontiguousarray((rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])
    name = session.get_inputs()[0].name
    out = session.run(None, {name: x})[0]
    dets = postprocess(out, r, left, top, bgr_crop.shape)
    ox, oy = offset
    return [(n, c, x1 + ox, y1 + oy, x2 + ox, y2 + oy) for (n, c, x1, y1, x2, y2) in dets]


def detect(session, bgr):
    """Multi-scale detection. Returns list of (label, dt_ms, detections) per pass."""
    h, w = bgr.shape[:2]
    passes = []

    for label, sz in (("S 1024", 1024), ("L 2048", 2048)):
        t0 = time.perf_counter()
        dets = infer(session, bgr, sz, (0, 0))
        passes.append((label, (time.perf_counter() - t0) * 1000, dets))

    # 360 panoramic: extra high-res pass on middle band of each half
    if w >= 5760 and w >= h * 2:
        split = w // 2
        ho = h // 4
        xl = min((w >> 5) << 5, 4096)
        crops = [("XL-L", bgr[ho:h * 3 // 4, 0:split], (0, ho)),
                 ("XL-R", bgr[ho:h * 3 // 4, split:w], (split, ho))]
        for name, crop, off in crops:
            t0 = time.perf_counter()
            dets = infer(session, crop, xl, off)
            passes.append((f"{name} {xl}", (time.perf_counter() - t0) * 1000, dets))

    return passes


def merge(passes, iou_thr=0.33):
    """Dedup across passes; process highest-res (last) first so it wins ties."""
    kept = []
    for _, _, dets in reversed(passes):
        for d in dets:
            if not any(iou(d[2:6], k[2:6]) > iou_thr for k in kept):
                kept.append(d)
    return kept


def annotate(bgr, dets, out_path):
    vis = bgr.copy()
    for name, conf, x1, y1, x2, y2 in dets:
        cv2.rectangle(vis, (x1, y1), (x2, y2), COLORS[name], 3)
        cv2.putText(vis, f"{name} {conf:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS[name], 2)
    cv2.imwrite(str(out_path), vis)


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG
    model_path = Path(sys.argv[2]) if len(sys.argv) > 2 else MODEL
    bgr = cv2.imread(img_path)
    if bgr is None:
        sys.exit(f"Could not read image: {img_path}")
    print(f"image: {img_path}  {bgr.shape[1]}x{bgr.shape[0]}")
    print(f"model: {model_path}\n")

    session = make_session(model_path)
    print(f"provider: {session.get_providers()[0]}\n")

    detect(session, bgr)  # warmup (dynamic shapes compile on first use per size)
    t0 = time.perf_counter()
    passes = detect(session, bgr)
    total = (time.perf_counter() - t0) * 1000

    for label, dt, dets in passes:
        counts = {n: sum(1 for d in dets if d[0] == n) for n in NAMES}
        print(f"  {label:>10}: {dt:5.0f} ms  {len(dets)} dets  {counts}")

    merged = merge(passes)
    counts = {n: sum(1 for d in merged if d[0] == n) for n in NAMES}
    print(f"\n  merged: {len(merged)} dets  {counts}")
    print(f"  total inference: {total:.0f} ms/image")

    out = Path("port/detect_dml_out.jpg")
    annotate(bgr, merged, out)
    print(f"  annotated -> {out}")


if __name__ == "__main__":
    main()
