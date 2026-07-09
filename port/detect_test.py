"""Phase 1 smallest test: run YOLOv11 face/plate/sign detection on the AMD
Radeon 890M iGPU via ONNX Runtime + DirectML, on one 5760x2880 equirect image.

Why this exists: Ultralytics' own ONNX loader only ever selects CUDA / CoreML /
CPU execution providers (see ultralytics/nn/backends/onnx.py) -- it will NOT use
DirectML, so loading the .onnx through Ultralytics silently runs on CPU. Here we
drive onnxruntime directly with DmlExecutionProvider to actually hit the iGPU,
and do the letterbox pre-processing + NMS post-processing ourselves.

Runs the same model on DirectML and on CPU, prints which provider really ran,
compares detection counts and wall-clock, and writes an annotated JPEG so the
boxes can be eyeballed.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch  # only for torchvision NMS
import torchvision

# --- config (mirrors src/detect/detect.py) ---
MODEL = Path("models/yolo11s_panoramax.onnx")
IMGSZ = 2048  # must match the size the ONNX was exported at (fixed, dynamic=False)
MIN_CONF = 0.30
IOU = 0.70  # Ultralytics default NMS IoU
NAMES = ["sign", "plate", "face"]
COLORS = {"sign": (0, 200, 255), "plate": (0, 255, 0), "face": (255, 0, 0)}  # BGR

DEFAULT_IMG = r"C:\Users\kk\Code\mapillary-export\staging\2021 10 09 kiev lybid 2\GSAK3613.JPG"


def letterbox(im, new_shape=IMGSZ, color=(114, 114, 114)):
    """Resize keeping aspect ratio, pad to a square. Returns padded image plus
    the ratio and (left, top) padding needed to map boxes back to the original."""
    h, w = im.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    left = (new_shape - nw) // 2
    top = (new_shape - nh) // 2
    out = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out, r, left, top


def preprocess(bgr):
    padded, r, left, top = letterbox(bgr)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None]  # HWC -> 1,C,H,W
    return np.ascontiguousarray(x), r, left, top


def postprocess(out, r, left, top, orig_shape):
    """out: (1, 4+nc, N) YOLO11 head. Returns list of (cls_name, conf, x1,y1,x2,y2)."""
    preds = out[0].transpose(1, 0)  # (N, 4+nc)
    boxes_xywh = preds[:, :4]
    scores_all = preds[:, 4:]
    cls = scores_all.argmax(1)
    conf = scores_all.max(1)

    keep = conf >= MIN_CONF
    boxes_xywh, conf, cls = boxes_xywh[keep], conf[keep], cls[keep]
    if len(boxes_xywh) == 0:
        return []

    # xywh (centre) -> xyxy in letterboxed space
    xy = boxes_xywh[:, :2]
    wh = boxes_xywh[:, 2:4]
    xyxy = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)

    # undo padding + scale back to original image coords
    xyxy[:, [0, 2]] -= left
    xyxy[:, [1, 3]] -= top
    xyxy /= r
    h, w = orig_shape[:2]
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)

    # per-class NMS (matches Ultralytics default agnostic=False)
    tb = torch.from_numpy(xyxy)
    ts = torch.from_numpy(conf)
    tc = torch.from_numpy(cls)
    dets = []
    for c in np.unique(cls):
        m = tc == int(c)
        idx = torchvision.ops.nms(tb[m], ts[m], IOU)
        for i in idx.tolist():
            b = tb[m][i].tolist()
            dets.append((NAMES[int(c)], float(ts[m][i]), *[int(v) for v in b]))
    return dets


def run(session, x, warmup=1, iters=3):
    name = session.get_inputs()[0].name
    for _ in range(warmup):
        session.run(None, {name: x})
    t0 = time.perf_counter()
    for _ in range(iters):
        out = session.run(None, {name: x})
    dt = (time.perf_counter() - t0) / iters
    return out[0], dt


def make_session(providers):
    return ort.InferenceSession(str(MODEL), providers=providers)


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG
    if not MODEL.exists():
        sys.exit(f"Model not found: {MODEL}. Export it first (see README).")
    bgr = cv2.imread(img_path)
    if bgr is None:
        sys.exit(f"Could not read image: {img_path}")
    print(f"image: {img_path}  {bgr.shape[1]}x{bgr.shape[0]}")
    print(f"model: {MODEL}  imgsz={IMGSZ}  conf>={MIN_CONF}\n")

    x, r, left, top = preprocess(bgr)

    results = {}
    for label, providers in [
        ("DirectML (iGPU)", ["DmlExecutionProvider", "CPUExecutionProvider"]),
        ("CPU", ["CPUExecutionProvider"]),
    ]:
        try:
            sess = make_session(providers)
        except Exception as e:
            print(f"[{label}] session failed: {e}")
            continue
        active = sess.get_providers()[0]
        out, dt = run(sess, x)
        dets = postprocess(out, r, left, top, bgr.shape)
        results[label] = (active, dt, dets)
        counts = {n: sum(1 for d in dets if d[0] == n) for n in NAMES}
        print(f"[{label}] provider used: {active}")
        print(f"[{label}] {dt * 1000:.0f} ms/inference   detections={len(dets)}  {counts}")

    # annotate using the DirectML result if present, else whatever ran
    key = "DirectML (iGPU)" if "DirectML (iGPU)" in results else next(iter(results), None)
    if key:
        _, _, dets = results[key]
        vis = bgr.copy()
        for name, conf, x1, y1, x2, y2 in dets:
            cv2.rectangle(vis, (x1, y1), (x2, y2), COLORS[name], 3)
            cv2.putText(vis, f"{name} {conf:.2f}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS[name], 2)
        outp = Path("port/detect_test_out.jpg")
        cv2.imwrite(str(outp), vis)
        print(f"\nannotated image -> {outp}")

    # correctness cross-check: DirectML detections should match CPU
    if "DirectML (iGPU)" in results and "CPU" in results:
        d_dml = len(results["DirectML (iGPU)"][2])
        d_cpu = len(results["CPU"][2])
        speedup = results["CPU"][1] / results["DirectML (iGPU)"][1]
        print(f"\ncross-check: DirectML {d_dml} vs CPU {d_cpu} detections; "
              f"iGPU speedup {speedup:.1f}x")


if __name__ == "__main__":
    main()
