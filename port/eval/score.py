"""Score detection configs against a recall-ceiling reference.

Certifying absolute ground truth on dense 360 street scenes needs human review
at full resolution (a privacy call), so instead of inventing labels we measure
RELATIVE recall: take the most generous config (yolo11l, all scales, low conf)
as the reference "recall ceiling", then report how much of it each cheaper /
faster config retains, plus speed. This answers the tuning question -- "what do
we give up for throughput?" -- honestly. Pruning false positives out of the
reference (candidates.json) later only sharpens the numbers.

Reference = port/eval/candidates.json (yolo11l, all scales, conf 0.15).
"""

import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))  # make port/ importable
from detect_dml import make_session, detect, merge, iou  # noqa: E402

CANDIDATES = Path("port/eval/candidates.json")
CLASSES = ("face", "plate", "sign")
MATCH_IOU = 0.5

# (label, model_onnx, conf, do_xl, xl_size). Edit per experiment.
# Frontier snapshot (14 frames): no free lunch -- every ~2x speedup costs real
# face/plate recall. Best recall = l all 0.15 fp32 (~11.9s/img, ~9.6d for 70k).
# fp16 is ~2.4x faster but drops face 1.00->0.67, plate 1.00->0.57 AND is flaky
# on this DirectML runtime, so it is not recommended.
CONFIGS = [
    ("l all 0.15 fp32 ref", "models/yolo11l_panoramax.onnx", 0.15, True,  4096),
    ("l all 0.30 fp32",     "models/yolo11l_panoramax.onnx", 0.30, True,  4096),
    ("l S+L 0.15 fp32",     "models/yolo11l_panoramax.onnx", 0.15, False, 4096),
    ("s all 0.15 fp32",     "models/yolo11s_panoramax.onnx", 0.15, True,  4096),
    ("l all 0.15 fp16",     "models/yolo11l_panoramax_fp16.onnx", 0.15, True, 4096),
]


def load_reference():
    data = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    ref = {}
    for name, entry in data.items():
        boxes = {c: [] for c in CLASSES}
        for d in entry["dets"]:
            cls, _conf, x1, y1, x2, y2 = d
            if cls in boxes:
                boxes[cls].append((x1, y1, x2, y2))
        ref[name] = {"path": entry["path"], "boxes": boxes}
    return ref


def recall_vs_ref(ref_boxes, got_boxes):
    """Greedy IoU match. Returns matched count per class."""
    matched = {}
    for c in CLASSES:
        used = [False] * len(got_boxes[c])
        m = 0
        for rb in ref_boxes[c]:
            best, bi = MATCH_IOU, -1
            for i, gb in enumerate(got_boxes[c]):
                if used[i]:
                    continue
                v = iou(rb, gb)
                if v >= best:
                    best, bi = v, i
            if bi >= 0:
                used[bi] = True
                m += 1
        matched[c] = m
    return matched


def main():
    ref = load_reference()
    totals_ref = {c: sum(len(r["boxes"][c]) for r in ref.values()) for c in CLASSES}
    print(f"reference objects: " + "  ".join(f"{c}={totals_ref[c]}" for c in CLASSES))
    print(f"frames: {len(ref)}\n")

    sessions = {}
    rows = []
    for label, model, conf, do_xl, xl_size in CONFIGS:
        if model not in sessions:
            sessions[model] = make_session(model)
        sess = sessions[model]

        matched = {c: 0 for c in CLASSES}
        extra = {c: 0 for c in CLASSES}
        t0 = time.perf_counter()
        for name, r in ref.items():
            bgr = cv2.imread(r["path"])
            try:
                dets = merge(detect(sess, bgr, conf_thr=conf, do_xl=do_xl, xl_size=xl_size))
            except Exception as e:
                print(f"    ! {label} failed on {name}: {repr(e)[:120]}")
                continue
            got = {c: [(d[2], d[3], d[4], d[5]) for d in dets if d[0] == c] for c in CLASSES}
            m = recall_vs_ref(r["boxes"], got)
            for c in CLASSES:
                matched[c] += m[c]
                extra[c] += max(0, len(got[c]) - m[c])
        dt = (time.perf_counter() - t0) / len(ref)

        rec = {c: (matched[c] / totals_ref[c] if totals_ref[c] else 1.0) for c in CLASSES}
        rows.append((label, rec, extra, dt))
        print(f"  {label:18} face {rec['face']:.2f}  plate {rec['plate']:.2f}  "
              f"sign {rec['sign']:.2f}   +extra f/p/s {extra['face']}/{extra['plate']}/{extra['sign']}"
              f"   {dt:5.1f} s/img")

    print("\n(recall = fraction of the generous reference's boxes this config kept;")
    print(" 'extra' = boxes this config found beyond the reference -- new hits or false positives)")


if __name__ == "__main__":
    main()
