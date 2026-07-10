"""Generate generous detection candidates for the eval sample, to bootstrap
ground-truth labeling. Runs yolo11l at all scales with a LOW confidence
threshold (over-detect), writes annotated images for visual review plus a
candidates JSON. A human (or vision pass) then prunes false positives and adds
any missed faces/plates to produce ground_truth.json.
"""

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))  # make port/ importable
from detect_dml import make_session, detect, merge, annotate  # noqa: E402

MODEL = Path("models/yolo11l_panoramax.onnx")
CONF = 0.15  # generous: surface weak detections for review
SAMPLE = Path("port/eval/sample.txt")
ANNOT_DIR = Path("port/eval/annotated")
CANDIDATES = Path("port/eval/candidates.json")


def main():
    frames = [ln for ln in SAMPLE.read_text(encoding="utf-8").splitlines() if ln.strip()]
    session = make_session(MODEL)
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)

    out = {}
    for i, path in enumerate(frames, 1):
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"  skip (unreadable): {path}")
            continue
        passes = detect(session, bgr, conf_thr=CONF, do_xl=True)
        dets = merge(passes)
        name = Path(path).name
        out[name] = {"path": path,
                     "dets": [[d[0], round(d[1], 3), d[2], d[3], d[4], d[5]] for d in dets]}
        annotate(bgr, dets, ANNOT_DIR / name)
        counts = {c: sum(1 for d in dets if d[0] == c) for c in ("sign", "plate", "face")}
        print(f"  [{i}/{len(frames)}] {name}: {len(dets)} candidates {counts}")

    CANDIDATES.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nannotated -> {ANNOT_DIR}\ncandidates -> {CANDIDATES}")


if __name__ == "__main__":
    main()
