"""End-to-end proof: detect (AMD DirectML) -> blur faces/plates -> output JPEG.

Purpose is to judge DETECTION quality visually -- does the ported detection catch
the faces/plates, so the right regions get blurred? Uses the same pixelate +
box-blur math as upstream src/blur/blur.py, but applied on a full PIL decode of
the image instead of the MCU-lossless jpegtran '-drop' path.

Why not the real MCU path yet: blur.py's losslessness comes from native tools
(turbojpeg + jpegtran/djpeg/cjpeg from libjpeg-turbo) that need a separate Windows
port. That matters for output file size / avoiding whole-image recompression, NOT
for which regions get blurred -- so it is deferred. This full-decode blur is only
for validation, not production (it recompresses the whole JPEG).

Only face + plate are blurred (signs are detected but never blurred, matching
blur.py). Boxes < 12px are skipped, as in blur.py.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

sys.path.insert(0, str(Path(__file__).parent))  # make detect_dml importable
from detect_dml import make_session, detect, merge  # noqa: E402

MODEL = Path("models/yolo11l_panoramax.onnx")  # best-recall config
CONF = 0.15
BLUR_CLASSES = ("face", "plate")
DEFAULT_IMG = r"C:\Users\kk\Code\mapillary-export\staging\2021 10 09 kiev lybid 2\GSAK3613.JPG"


def blur_region(img, box):
    """Pixelate then box-blur one region, matching blur.py's math."""
    x1, y1, x2, y2 = box
    region = img.crop((x1, y1, x2, y2))
    w, h = region.size
    radius = max(int(max(w, h) / 12) >> 3 << 3, 8)
    reduced = ImageOps.scale(region, 1 / radius, resample=Image.NEAREST)
    pixelated = ImageOps.scale(reduced, radius, resample=Image.NEAREST)
    blurred = pixelated.filter(ImageFilter.BoxBlur(radius))
    img.paste(blurred, (x1, y1))


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG
    model_path = Path(sys.argv[2]) if len(sys.argv) > 2 else MODEL

    bgr = cv2.imread(img_path)
    if bgr is None:
        sys.exit(f"Could not read image: {img_path}")

    session = make_session(model_path)
    dets = merge(detect(session, bgr, conf_thr=CONF, do_xl=True))
    to_blur = [d for d in dets if d[0] in BLUR_CLASSES]
    print(f"image: {img_path}  {bgr.shape[1]}x{bgr.shape[0]}")
    print(f"detections: {len(dets)}  to blur (face/plate): {len(to_blur)}")

    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    blurred = 0
    for name, conf, x1, y1, x2, y2 in to_blur:
        if x2 - x1 < 12 or y2 - y1 < 12:
            print(f"  skip {name} (too small: {x2 - x1}x{y2 - y1})")
            continue
        blur_region(img, (x1, y1, x2, y2))
        blurred += 1
        print(f"  blurred {name} {conf:.2f} at ({x1},{y1},{x2},{y2})")

    out = Path("port/blur_test_out.jpg")
    img.save(out, quality=90)
    print(f"\n{blurred} regions blurred -> {out}")


if __name__ == "__main__":
    main()
