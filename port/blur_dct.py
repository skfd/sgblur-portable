"""Production blur, pure-Python and portable: detect (AMD DirectML) then blur
face/plate regions by editing JPEG DCT coefficients directly -- no native
binaries, no subprocess, no whole-image recompression.

For every 8x8 DCT block that a face/plate box touches, we zero the AC
coefficients and keep the DC (block average). Each block collapses to its mean
colour = clean block pixelation, done in the coefficient domain. Every OTHER
block is written back with identical coefficients, so the rest of the image is
pixel-for-pixel preserved (no generational loss). This replaces blur_test.py's
full-decode PIL blur (which recompressed the whole 16 MP).

Signs are detected but never blurred (matches upstream). Chroma subsampling is
handled per component via the coefficient-array shapes.
"""

import sys
from pathlib import Path

import cv2
import jpeglib

sys.path.insert(0, str(Path(__file__).parent))  # make detect_dml importable
from detect_dml import make_session, detect, merge  # noqa: E402

MODEL = Path("models/yolo11l_panoramax.onnx")  # best-recall config
CONF = 0.15
BLUR_CLASSES = ("face", "plate")


def zero_ac(coef, r0, r1, c0, c1):
    """Zero AC coefficients (keep DC) for blocks [r0..r1] x [c0..c1] inclusive."""
    R, C = coef.shape[:2]
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(R - 1, r1), min(C - 1, c1)
    if r1 < r0 or c1 < c0:
        return 0
    block = coef[r0:r1 + 1, c0:c1 + 1]
    dc = block[:, :, 0, 0].copy()
    block[:, :, :, :] = 0
    block[:, :, 0, 0] = dc
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def blur_box_dct(components, y_shape, box):
    """Zero AC in every block each component's box overlaps, honouring subsampling."""
    x1, y1, x2, y2 = box
    for comp in components:
        hsub = round(y_shape[1] / comp.shape[1])  # 1 (luma) or 2 (subsampled chroma)
        vsub = round(y_shape[0] / comp.shape[0])
        c0 = x1 // (8 * hsub)
        c1 = (x2 - 1) // (8 * hsub)
        r0 = y1 // (8 * vsub)
        r1 = (y2 - 1) // (8 * vsub)
        zero_ac(comp, r0, r1, c0, c1)


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: blur_dct.py <image.jpg> [out.jpg] [model.onnx]")
    img_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "port/blur_dct_out.jpg"
    model_path = Path(sys.argv[3]) if len(sys.argv) > 3 else MODEL

    bgr = cv2.imread(img_path)
    if bgr is None:
        sys.exit(f"Could not read image: {img_path}")

    session = make_session(model_path)
    dets = merge(detect(session, bgr, conf_thr=CONF, do_xl=True))
    to_blur = [d for d in dets if d[0] in BLUR_CLASSES]
    print(f"image: {img_path}  {bgr.shape[1]}x{bgr.shape[0]}")
    print(f"detections: {len(dets)}  to blur (face/plate): {len(to_blur)}")

    d = jpeglib.read_dct(img_path)
    # sanity: DCT grid must match the pixels detection ran on
    assert d.Y.shape[1] * 8 == bgr.shape[1] and d.Y.shape[0] * 8 == bgr.shape[0], \
        f"DCT grid {d.Y.shape[:2]} does not match image {bgr.shape[:2]}"
    components = [d.Y] + ([d.Cb, d.Cr] if d.has_chrominance else [])

    blurred = 0
    for name, conf, x1, y1, x2, y2 in to_blur:
        if x2 - x1 < 12 or y2 - y1 < 12:
            print(f"  skip {name} (too small: {x2 - x1}x{y2 - y1})")
            continue
        blur_box_dct(components, d.Y.shape, (x1, y1, x2, y2))
        blurred += 1
        print(f"  blurred {name} {conf:.2f} at ({x1},{y1},{x2},{y2})")

    d.write_dct(out_path)
    print(f"\n{blurred} regions blurred -> {out_path}")


if __name__ == "__main__":
    main()
