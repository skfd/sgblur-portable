"""Phase 3 batch runner: blur a whole folder of JPEGs (faces/plates) on the AMD
iGPU, writing blurred copies to an output folder. Resumable -- skips images whose
output already exists -- so a multi-day 70k run can be stopped and restarted.

Loads the model once and reuses it across the batch. Per-image failures are
logged and skipped, never aborting the run. Relative folder structure (e.g. one
subfolder per sequence) is mirrored into the output.

    python port/blur_batch.py <input_dir> <output_dir> [model.onnx]
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # make sibling modules importable
from detect_dml import make_session  # noqa: E402
from blur_dct import blur_image, MODEL  # noqa: E402

EXTS = {".jpg", ".jpeg"}


def find_images(root):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)


def fmt_dur(sec):
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: blur_batch.py <input_dir> <output_dir> [model.onnx]")
    in_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    model_path = Path(sys.argv[3]) if len(sys.argv) > 3 else MODEL
    if not in_dir.is_dir():
        sys.exit(f"input dir not found: {in_dir}")

    images = find_images(in_dir)
    if not images:
        sys.exit(f"no JPEGs found under {in_dir}")
    print(f"{len(images)} images under {in_dir}\nmodel: {model_path}\noutput: {out_dir}\n")

    session = make_session(model_path)

    done = skipped = failed = 0
    total_blurred = 0
    t_start = time.perf_counter()
    proc_times = []
    for i, img in enumerate(images, 1):
        out = out_dir / img.relative_to(in_dir)
        if out.exists():
            skipped += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        try:
            _, n_blur = blur_image(session, img, out, verbose=False)
        except Exception as e:
            failed += 1
            print(f"  [{i}/{len(images)}] FAIL {img.name}: {repr(e)[:100]}")
            continue
        dt = time.perf_counter() - t0
        proc_times.append(dt)
        done += 1
        total_blurred += n_blur

        avg = sum(proc_times) / len(proc_times)
        remaining = (len(images) - i) * avg
        print(f"  [{i}/{len(images)}] {img.name}  {n_blur} blurred  "
              f"{dt:.1f}s  (avg {avg:.1f}s, ETA {fmt_dur(remaining)})")

    elapsed = time.perf_counter() - t_start
    print(f"\ndone={done} skipped={skipped} failed={failed}  "
          f"regions blurred={total_blurred}  elapsed={fmt_dur(elapsed)}")


if __name__ == "__main__":
    main()
