"""Pick a diverse eval sample: evenly-spaced frames across each staging
sequence, so tuning sees varied scenes (not N near-identical frames). Writes
port/eval/sample.txt (one absolute path per line)."""

from pathlib import Path

STAGING = Path(r"C:\Users\kk\Code\mapillary-export\staging")
PER_SEQUENCE = 7
OUT = Path("port/eval/sample.txt")


def main():
    picks = []
    for seq in sorted(p for p in STAGING.iterdir() if p.is_dir()):
        frames = sorted(seq.glob("*.JPG")) + sorted(seq.glob("*.jpg"))
        frames = sorted(set(frames))
        if not frames:
            continue
        n = min(PER_SEQUENCE, len(frames))
        # evenly spaced indices across the sequence
        idxs = [round(i * (len(frames) - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]
        for i in sorted(set(idxs)):
            picks.append(frames[i])
        print(f"{seq.name}: {len(frames)} frames -> picked {len(set(idxs))}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(str(p) for p in picks), encoding="utf-8")
    print(f"\n{len(picks)} frames -> {OUT}")


if __name__ == "__main__":
    main()
