"""Build the A1 splits from Food-101 and pre-resize the selected images.

Splits are stratified by class with a fixed seed:
  train: first `--train` images of each class (from the official train split, shuffled)
  val:   next `--val` images of each class (same official train split, disjoint from train)
  test:  first `--test` images of each class (from the official test split, shuffled)
Lists are written to A1/splits/{train,val,test}.txt as "<class>/<image id>".

Images are resized once so that the short side is 256 px (JPEG decoding of the
512 px originals is the CPU bottleneck otherwise). Run from the A1/ folder:
    python scripts/prepare_data.py
"""
import argparse, random
from multiprocessing import Pool
from pathlib import Path
from PIL import Image

A1 = Path(__file__).resolve().parents[1]
RAW = A1.parent / "data" / "food-101"
OUT = A1.parent / "data" / "food-101-256"
SHORT = 256


def read_split(name):
    """{class: [ids]} from Food-101's meta/{train,test}.txt (lines 'class/id')."""
    by_class = {}
    for line in (RAW / "meta" / f"{name}.txt").read_text().split():
        by_class.setdefault(line.split("/")[0], []).append(line)
    return by_class


def stratified(by_class, counts, seed):
    """Shuffle each class with `seed`, then cut consecutive, non-overlapping chunks."""
    rng = random.Random(seed)
    parts = [[] for _ in counts]
    for cls in sorted(by_class):
        items = sorted(by_class[cls])
        rng.shuffle(items)
        start = 0
        for part, n in zip(parts, counts):
            assert start + n <= len(items), f"class {cls}: not enough images"
            part += items[start:start + n]
            start += n
    return parts


def resize(item):
    dst = OUT / f"{item}.jpg"
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(RAW / "images" / f"{item}.jpg") as im:
        im.draft("RGB", (SHORT, SHORT))          # let the JPEG decoder downscale cheaply
        im = im.convert("RGB")
        w, h = im.size
        s = SHORT / min(w, h)
        im.resize((round(w * s), round(h * s)), Image.BICUBIC).save(dst, quality=92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=100, help="images per class (protocol A1.1: 300)")
    ap.add_argument("--val", type=int, default=20, help="images per class (protocol A1.1: 50)")
    ap.add_argument("--test", type=int, default=50, help="images per class (protocol A1.1: 250 = full)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    train, val = stratified(read_split("train"), [a.train, a.val], a.seed)
    (test,) = stratified(read_split("test"), [a.test], a.seed)
    split_dir = A1 / "splits"
    split_dir.mkdir(exist_ok=True)
    for name, items in (("train", train), ("val", val), ("test", test)):
        (split_dir / f"{name}.txt").write_text("\n".join(items) + "\n")
        print(f"{name}: {len(items)} images")
    assert not set(train) & set(val), "train and val overlap"

    with Pool(a.workers) as pool:
        pool.map(resize, train + val + test, chunksize=64)
    print("resized images in", OUT)


if __name__ == "__main__":
    main()
