#!/usr/bin/env python3
"""
box2poly.py – convert YOLO‑v5/8 detection labels (cx,cy,w,h)
             → YOLO segmentation polygons (x1,y1 … x4,y4)

Usage:
    python emre_test.py --src data/labels      \
                        --dst data/labels-seg  \
                        --img-ext .jpg         # or .png

Keeps folder structure (train/val/test) intact.
"""
from pathlib import Path
import argparse, shutil

def convert_line(line: str) -> str:
    cls, cx, cy, w, h = map(float, line.strip().split())
    x1, y1 = cx - w/2, cy - h/2           # top‑left
    x2, y2 = cx + w/2, cy - h/2           # top‑right
    x3, y3 = cx + w/2, cy + h/2           # bottom‑right
    x4, y4 = cx - w/2, cy + h/2           # bottom‑left
    return f"{int(cls)} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} " \
           f"{x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}\n"

def main(src, dst, img_ext):
    src, dst = Path(src), Path(dst)
    if dst.exists(): shutil.rmtree(dst)
    count = 0
    for txt in src.rglob("*.txt"):
        rel = txt.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        with txt.open() as f_in, out.open("w") as f_out:
            for line in f_in:
                f_out.write(convert_line(line))
        count += 1
    print(f"✔ Converted {count} label files → {dst}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert YOLO detection labels to segmentation polygons.")
    ap.add_argument("--src", required=True, help="root directory of detection labels (e.g., dataset/labels)")
    ap.add_argument("--dst", required=True, help="root directory to write segmentation labels (e.g., dataset/labels-seg)")
    ap.add_argument("--img-ext", default=".jpg", help="image file extension (used for reference in comments/YAML, not directly by script)")
    args = ap.parse_args()
    main(args.src, args.dst, args.img_ext)
