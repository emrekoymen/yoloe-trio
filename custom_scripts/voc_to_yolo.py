#!/usr/bin/env python3
# save as voc2yolo.py  and  python voc2yolo.py

import xml.etree.ElementTree as ET
from pathlib import Path
import shutil, random, sys
import math

# --- edit here ---------------------------------------------------------
DATASET_ROOT = Path('/home/emrek/cursor/yoloe/yoloe_test_dataset')
DEST_ROOT    = Path('/home/emrek/Desktop/yoloe_fine_tuning')
CLASS_NAMES = [
    "PEDESTRIAN_SIDE",
    "FORKLIFT",
    # "REACH_TRUCK",    # Removed as unused
    # "ORDER_PICKER",   # Removed as unused
    "OPERATOR",
    # "STACKER",        # Removed as unused
]
# TRAIN_LIST = DATASET_ROOT/'ImageSets'/'Main'/'train.txt'   # Removed - splitting handled automatically
# VAL_LIST   = DATASET_ROOT/'ImageSets'/'Main'/'val.txt'     # Removed - splitting handled automatically
# ----------------------------------------------------------------------

def convert_box(size, box):
    dw, dh = 1/size[0], 1/size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x*dw, y*dh, w*dw, h*dh

def make_dirs():
    for split in ('train', 'val', 'test'):
        (DEST_ROOT/f'images/{split}').mkdir(parents=True, exist_ok=True)
        (DEST_ROOT/f'labels/{split}').mkdir(parents=True, exist_ok=True)

def perform_split(all_ids, train_n=4000, val_n=500, test_n=500):
    """Shuffles and splits the image IDs into train, validation, and test sets."""
    n_total = len(all_ids)
    print(f"Found {n_total} total annotation files.")

    target_total = train_n + val_n + test_n
    print(f"Attempting to split into: Train={train_n}, Val={val_n}, Test={test_n} (Total requested: {target_total})")

    if n_total < target_total:
        print(f"Warning: Total images ({n_total}) is less than the desired split size ({target_total}). Adjusting split.")
        # Adjust proportionally if not enough images, or simply cap if preferred.
        # Let's cap for simplicity, ensuring no overlap.
        train_n = min(train_n, n_total)
        val_n = min(val_n, n_total - train_n)
        test_n = min(test_n, n_total - train_n - val_n)
        print(f"Adjusted split due to insufficient images: Train={train_n}, Val={val_n}, Test={test_n}")

    random.shuffle(all_ids)

    train_ids = all_ids[:train_n]
    val_ids   = all_ids[train_n : train_n + val_n]
    test_ids  = all_ids[train_n + val_n : train_n + val_n + test_n]

    actual_train = len(train_ids)
    actual_val = len(val_ids)
    actual_test = len(test_ids)
    actual_total_split = actual_train + actual_val + actual_test
    unused_count = n_total - actual_total_split

    print(f"Actual split created: Train={actual_train}, Validation={actual_val}, Test={actual_test}")
    print(f"Total images used in split: {actual_total_split}")
    if unused_count > 0:
        print(f"Number of images not used in any split: {unused_count}")

    # Sanity check (should always pass with slicing unless train_n/val_n/test_n were adjusted down)
    if actual_train != train_n or actual_val != val_n or actual_test != test_n:
         print(f"Warning: Actual split counts ({actual_train}, {actual_val}, {actual_test}) differ from target ({train_n}, {val_n}, {test_n}) - likely due to insufficient total images.")

    return train_ids, val_ids, test_ids

def process(split, ids):
    for img_id in ids:
        img_src = DATASET_ROOT/'JPEGImages'/f'{img_id}.jpg'
        xml = ET.parse(DATASET_ROOT/'Annotations'/f'{img_id}.xml').getroot()
        size = xml.find('size')
        w, h = int(size.find('width').text), int(size.find('height').text)

        yolo_lines = []
        for obj in xml.iter('object'):
            cls_name = obj.find('name').text
            if cls_name not in CLASS_NAMES:
                continue
            bnd = obj.find('bndbox')
            bb  = [float(bnd.find(k).text) for k in ('xmin','xmax','ymin','ymax')]
            bb  = convert_box((w, h), bb)
            yolo_lines.append(f"{CLASS_NAMES.index(cls_name)} " +
                              " ".join(f"{c:.6f}" for c in bb))

        (DEST_ROOT/f'images/{split}/{img_id}.jpg').write_bytes(img_src.read_bytes())
        (DEST_ROOT/f'labels/{split}/{img_id}.txt').write_text("\n".join(yolo_lines))

if __name__ == "__main__":
    make_dirs()
    # Always find all annotations and perform the split
    all_ids = [p.stem for p in (DATASET_ROOT/'Annotations').glob('*.xml')]
    if not all_ids:
        print("Error: No XML annotation files found in", DATASET_ROOT/'Annotations')
        sys.exit(1)

    train_ids, val_ids, test_ids = perform_split(all_ids)

    process('train', train_ids)
    process('val',   val_ids)
    process('test',  test_ids)
    print("✅ VOC → YOLO conversion finished.")

