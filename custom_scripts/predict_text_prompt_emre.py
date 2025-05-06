#!/usr/bin/env python3
"""
predict_text_prompt_batch.py
Apply a YOLOE-v8 model with text prompts to **all** images under a VOC-style
JPEGImages directory, write annotated images to one folder and Pascal-VOC
XML files (containing the predicted boxes) to another.

Dependencies:
  pip install ultralytics supervision tqdm pillow  # Note: Removed specific ultralytics version
"""

import argparse
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm
import supervision as sv
# Removed old import: from supervision.annotation.voc import detections_to_voc_xml
# Added potential new import path based on supervision library structure changes
import supervision.dataset.formats.pascal_voc as pascal_voc_format
from ultralytics import YOLOE


# --------------------------------------------------------------------------- #
#                             argument parsing                                #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True,
                        help="Path to JPEGImages directory")
    parser.add_argument("--checkpoint", type=str, default="yoloe-v8l-seg.pt",
                        help="YOLOE checkpoint (.pt) or HF repo id")
    parser.add_argument("--names", nargs="+", default=["person"],
                        help="Class names used as text prompts")
    parser.add_argument("--out-images-dir", type=str, default="pred_images",
                        help="Folder to save annotated images")
    parser.add_argument("--out-ann-dir", type=str, default="pred_annotations",
                        help="Folder to save Pascal-VOC XML files")
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu | cuda | mps")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference resolution")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
#                       helper: iterate image paths                           #
# --------------------------------------------------------------------------- #
def iter_images(images_dir: Path):
    exts = {".jpg", ".jpeg", ".png"}
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() in exts:
            yield p


# --------------------------------------------------------------------------- #
#                                main routine                                 #
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()

    src_dir = Path(args.source).expanduser().resolve()
    img_out_dir = Path(args.out_images_dir).expanduser().resolve()
    ann_out_dir = Path(args.out_ann_dir).expanduser().resolve()

    img_out_dir.mkdir(parents=True, exist_ok=True)
    ann_out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load model once
    model = YOLOE(args.checkpoint)
    model.to(args.device)
    model.set_classes(args.names, model.get_text_pe(args.names))

    # 2) loop over images
    for img_path in tqdm(list(iter_images(src_dir)), desc="Predicting"):
        image = Image.open(img_path).convert("RGB")

        results = model.predict(image, imgsz=args.imgsz, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])

        # -- create labelled visual ------------------------------------------- #
        resolution_wh = image.size
        thickness = sv.calculate_optimal_line_thickness(resolution_wh)
        text_scale = sv.calculate_optimal_text_scale(resolution_wh)

        labels = [
            f"{cls} {conf:.2f}"
            for cls, conf in zip(detections["class_name"], detections.confidence)
        ]

        vis_img = image.copy()
        vis_img = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX,
                                     opacity=0.4).annotate(vis_img, detections)
        vis_img = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX,
                                    thickness=thickness).annotate(vis_img, detections)
        vis_img = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX,
                                      text_scale=text_scale,
                                      smart_position=True).annotate(vis_img, detections,
                                                                      labels=labels)

        vis_img.save(img_out_dir / img_path.name)

        # -- write Pascal-VOC XML -------------------------------------------- #
        # Get image dimensions (width, height) from PIL Image
        width, height = image.size
        # Create image_shape tuple (height, width) if needed by the function
        depth = 3
        image_shape = (height, width, depth)

        # Use the potentially new function from the imported module
        # Ensure the arguments match the function's signature in your supervision version
        xml_str = pascal_voc_format.detections_to_pascal_voc(
            detections=detections,
            classes=args.names,
            filename=img_path.name,
	    image_shape=image_shape
        )
        (ann_out_dir / f"{img_path.stem}.xml").write_text(xml_str, encoding="utf-8")

    print(f"Done!  Images -> {img_out_dir}")
    print(f"        XMLs   -> {ann_out_dir}")


if __name__ == "__main__":
    main()
