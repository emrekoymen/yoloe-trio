from pathlib import Path
import numpy as np
from lxml import etree
import torch
import cv2 # Added for visualization
import os  # Added for creating directory

# Target classes and mapping from XML names
CLASS_NAMES  = ["pedestrian_side", "forklift", "operator"]
CLASS2ID     = {n:i for i,n in enumerate(CLASS_NAMES)}
# --- Configuration ---
MIN_BOX_AREA = 900 # Minimum bounding box area (width * height) in pixels to consider
LABEL_MAP = {
    "PEDESTRIAN_SIDE": "pedestrian_side",
    "FORKLIFT": "forklift",
    "OPERATOR": "operator",
    # Add identity mappings for the new canonical names
    "pedestrian_side": "pedestrian_side",
    "forklift": "forklift",
    "operator": "operator",
    # Map 'driver' to 'operator' if it can appear in XMLs
    "driver": "operator",
    # "person" is no longer a canonical name and will be ignored unless mapped otherwise
    # Other names like REACH_TRUCK, STACKER will be ignored as they are not mapped
    "person": "pedestrian_side",
}

def voc_xml_to_boxes(xml_path, is_prediction):
    """
    Returns list of dicts   {'boxes': FloatTensor[N,4],
                             'labels': LongTensor[N],
                             'scores': FloatTensor[N] or None (only if is_prediction=True)}
    ready for torchmetrics.
    Applies label mapping and handles missing scores for predictions.
    """
    try:
        tree = etree.parse(str(xml_path))
    except etree.XMLSyntaxError as e:
        print(f"Error parsing XML file {xml_path}: {e}")
        # Return structure consistent with no detections
        d = {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "labels": torch.empty(0, dtype=torch.int64),
        }
        if is_prediction:
            d["scores"] = torch.empty(0, dtype=torch.float32)
        return d
    except FileNotFoundError:
        print(f"Error: XML file not found {xml_path}")
        # Return structure consistent with no detections
        d = {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "labels": torch.empty(0, dtype=torch.int64),
        }
        if is_prediction:
            d["scores"] = torch.empty(0, dtype=torch.float32)
        return d

    root = tree.getroot()
    # Basic check for expected structure
    if root.find("size/width") is None or root.find("size/height") is None:
         print(f"Warning: Missing size information in {xml_path.name}. Skipping file.")
         d = {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "labels": torch.empty(0, dtype=torch.int64),
         }
         if is_prediction:
             d["scores"] = torch.empty(0, dtype=torch.float32)
         return d

    # w  = int(root.findtext("size/width")) # Not currently used
    # h  = int(root.findtext("size/height")) # Not currently used

    boxes, labels, scores = [], [], []
    objects_skipped = 0
    for obj in root.findall("object"):
        name_in_xml = obj.findtext("name")
        bndbox_node = obj.find("bndbox")

        # Check for essential elements
        if name_in_xml is None or bndbox_node is None:
            objects_skipped += 1
            continue

        try:
            xyxy = [
                float(bndbox_node.findtext("xmin")),
                float(bndbox_node.findtext("ymin")),
                float(bndbox_node.findtext("xmax")),
                float(bndbox_node.findtext("ymax"))
            ]
        except (TypeError, ValueError):
             # Handle cases where coordinates are not valid floats
             objects_skipped += 1
             continue

        # Calculate box area
        box_w = xyxy[2] - xyxy[0]
        box_h = xyxy[3] - xyxy[1]
        box_area = box_w * box_h

        # Skip box if area is too small or invalid
        if box_area < MIN_BOX_AREA:
            objects_skipped += 1
            continue # Skip to next object

        # Map the label name
        standard_name = LABEL_MAP.get(name_in_xml)

        # Check if the mapped name is one of the classes we care about
        if standard_name and standard_name in CLASS2ID:
            label_id = CLASS2ID[standard_name]
            boxes.append(xyxy)
            labels.append(label_id)

            if is_prediction:
                # Predictions require a score for every box
                conf = obj.findtext("confidence")
                try:
                    # Use confidence if available, otherwise default to 1.0
                    score = float(conf) if conf is not None else 1.0
                except (ValueError, TypeError):
                    score = 1.0 # Default if confidence is not a valid float
                scores.append(score)
        else:
            objects_skipped += 1
            # Optional: print warning about skipped label
            # print(f"Warning: Skipping object with unknown or unmapped label '{name_in_xml}' in {xml_path.name}")

    if objects_skipped > 0:
         print(f"Info: Skipped {objects_skipped} objects with missing/invalid data or unmapped labels in {xml_path.name}")

    # Handle cases where no *valid* objects matching the desired classes are found
    if not boxes:
        d = {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "labels": torch.empty(0, dtype=torch.int64),
        }
        if is_prediction:
             d["scores"] = torch.empty(0, dtype=torch.float32) # Still need scores key for preds
        return d

    # --- Create tensors --- 
    d = {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }
    # Add scores tensor only if it's a prediction
    if is_prediction:
         # This assertion should hold true given the logic above
         assert len(scores) == len(boxes), f"Internal error: Score/Box mismatch in {xml_path.name}"
         d["scores"] = torch.tensor(scores, dtype=torch.float32)

    return d


from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
import torch, os

GT_XML_DIR   = Path("/home/emrek/cursor/yoloe/yoloe_test_dataset/Annotations")
PRED_XML_DIR = Path("/home/emrek/cursor/yoloe/pred-dir/annotations")   # from previous script
# Assume images are in a parallel directory
IMG_DIR      = Path("/home/emrek/cursor/yoloe/yoloe_test_dataset/JPEGImages") 
VISUALIZATION_OUTPUT_DIR = Path("output_visualizations")
CREATE_VISUALIZATIONS = True # Set to False to disable drawing boxes

# CLASS_NAMES/CLASS2ID defined above with LABEL_MAP

# ➊ "full COCO" metric – what you already had
map_coco_metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True) # Default IoU = [0.5...0.95]
# ➋ Single-threshold metric – AP@0.50 only
map_50_metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True,
                                     iou_thresholds=[0.5])      # Explicitly set IoU = 0.50 only

# Create output directory if needed
if CREATE_VISUALIZATIONS:
    VISUALIZATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Define drawing colors (BGR format for OpenCV) ---
COLOR_GT = (0, 255, 0)    # Green
COLOR_PRED = (0, 0, 255)  # Red

# --- Evaluation Loop --- 
for xml_gt in tqdm(sorted(GT_XML_DIR.glob("*.xml"))):
    img_id = xml_gt.stem
    xml_pred_path = PRED_XML_DIR / f"{img_id}.xml"

    # Process Ground Truth (no scores needed)
    target = voc_xml_to_boxes(xml_gt, is_prediction=False)

    # Process Predictions (scores needed)
    if xml_pred_path.exists():
        preds = voc_xml_to_boxes(xml_pred_path, is_prediction=True)
    else:
        # If no prediction file, assume no detections for this image
        print(f"Warning: Prediction file not found for {img_id}.xml. Assuming no detections.")
        preds = {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "labels": torch.empty(0, dtype=torch.int64),
            "scores": torch.empty(0, dtype=torch.float32) # Must include scores key
        }

    # Add robust error handling around metric update
    try:
        # Update both metrics
        map_coco_metric.update([preds], [target])
        map_50_metric.update([preds], [target])
    except Exception as e:
        print(f"\nError updating metric for image {img_id}:")
        print(f"  GT path: {xml_gt}")
        print(f"  Pred path: {xml_pred_path}")
        print(f"  Target dict: {target}")
        print(f"  Preds dict: {preds}")
        print(f"  Error: {e}")
        # Decide whether to continue or stop
        # continue # Skip this image
        raise # Re-raise the exception to stop execution

    # --- Visualization (Optional) ---
    if CREATE_VISUALIZATIONS:
        img_path = IMG_DIR / f"{img_id}.jpg" # Assumes JPG format
        if not img_path.exists():
            img_path = IMG_DIR / f"{img_id}.png" # Try PNG

        if img_path.exists():
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Failed to read image {img_path}. Skipping visualization.")
            else:
                # Draw Ground Truth Boxes
                for i in range(len(target['boxes'])):
                    box = target['boxes'][i].int().numpy()
                    label_idx = target['labels'][i].item()
                    label_name = CLASS_NAMES[label_idx]
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), COLOR_GT, 2)
                    cv2.putText(image, f"GT: {label_name}", (box[0], box[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GT, 2)

                # Draw Prediction Boxes
                for i in range(len(preds['boxes'])):
                    box = preds['boxes'][i].int().numpy()
                    label_idx = preds['labels'][i].item()
                    score = preds['scores'][i].item()
                    label_name = CLASS_NAMES[label_idx]
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), COLOR_PRED, 2)
                    cv2.putText(image, f"PD: {label_name} ({score:.2f})", (box[0], box[1] - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PRED, 2) # Offset slightly

                # Save the annotated image
                output_path = VISUALIZATION_OUTPUT_DIR / f"{img_id}_annotated.jpg"
                cv2.imwrite(str(output_path), image)
        else:
            print(f"Warning: Image file not found for {img_id} at {img_path} or {img_path.with_suffix('.png')}. Skipping visualization.")

# --- Compute and Print Results --- 
try:
    coco_results = map_coco_metric.compute()
    map50_results = map_50_metric.compute()

    print("\n--- Full Results Dictionary (COCO Metric) ---")
    print(coco_results)
    print("\n--- Full Results Dictionary (AP@0.50 Metric) ---")
    print(map50_results)

    print("\n--- Formatted Results ---")
    # Results from the COCO metric instance
    print(f"mAP@0.5:0.95: {coco_results.get('map', -1.0):.4f}")
    print(f"mAP@0.75:    {coco_results.get('map_75', -1.0):.4f}")
    # Use map_50 from the dedicated metric instance for consistency
    print(f"mAP@0.50:    {map50_results.get('map', -1.0):.4f}") # Note: map_50 uses 'map' key here

    # Check only for key existence, as comparing the tensor directly is ambiguous
    if 'map_per_class' in coco_results:
        print("\nPer-class AP @ 0.5:0.95:")
        per_class_ap_coco = coco_results['map_per_class']
        # Check if it's the placeholder value (-1) before iterating
        if per_class_ap_coco.numel() == 1 and per_class_ap_coco.item() == -1:
            print("  Per-class AP data is invalid (-1).")
        else:
            for i, class_name in enumerate(CLASS_NAMES):
                # Ensure index is within tensor bounds
                ap = per_class_ap_coco[i].item() if i < len(per_class_ap_coco) else -1.0
                print(f"  - {class_name}: {ap:.4f}")
    else:
        print("\nPer-class AP @ 0.5:0.95 not available in results.")

    # Check only for key existence
    if 'map_per_class' in map50_results:
        print("\nPer-class AP @ 0.50:")
        per_class_ap_50 = map50_results['map_per_class'] # Key is 'map_per_class' here
        # Check if it's the placeholder value (-1) before iterating
        if per_class_ap_50.numel() == 1 and per_class_ap_50.item() == -1:
            print("  Per-class AP @ 0.50 data is invalid (-1).")
        else:
            # Map class IDs from the results to names
            class_ids_present = map50_results.get('classes', torch.tensor([])).tolist()
            ap_values_50 = per_class_ap_50.tolist()

            # Create a dictionary mapping class_id to AP value
            ap_dict_50 = {cls_id: ap for cls_id, ap in zip(class_ids_present, ap_values_50)}

            # Print AP for each class in CLASS_NAMES, handling missing classes
            for i, class_name in enumerate(CLASS_NAMES):
                class_id = CLASS2ID[class_name] # Get the expected class ID
                ap = ap_dict_50.get(class_id, -1.0) # Get AP if class_id was in results, else -1
                print(f"  - {class_name}: {ap:.4f}")
    else:
        print("\nPer-class AP @ 0.50 not available in results (missing 'map_per_class' key in map50_results).")

except Exception as e:
    print(f"\nError computing final metrics: {e}")
