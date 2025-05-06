from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.train_pe import YOLOEPETrainer, YOLOEPESegTrainer
import os
from ultralytics.nn.tasks import guess_model_scale
from ultralytics.utils import yaml_load, LOGGER
import torch
import argparse

os.environ["PYTHONHASHSEED"] = "0"

# --- Argument Parsing Start ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOE with Prompt Embeddings")
    parser.add_argument('--yaml-file', type=str, required=True, help='Path to the dataset YAML configuration file (e.g., mydata.yaml)')
    parser.add_argument('--labels-pt', type=str, required=True, help='Path to the pre-computed prompt embeddings file (e.g., my_labels.pt)')
    # Add other arguments from model.train if needed to be configurable, e.g.:
    # parser.add_argument('--model', type=str, default='yoloe-v8l-seg.pt', help='Path to the base model weights')
    # parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    # parser.add_argument('--batch', type=int, default=128, help='Batch size')
    # parser.add_argument('--workers', type=int, default=4, help='Number of dataloader workers')
    # parser.add_argument('--device', type=str, default="0", help='Device to run on (e.g., "0" or "cpu")')
    return parser.parse_args()

args = parse_args()
# --- Argument Parsing End ---

# Use parsed arguments instead of hardcoded paths
data = args.yaml_file 
# data = "/home/emrek/cursor/yoloe/yoloe/custom_scripts/mydata.yaml" # Old hardcoded path

model_path = "yoloe-v8l-seg.yaml"

scale = guess_model_scale(model_path)
cfg_dir = "ultralytics/cfg"
default_cfg_path = f"{cfg_dir}/default.yaml"
extend_cfg_path = f"{cfg_dir}/coco_{scale}_train.yaml"
defaults = yaml_load(default_cfg_path)
extends = yaml_load(extend_cfg_path)
assert(all(k in defaults for k in extends))
LOGGER.info(f"Extends: {extends}")

model = YOLOE("yoloe-v8l-seg.pt")

# Ensure pe is set for classes
#names = list(yaml_load(data)['names'].values())
#tpe = model.get_text_pe(names)
#pe_path = "coco-pe.pt"
#torch.save({"names": names, "pe": tpe}, pe_path)

# — instead of recomputing pe, just load the one you already made —
# Use parsed argument instead of hardcoded path
pe_path = args.labels_pt
# pe_path = "/home/emrek/cursor/yoloe/finetuning/my_labels.pt" # Old hardcoded path
d = torch.load(pe_path)
names = d["names"]
# d["pe"] already contains the embeddings

head_index = len(model.model.model) - 1
freeze = [str(f) for f in range(0, head_index)]
for name, child in model.model.model[-1].named_children():
    if 'cv3' not in name:
        freeze.append(f"{head_index}.{name}")

freeze.extend([f"{head_index}.cv3.0.0", f"{head_index}.cv3.0.1", f"{head_index}.cv3.1.0", f"{head_index}.cv3.1.1", f"{head_index}.cv3.2.0", f"{head_index}.cv3.2.1"])
        
model.train(data=data, epochs=10, close_mosaic=5, batch=128, 
            optimizer='AdamW', lr0=1e-3, warmup_bias_lr=0.0, \
            weight_decay=0.025, momentum=0.9, workers=4, \
            device="0", **extends, \
            trainer=YOLOEPESegTrainer, freeze=freeze, train_pe_path=pe_path)
