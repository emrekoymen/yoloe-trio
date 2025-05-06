# make_my_labels.py
import torch
from ultralytics.nn.text_model import build_text_model

# 1) Build the MobileCLIP text model (as used by YOLOE)
text_model = build_text_model('mobileclip:blt', device='cuda').eval()

# 2) Your six class names (exactly as in your VOC XML <name> tags)
labels = [
    "PEDESTRIAN_SIDE",
    "FORKLIFT",
    "OPERATOR",
]

# 3) Tokenize & encode in one batch
tokens = text_model.tokenizer(labels).to(text_model.device)
with torch.no_grad():
    feats = text_model.encode_text(tokens)  # shape: (num_classes, D)
    # Add an extra dimension to match expected format [1, num_classes, D]
    feats = feats.unsqueeze(0)

# 4) Save in the format expected by train_pe.py: {"names": [...], "pe": tensor(...)}
save_data = {"names": labels, "pe": feats.cpu()}
torch.save(save_data, 'my_labels.pt')

print(f"✅ my_labels.pt saved with keys: {list(save_data.keys())}")

