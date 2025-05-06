from ultralytics import YOLOE

# Initialize a YOLOE model
model = YOLOE("yoloe-v8l-seg.pt")  # or select yoloe-11s/m-seg.pt for different sizes

# Set text prompt to detect person and bus. You only need to do this once after you load the model.
names = ["driver", "forklift", "suv"]
model.set_classes(names, model.get_text_pe(names))

# Run detection on the given image
results = model.predict("/home/emrek/cursor/yoloe/yoloe_test_dataset/JPEGImages/F0C0BD6F230819004051W_00016.jpg")

# Show results
results[0].show()