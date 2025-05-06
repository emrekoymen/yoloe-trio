from ultralytics import YOLOE

# Initialize a YOLOE model
model = YOLOE("/home/emrek/Desktop/best.pt")  # or select yoloe-11s/m-seg.pt for different sizes

# Set text prompt to detect person and bus. You only need to do this once after you load the model.
names = ["person", "forklift", "driver"]
model.set_classes(names, model.get_text_pe(names))

# Run detection on the given image
results = model.predict("/home/emrek/Desktop/yoloe_fine_tuning/images/test/6556E2F7@0@231019002959_00090.jpg")

# Show results
results[0].show()
