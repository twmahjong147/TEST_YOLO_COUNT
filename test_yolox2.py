#!/usr/bin/env python3
"""Test YOLOX with proper postprocessing"""

import cv2
import torch
import numpy as np

# Load image
image = cv2.imread('sample1.JPG')
h, w = image.shape[:2]
print(f"Image size: {w}x{h}")

# Load YOLOX model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('Megvii-BaseDetection/YOLOX', 'yolox_nano', pretrained=True)
model.to(device)
model.eval()

# Preprocess
input_size = (416, 416)
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, input_size)
img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
img_tensor = img_tensor.unsqueeze(0).to(device)

# Run inference
with torch.no_grad():
    outputs = model(img_tensor)

print(f"\nRaw output shape: {outputs.shape}")

# Check if model has postprocess method
print(f"\nModel attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")

# Try to use postprocess if available
if hasattr(model, 'postprocess'):
    print("\nModel has postprocess method!")
    # Most YOLOX models need: outputs, num_classes, conf_thre, nms_thre
    processed = model.postprocess(outputs, num_classes=80, conf_thre=0.01, nms_thre=0.45)
    print(f"Processed output type: {type(processed)}")
    if processed and len(processed) > 0:
        result = processed[0]
        if result is not None:
            print(f"Detections shape: {result.shape}")
            print(f"Number of detections: {len(result)}")
            print(f"First detection: {result[0]}")
