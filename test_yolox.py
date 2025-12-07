#!/usr/bin/env python3
"""Debug script to test YOLOX detection"""

import cv2
import torch
import numpy as np

# Load image
image = cv2.imread('sample1.JPG')
h, w = image.shape[:2]
print(f"Image size: {w}x{h}")

# Load YOLOX model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = torch.hub.load('Megvii-BaseDetection/YOLOX', 'yolox_nano', pretrained=True)
model.to(device)
model.eval()

# Preprocess
input_size = (416, 416)
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, input_size)
img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
img_tensor = img_tensor.unsqueeze(0).to(device)

print(f"Input tensor shape: {img_tensor.shape}")

# Run inference
with torch.no_grad():
    outputs = model(img_tensor)

print(f"\nOutput type: {type(outputs)}")
if isinstance(outputs, torch.Tensor):
    print(f"Output shape: {outputs.shape}")
    output = outputs
    print(f"Output data sample:\n{output[:5]}")
    
    if hasattr(output, 'shape') and len(output.shape) > 1:
        print(f"Output[0] data:\n{output[:5]}")  # First 5 detections
        print(f"\nConfidence scores (column 4): {output[:10, 4]}")
        
        # Try different confidence thresholds
        for thresh in [0.5, 0.3, 0.25, 0.1, 0.05, 0.01]:
            mask = output[:, 4] > thresh
            count = mask.sum().item()
            print(f"Threshold {thresh}: {count} detections")
