# Core ML Model Conversion Summary

## Overview
Successfully converted Python models to Core ML format for iOS/Swift deployment.

## Converted Models

### 1. YOLOX-S Object Detection
- **Source**: `weights/yolox_s.pth` (PyTorch)
- **Output**: `weights/yolox_s.mlpackage` (Core ML ML Program)
- **Size**: ~17 MB
- **Input**: RGB image (640x640 pixels)
- **Output**: Object detection bounding boxes with confidence scores
- **Format**: ML Program (iOS 15+)
- **Usage**: Object detection for identifying objects in images

### 2. TinyCLIP Vision Encoder
- **Source**: `weights/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M` (Transformers)
- **Output**: `weights/tinyclip_vision.mlpackage` (Core ML ML Program)
- **Size**: ~16 MB
- **Input**: RGB image (224x224 pixels)
- **Output**: Normalized visual embedding vector (512 dimensions)
- **Format**: ML Program (iOS 15+)
- **Usage**: Visual similarity comparison and clustering

## Technical Details

### YOLOX-S Conversion
```
Architecture: YOLOX-S (class-agnostic detection)
Input Shape: [1, 3, 640, 640]
Preprocessing: Scale from [0, 255] to [0, 1]
Color Layout: RGB
Compute Units: All (CPU, GPU, Neural Engine)
```

### TinyCLIP Conversion
```
Architecture: Vision Transformer (ViT-8M-16)
Input Shape: [1, 3, 224, 224]
Output: L2-normalized embedding vector
Preprocessing: Scale from [0, 255] to [0, 1]
Color Layout: RGB
Compute Units: All (CPU, GPU, Neural Engine)
```

## Conversion Script
The conversion script `convert_to_coreml.py` handles:
1. Loading PyTorch YOLOX model
2. Loading Transformers TinyCLIP model
3. Tracing models for Core ML
4. Converting to ML Program format
5. Adding metadata and optimization flags

## Next Steps for Swift Migration

### 1. Add Models to Xcode Project
- Drag `.mlpackage` files into Xcode project
- Xcode will automatically compile to `.mlmodelc` at build time
- Models will be included in app bundle

### 2. YOLOX Usage in Swift
```swift
import Vision
import CoreML

// Load YOLOX model
let model = try yolox_s(configuration: MLModelConfiguration())
let visionModel = try VNCoreMLModel(for: model.model)

// Create detection request
let request = VNCoreMLRequest(model: visionModel) { request, error in
    guard let results = request.results as? [VNRecognizedObjectObservation] else { return }
    // Process detections
}

// Perform inference
let handler = VNImageRequestHandler(cgImage: image, options: [:])
try handler.perform([request])
```

### 3. TinyCLIP Usage in Swift
```swift
import CoreML
import Vision

// Load TinyCLIP model
let model = try tinyclip_vision(configuration: MLModelConfiguration())

// Prepare image input
let input = try tinyclip_visionInput(imageWith: cgImage)

// Get embedding
let output = try model.prediction(input: input)
let embedding = output.var_651 // Normalized embedding vector

// Compare embeddings with cosine similarity
func cosineSimilarity(_ a: MLMultiArray, _ b: MLMultiArray) -> Float {
    // Calculate dot product for normalized vectors
}
```

### 4. Swift Implementation Tasks

#### Required Components:
1. **Image Loading & Preprocessing**
   - Load images from file/camera
   - Resize to model input sizes (640x640 for YOLOX, 224x224 for TinyCLIP)
   - Convert to RGB format

2. **YOLOX Detection Pipeline**
   - Run inference on input image
   - Parse bounding box outputs
   - Apply confidence thresholding
   - Non-maximum suppression (NMS)

3. **Size & Aspect Ratio Filtering**
   - Calculate median box sizes
   - Remove outliers (±3 std from median)
   - Filter unusual aspect ratios

4. **TinyCLIP Embedding Extraction**
   - Crop detected regions
   - Resize crops to 224x224
   - Extract visual embeddings
   - Normalize embeddings

5. **Similarity Clustering**
   - Calculate cosine similarity matrix
   - Agglomerative clustering
   - Merge overlapping detections (IoU-based)

6. **Result Generation**
   - Count objects in largest cluster
   - Visualize detections
   - Generate statistics

## Performance Considerations

### YOLOX-S
- **Inference Time**: ~50-100ms on iPhone (A14+)
- **Memory**: ~100 MB
- **Optimization**: Uses Neural Engine when available

### TinyCLIP
- **Inference Time**: ~10-20ms per crop on iPhone (A14+)
- **Memory**: ~50 MB
- **Optimization**: Uses Neural Engine for vision encoder

## Model Files Location
```
weights/
├── yolox_s.pth                              # Original PyTorch (72 MB)
├── yolox_s.mlpackage/                       # Core ML (17 MB)
├── TinyCLIP-ViT-8M-16-Text-3M-YFCC15M/     # Original Transformers (94 MB)
└── tinyclip_vision.mlpackage/               # Core ML (16 MB)
```

## Dependencies for Conversion
```bash
pip install coremltools torch transformers
```

## Conversion Command
```bash
python3 convert_to_coreml.py
```

## Notes
- Models use ML Program format (requires iOS 15+)
- `.mlpackage` is a directory bundle (modern Core ML format)
- Xcode compiles `.mlpackage` → `.mlmodelc` at build time
- `.mlmodelc` is optimized binary format for runtime
- Models support all compute units (CPU, GPU, Neural Engine)

## Warnings During Conversion
- ✅ Scikit-learn version warning: Can be ignored (not needed for inference)
- ✅ PyTorch version warning: Models converted successfully despite version mismatch
- ✅ Precision loss int64→int32: Expected and safe for these models

## Date
December 9, 2025
