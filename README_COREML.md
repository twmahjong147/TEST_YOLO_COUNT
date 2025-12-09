# Core ML Model Conversion - Quick Start

## ✅ Completed Tasks

1. **Created conversion script**: `convert_to_coreml.py`
2. **Converted YOLOX-S model**: `weights/yolox_s.mlpackage` (17 MB)
3. **Converted TinyCLIP vision model**: `weights/tinyclip_vision.mlpackage` (16 MB)
4. **Created documentation**:
   - `COREML_CONVERSION_SUMMARY.md` - Technical details about the conversion
   - `SWIFT_MIGRATION_GUIDE.md` - Complete Swift implementation guide

## 📦 Converted Models

### YOLOX-S (Object Detection)
```
Input:  RGB image, 640x640 pixels
Output: [1, 8400, 85] predictions
        - 8400 potential detections
        - 85 features per detection (bbox + confidence + 80 classes)
Size:   17 MB
```

### TinyCLIP Vision (Embeddings)
```
Input:  RGB image, 224x224 pixels
Output: [1, 256] normalized embedding vector
Size:   16 MB
```

## 🚀 Usage

### Run Conversion (Already Done!)
```bash
python3 convert_to_coreml.py
```

### Add to Xcode Project
1. Drag `.mlpackage` files into your Xcode project
2. Xcode will auto-compile to `.mlmodelc` at build time
3. Import and use in Swift code (see SWIFT_MIGRATION_GUIDE.md)

### Swift Quick Start
```swift
import CoreML

// Load models
let yolox = try yolox_s(configuration: MLModelConfiguration())
let tinyclip = try tinyclip_vision(configuration: MLModelConfiguration())

// Use for object counting
let counter = try AICounter()
let result = try counter.count(image: cgImage)
print("Found \(result.count) objects")
```

## 📚 Documentation Files

1. **COREML_CONVERSION_SUMMARY.md**
   - Model specifications
   - Conversion details
   - Technical architecture

2. **SWIFT_MIGRATION_GUIDE.md**
   - Complete Swift implementation
   - YOLOXDetector class
   - TinyCLIPEmbedder class
   - SimilarityClusterer class
   - AICounter class
   - SwiftUI example

3. **MODEL_SIZE_EXPLANATION.md**
   - Why Core ML is 75-83% smaller than PyTorch
   - Detailed size breakdown
   - Optimization techniques

4. **convert_to_coreml.py**
   - Source code for conversion
   - Can re-run if needed
   - Handles both models

## 🎯 Next Steps for Swift Migration

1. **Create Xcode Project**
   - iOS App project
   - Minimum deployment target: iOS 15+

2. **Add Models**
   - Drag `yolox_s.mlpackage` to project
   - Drag `tinyclip_vision.mlpackage` to project
   - Ensure "Target Membership" is checked

3. **Implement Core Classes**
   - Copy code from `SWIFT_MIGRATION_GUIDE.md`
   - YOLOXDetector.swift
   - TinyCLIPEmbedder.swift
   - SimilarityClusterer.swift
   - AICounter.swift

4. **Create UI**
   - Image picker
   - Detection visualization
   - Count display

5. **Test & Optimize**
   - Verify accuracy matches Python
   - Profile performance
   - Optimize if needed

## 📊 Model Performance (Estimated)

### iPhone 12+ (A14 Bionic or newer)
- YOLOX inference: ~50-100ms
- TinyCLIP per crop: ~10-20ms
- Total processing: ~1-2 seconds for typical image

### iPad Pro (M1/M2)
- YOLOX inference: ~30-50ms
- TinyCLIP per crop: ~5-10ms
- Total processing: ~500ms-1s for typical image

## 🔧 Requirements

### For Conversion (Already Done)
- Python 3.8+
- PyTorch 2.0+
- coremltools 8.0+
- transformers 4.0+

### For Swift/iOS Development
- macOS with Xcode 15+
- iOS 15.0+ deployment target
- Core ML framework
- Vision framework
- Accelerate framework

## 📝 Model Details

### YOLOX-S Architecture
- Backbone: CSPDarknet-S
- Parameters: ~9M
- Input: 640×640 RGB
- Output: 8400 predictions with 85 features each
- Class-agnostic mode: Uses objectness score

### TinyCLIP Architecture
- Vision Encoder: ViT-8M-16
- Parameters: ~8M
- Input: 224×224 RGB
- Output: 256-dim normalized embedding
- Pre-trained on YFCC15M dataset

## ⚠️ Important Notes

1. **Model Format**: `.mlpackage` is the modern Core ML format (iOS 15+)
2. **Compilation**: Xcode auto-compiles to `.mlmodelc` at build time
3. **Compute Units**: Models use CPU, GPU, and Neural Engine automatically
4. **Memory**: Allocate ~150 MB for both models in memory
5. **Threading**: Run inference on background threads to keep UI responsive

## 🐛 Troubleshooting

### Model won't load in Swift
- Check minimum deployment target is iOS 15+
- Verify model is added to Target Membership
- Clean build folder and rebuild

### Slow performance
- Ensure device has A14 or newer chip
- Check compute units are set to `.all`
- Profile with Instruments to find bottlenecks

### Different results from Python
- Verify preprocessing matches (image scaling, normalization)
- Check coordinate system (Core ML uses normalized coordinates)
- Validate threshold values match Python

## 📅 Conversion Date
December 9, 2025

## ✨ Ready for Swift!
All models are converted and ready for iOS deployment. Follow the SWIFT_MIGRATION_GUIDE.md for implementation details.
