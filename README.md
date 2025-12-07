# AICounter - Object Counting Application

A Python implementation of the AICounter object counting system based on the PRD requirements.

## Features

- **Capture First, Count Second**: Static image analysis (no real-time video)
- **Zero-Shot Object Detection**: Counts objects without pre-training
- **Hybrid Vocabulary System**: 
  - Dynamic LIFO list for recently used items
  - Static base vocabulary for common objects
- **Main Object Algorithm**: Uses `Score = Area × Count` formula
- **Auto-Detection & Precision Mode**: 
  - Auto: Detects from vocabulary automatically
  - Manual: User specifies exact object name
- **Result Burning**: Overlays bounding boxes and count on images
- **History Management**: Saves and retrieves past counting records

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Count Objects in an Image

**Auto-detection mode:**
```bash
python object_counter.py count -i /path/to/image.jpg
```

**Precision mode (manual object specification):**
```bash
python object_counter.py count -i /path/to/image.jpg -o "cookie"
```

### View History

```bash
python object_counter.py history
```

## Architecture

Based on the PRD "Free Stack" architecture:

1. **Stage 1: The Proposer (YOLOX-Nano)**
   - Real YOLOX-Nano model for class-agnostic object detection
   - Finds *where* objects are located
   - Loaded from Megvii-BaseDetection/YOLOX torch hub
   - Fallback to selective search if model unavailable

2. **Stage 2: The Classifier (TinyCLIP)**
   - Real TinyCLIP model (wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M)
   - Lightweight zero-shot classification optimized for edge devices
   - Compares image crops against text vocabulary
   - Uses vision-language embeddings for matching

3. **Stage 3: The Logic Layer**
   - Applies `Area × Count` heuristic to determine main object
   - Manages LIFO vocabulary updates
   - Generates overlay images

## Output

- Overlay images saved to `./output/` directory
- History saved to `count_history.json`
- Each result includes:
  - Object name
  - Count
  - Timestamp
  - Path to overlay image

## Examples

```bash
# Count cookies in an image
python object_counter.py count -i bakery_tray.jpg

# Count bolts with precision mode
python object_counter.py count -i hardware.jpg -o "bolt"

# View all past counting records
python object_counter.py history
```

## Implementation Notes

This Python implementation uses real AI models:
- **YOLOX-Nano** for object detection (loaded from torch hub)
- **TinyCLIP** (wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M) for zero-shot classification

The production iOS app would:
- Convert YOLOX-Nano to CoreML format
- Convert TinyCLIP to CoreML format for on-device inference
- Add iOS-specific UI with camera integration
- Use Core Data for history persistence
- Implement PHPickerViewController for photo library access

### GPU Acceleration
The models automatically use CUDA if available, otherwise CPU. For best performance, use a GPU-enabled machine.

## Requirements Compliance

✓ FR-01: Dual Input Support (Camera simulation via file input)  
✓ FR-02: "Capture First, Count Second" workflow  
✓ FR-03: Single "Main" Object enforced  
✓ FR-04: Main Object Algorithm (Area × Count)  
✓ FR-05: Auto-Detection with Hybrid Vocabulary  
✓ FR-06: Manual Precision mode  
✓ FR-07: Result Burning (overlay generation)  
✓ FR-08: Data Saving (JSON persistence)  
✓ FR-09: History management  
✓ NFR-01: Offline operation (no network calls)  

## License

Apache 2.0 (YOLOX-Nano compatible)  
MIT (TinyCLIP compatible)
