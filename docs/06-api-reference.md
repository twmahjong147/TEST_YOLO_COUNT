# API Reference

This document provides an overview of the main public APIs and methods in the Object Counting System.

## Core Classes and Methods

### CameraManager
- `startSession()`: Initialize camera session.
- `capturePhoto() -> UIImage`: Capture a still image.
- `stopSession()`: Release camera resources.
- `toggleFlash()`: Toggle flash mode.
- `focusAt(point: CGPoint)`: Set manual focus.

### AICounter
- `count(image: UIImage, confThreshold: Float, similarityThreshold: Float) -> CountResult`
  - Orchestrates the full counting pipeline.
- `filterSizeOutliers(_ detections: [Detection]) -> [Detection]`
  - Removes detections that are size outliers.
- `cropImage(_ image: UIImage, to rect: CGRect) -> CGImage?`
  - Crops a region from the image.

### YOLOXDetector
- `detect(image: UIImage, confThreshold: Float) -> [Detection]`
  - Runs YOLOX-S model inference and returns detections.
- `parseYOLOXOutput(_ output: MLMultiArray) -> [Detection]`
  - Parses model output into detection objects.

### TinyCLIPEmbedder
- `extractEmbedding(image: UIImage) -> [Float]`
  - Runs TinyCLIP model to get a 256-dim normalized embedding.

### SimilarityClusterer
- `cluster(embeddings: [[Float]]) -> [[Int]]`
  - Clusters embeddings using agglomerative clustering.

### HistoryManager
- `save(result: CountResult)`: Persist a counting result.
- `fetchHistory() -> [CountResult]`: Retrieve past results.

## Data Structures

### Detection
- `boundingBox: CGRect`
- `confidence: Float`
- `embedding: [Float]?`

### CountResult
- `count: Int`
- `image: UIImage`
- `timestamp: Date`
- `details: [Detection]`

### ProcessingError
- `message: String`
- `code: Int`

## Notes
- All model inferences are performed asynchronously.
- Errors are reported via `ProcessingError`.

---
