# Additional Technical Notes

This document covers other crucial technical aspects for understanding the codebase, beyond architecture, design, and API.

## Testing Strategy

### Unit Tests
- YOLOXDetector output parsing
- TinyCLIP embedding extraction
- Cosine similarity calculation
- Clustering algorithm correctness
- Outlier filtering logic
- Core Data CRUD operations
- Thumbnail generation and compression
- History entry limit enforcement

### Integration Tests
- Full counting pipeline end-to-end
- Model loading and inference
- Image preprocessing
- Error handling paths

### UI Tests
- Camera capture flow
- Image selection flow
- Count button interaction
- Result display
- History view navigation
- History entry deletion
- Settings adjustments

### Performance Tests
- Processing time under target (< 2s)
- Memory usage within limits (< 200 MB)
- UI responsiveness during processing

## Model Conversion
- Conversion scripts (`convert_to_coreml.py`) handle tracing, Core ML conversion, and metadata.
- Both YOLOX-S and TinyCLIP models are converted to `.mlpackage` format for iOS 15+.
- Preprocessing and normalization are embedded in the Core ML models.

## Glossary
- **YOLOX:** Object detection model (You Only Look Once X)
- **TinyCLIP:** Compact vision-language model for embeddings
- **Core ML:** Apple's machine learning framework
- **Neural Engine:** Dedicated ML accelerator in Apple chips
- **Embedding:** Vector representation of visual features
- **Cosine Similarity:** Measure of similarity between vectors
- **Agglomerative Clustering:** Bottom-up clustering algorithm

## References
- YOLOX paper: https://arxiv.org/abs/2107.08430
- TinyCLIP: https://github.com/wkcn/TinyCLIP
- Core ML: https://developer.apple.com/documentation/coreml
- Vision framework: https://developer.apple.com/documentation/vision

---
