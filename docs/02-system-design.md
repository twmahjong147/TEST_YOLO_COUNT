# System Design

This document details the design of the Object Counting System, focusing on the responsibilities, interactions, and data handled by each component.

## Main Components

### 1. AICounter (Main Controller)
- **Role:** Orchestrates the counting pipeline, manages model lifecycle, and handles errors.
- **Responsibilities:**
  - Receives images from the camera or gallery.
  - Initiates object detection, embedding extraction, and clustering.
  - Aggregates results and manages persistence.

### 2. YOLOXDetector
- **Role:** Object detection using YOLOX-S Core ML model.
- **Responsibilities:**
  - Loads and manages the YOLOX-S model.
  - Runs inference to detect objects in images.
  - Parses model output into bounding boxes.

### 3. TinyCLIPEmbedder
- **Role:** Visual embedding extraction using TinyCLIP Core ML model.
- **Responsibilities:**
  - Loads and manages the TinyCLIP model.
  - Crops detected objects from the image.
  - Extracts normalized embedding vectors for each object.

### 4. SimilarityClusterer
- **Role:** Clusters objects based on visual similarity.
- **Responsibilities:**
  - Receives embedding vectors.
  - Performs agglomerative clustering using cosine similarity.
  - Identifies the largest cluster as the main object group.

### 5. CameraManager
- **Role:** Camera capture and session management.
- **Responsibilities:**
  - Manages AVCaptureSession.
  - Captures still images.
  - Handles flash and focus controls.

### 6. HistoryManager
- **Role:** Persistence of counting results.
- **Responsibilities:**
  - Stores and retrieves past counting sessions.
  - Manages history limits and data integrity.

## Data Structures
- **Detection:** Represents a detected object (bounding box, confidence, etc.).
- **CountResult:** Aggregates the final count and related metadata.
- **ProcessingError:** Encapsulates error information for robust error handling.

## Error Handling
- All errors are captured and reported via `ProcessingError`.
- The system gracefully handles model loading, inference, and data persistence failures.

## Threading and Performance
- All model inferences are performed on background threads to keep the UI responsive.
- Core ML models leverage CPU, GPU, and Neural Engine automatically.

## Design Principles
- **Modularity:** Each component has a single responsibility.
- **Extensibility:** New models or clustering methods can be added with minimal changes.
- **Testability:** Unit and integration tests cover all major logic paths.

---
