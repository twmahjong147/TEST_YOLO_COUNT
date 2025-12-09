# Feature Specification: iOS AICounter Migration

**Feature Branch**: `001-ios-migration`  
**Created**: December 9, 2025  
**Status**: Draft  
**Input**: User description: "Create a feature specification for migrating the Python-based count_objects_yolox.py application to a native iOS AICounter app."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Camera-Based Object Counting (Priority: P1)

A user launches the iOS app, points their iPhone camera at a collection of similar objects (bottles, boxes, coins, etc.), captures a photo, and receives an accurate count of the main object type within 2 seconds.

**Why this priority**: This is the core value proposition - users need a simple, fast way to count objects without manual work. This delivers immediate, tangible value and represents the minimum viable product.

**Independent Test**: Can be fully tested by opening the app, capturing a photo of 10 identical bottles, and verifying the count displays "10" within 2 seconds. Success means the user gets an accurate count without any technical knowledge.

**Acceptance Scenarios**:

1. **Given** the app is launched for the first time, **When** the user grants camera permission and taps "Capture Photo", **Then** the camera viewfinder opens with clean view (no real-time overlays)
2. **Given** the camera is active with 15 similar objects in frame, **When** the user taps the shutter button, **Then** the photo is captured and displayed for review
3. **Given** a captured photo shows 15 bottles, **When** the user taps "Count Objects", **Then** processing completes in < 2 seconds and displays "Count: 15"
4. **Given** processing is in progress, **When** the user views the screen, **Then** a progress indicator shows stages (Detecting → Clustering → Complete)
5. **Given** the count result is displayed, **When** the user views the number, **Then** only the main cluster count is shown (not multiple categories)

---

### User Story 2 - Photo Library Processing (Priority: P1)

A user selects an existing photo from their library showing multiple similar objects, processes it, and gets an accurate count without needing to retake the photo.

**Why this priority**: Users often have existing photos they want to analyze. This expands the utility beyond live capture and enables batch processing workflows in the future.

**Independent Test**: Can be tested by selecting a pre-existing photo of 20 coins from the library and verifying the app correctly identifies and counts them, delivering the same accuracy as camera capture.

**Acceptance Scenarios**:

1. **Given** the main screen is displayed, **When** the user taps "Select from Library", **Then** the iOS photo picker (PHPickerViewController) opens
2. **Given** the photo picker is open, **When** the user selects a JPEG/PNG/HEIC image of boxes, **Then** the image displays in the app with correct aspect ratio
3. **Given** a selected image shows 30 small screws, **When** the user taps "Count Objects", **Then** processing handles high object counts (30+) accurately
4. **Given** a selected image is 4K resolution, **When** processing begins, **Then** the app resizes appropriately without crashing or exceeding 200MB memory

---

### User Story 3 - Review Counting History (Priority: P2)

A user wants to review their past counting sessions to track inventory or reference previous work, viewing thumbnails and counts from the last 100 sessions.

**Why this priority**: Historical data enables workflows like inventory tracking, progress monitoring, and record-keeping. This transforms the app from a one-off tool to a productivity solution.

**Independent Test**: Can be tested by performing 5 counting operations, then opening the History view and verifying all 5 entries appear with thumbnails (cropped object images), counts, and timestamps. Deleting one entry should remove it permanently.

**Acceptance Scenarios**:

1. **Given** 3 counting sessions have been completed, **When** the user taps the history icon (📋), **Then** the History view opens showing 3 entries in reverse chronological order
2. **Given** the History view displays entries, **When** the user views each item, **Then** each shows a 200×200px thumbnail (cropped from main cluster), count number, and timestamp
3. **Given** an entry shows "Count: 25, Dec 9 2:30 PM", **When** the user taps it, **Then** full details are displayed (count, parameters used, timestamp)
4. **Given** 110 counting sessions exist, **When** the app opens, **Then** Core Data auto-deletes the 10 oldest entries, keeping exactly 100
5. **Given** the History view is open, **When** the user swipes left on an entry and taps Delete, **Then** the entry is removed from Core Data and UI
6. **Given** 20 history entries exist, **When** the user taps "Clear All History", **Then** a confirmation dialog appears, and upon confirmation, all entries are deleted

---

### User Story 4 - Adjust Detection Settings (Priority: P3)

A power user wants to fine-tune detection sensitivity for different scenarios (e.g., tighter clustering for very similar objects, or looser for varied items).

**Why this priority**: Different use cases require different sensitivity levels. Warehouse counting might need strict similarity, while mixed inventory needs flexibility. This enables advanced users to optimize accuracy for their specific needs.

**Independent Test**: Can be tested by processing the same image twice - once with default settings (0.80 similarity) showing 15 objects, then with strict settings (0.90 similarity) showing 12 objects (tighter clusters). Both results should be valid but reflect the threshold difference.

**Acceptance Scenarios**:

1. **Given** the main screen is displayed, **When** the user taps the settings icon (⚙️), **Then** the Settings screen opens
2. **Given** the Settings screen is open, **When** the user adjusts the "Confidence Threshold" slider to 0.40, **Then** the value displays "0.40" in real-time
3. **Given** confidence is set to 0.40, **When** the user processes an image, **Then** more detections are included (lower threshold = more lenient)
4. **Given** the Settings screen is open, **When** the user adjusts "Similarity Threshold" from 0.80 to 0.90, **Then** stricter clustering is applied (only very similar objects grouped)
5. **Given** custom thresholds are set, **When** the user taps "Reset to Defaults", **Then** all values return to: confidence=0.25, similarity=0.80
6. **Given** custom settings are used during counting, **When** a session is saved to history, **Then** the parameters (confidence, similarity) are stored in Core Data

---

### User Story 5 - Visualize Detection Overlay (Priority: P3)

A user wants to see which objects were detected and verify the AI counted correctly by viewing bounding boxes overlaid on the image.

**Why this priority**: Visual verification builds trust and helps users understand why certain objects were counted or excluded. Essential for debugging edge cases and user confidence.

**Independent Test**: Can be tested by processing an image with 10 bottles and 5 cans, then viewing the overlay showing 10 green boxes (main cluster - bottles) and 5 blue boxes (secondary objects - cans), confirming visual differentiation works correctly.

**Acceptance Scenarios**:

1. **Given** counting is complete with result "Count: 15", **When** the user taps "Show Details" or the result area, **Then** the original image displays with bounding boxes overlaid
2. **Given** the overlay is displayed, **When** the user views the image, **Then** objects in the main cluster (largest group) appear with green boxes, other objects with blue boxes
3. **Given** bounding boxes are shown, **When** the user taps a specific box, **Then** confidence score and cluster ID are displayed for that detection
4. **Given** the annotated image is displayed, **When** the user pinches to zoom, **Then** the image zooms smoothly with boxes scaling appropriately
5. **Given** 50+ objects are detected, **When** overlay is shown, **Then** performance remains smooth (60fps) with proper rendering optimization

---

### Edge Cases

- What happens when **no objects are detected** in an image? → Display "No objects detected" message with suggestion to adjust lighting/angle
- What happens when **processing fails** due to memory constraints? → Graceful error message, suggest closing other apps or using lower resolution image
- What happens when **camera permission is denied**? → Show alert explaining why camera access is needed with button to open Settings
- What happens when **photo library permission is denied**? → Disable "Select from Library" button and show explanation when tapped
- What happens when **user captures completely dark image**? → Detection returns 0 objects, suggest better lighting
- What happens when **objects are highly overlapping** (>80% IoU)? → IoU merging algorithm keeps highest confidence detection and suppresses duplicates
- What happens when **image contains 100+ objects**? → Processing may take 3-5 seconds (acceptable for edge case), memory optimization prevents crash
- What happens when **user rotates device during capture**? → Camera view adjusts orientation, captured image maintains correct orientation metadata
- What happens when **app is backgrounded during processing**? → Processing pauses, resumes when app returns to foreground (iOS background task handling)
- What happens when **Core Data storage reaches 100 entries**? → Auto-delete oldest entry before adding new one (FIFO queue behavior)

## Requirements *(mandatory)*

### Functional Requirements

**Core Counting Pipeline:**

- **FR-001**: System MUST implement YOLOX-S object detection with class-agnostic mode using CoreML (input: 640×640, confidence threshold: 0.001, NMS threshold: 0.65)
- **FR-002**: System MUST extract visual embeddings using TinyCLIP vision encoder (256-dimensional embeddings, L2-normalized)
- **FR-003**: System MUST perform size outlier filtering using median ± 1.0 standard deviations on bounding box areas
- **FR-004**: System MUST perform aspect ratio filtering using median ± 0.5 standard deviations on width/height ratios
- **FR-005**: System MUST perform area consistency filtering using median ± 1.0 standard deviations on crop areas before embedding extraction
- **FR-006**: System MUST cluster embeddings using agglomerative clustering (cosine distance, average linkage, distance threshold = 1 - similarity_threshold)
- **FR-007**: System MUST merge overlapping detections using IoU-based NMS (threshold: 0.5, keep highest confidence)
- **FR-008**: System MUST identify the largest cluster and return its count as the final result

**Image Input & Capture:**

- **FR-009**: System MUST support camera capture using AVCaptureSession with clean viewfinder (no real-time detection overlays)
- **FR-010**: System MUST support photo library selection using PHPickerViewController for JPEG, PNG, and HEIC formats
- **FR-011**: System MUST handle images from 100×100 to 4K resolution (3840×2160) with automatic resizing
- **FR-012**: System MUST trigger object detection ONLY after photo capture or library selection (no real-time video frame processing)
- **FR-013**: Users MUST be able to review captured/selected images before processing
- **FR-014**: System MUST request and handle camera and photo library permissions with appropriate error messages

**Result Display:**

- **FR-015**: System MUST display ONLY the main cluster count (not total detections or number of clusters)
- **FR-016**: System MUST show processing progress with stage indicators (Detecting → Clustering → Complete)
- **FR-017**: System MUST complete processing in < 2 seconds on iPhone 12+ for images with < 50 objects
- **FR-018**: System MUST display error messages clearly for failure cases (no objects, processing error, permission denied)

**History & Persistence:**

- **FR-019**: System MUST save counting sessions to Core Data with: UUID, thumbnailData (200×200 JPEG), count (Int), timestamp, confidenceThreshold, similarityThreshold, isFavorite (Bool)
- **FR-020**: System MUST generate thumbnails by cropping one representative object from the main cluster (not full scene)
- **FR-021**: System MUST compress thumbnails to JPEG format (quality: 80%) targeting ~10KB per thumbnail
- **FR-022**: System MUST maintain maximum 100 history entries using FIFO deletion (auto-delete oldest when adding 101st)
- **FR-023**: Users MUST be able to view history in reverse chronological order with thumbnail preview
- **FR-024**: Users MUST be able to delete individual history entries via swipe gesture
- **FR-025**: Users MUST be able to clear all history with confirmation dialog
- **FR-026**: System MUST persist history across app launches using Core Data

**Settings & Configuration:**

- **FR-027**: Users MUST be able to adjust confidence threshold (range: 0.1 - 0.9, default: 0.001)
- **FR-028**: Users MUST be able to adjust similarity threshold (range: 0.6 - 0.95, default: 0.80)
- **FR-029**: Users MUST be able to toggle size outlier filtering (on/off, default: on)
- **FR-030**: Users MUST be able to toggle aspect ratio filtering (on/off, default: on)
- **FR-031**: Users MUST be able to reset all settings to defaults with single button tap
- **FR-032**: System MUST persist user settings in UserDefaults across app launches

**Visual Overlay (Post-MVP):**

- **FR-033**: System MUST draw bounding boxes on detected objects (main cluster: green, others: blue)
- **FR-034**: System MUST display confidence scores on bounding boxes
- **FR-035**: Users MUST be able to zoom and pan the annotated image
- **FR-036**: System MUST render overlay at 60fps for smooth interaction

### Key Entities

- **Detection**: Represents a single detected object with bbox (CGRect), confidence (Float), cluster ID (Int), area (Float)
- **CountResult**: Represents final counting output with count (Int - main cluster only), detections ([Detection]), largestClusterId (Int), processingTime (TimeInterval)
- **CountingSession** (Core Data): Persistent history entry with id (UUID), thumbnailData (Data), count (Int16), timestamp (Date), confidenceThreshold (Float), similarityThreshold (Float), isFavorite (Bool)
- **Embedding**: 256-dimensional float array representing visual features extracted by TinyCLIP
- **ProcessingStage**: Enum tracking pipeline progress (detecting, extracting, clustering, merging, complete, error)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can complete photo capture, object counting, and result viewing in under 10 seconds from app launch to final count
- **SC-002**: Processing completes in under 2 seconds on iPhone 12 or newer for images containing 20-50 objects
- **SC-003**: System achieves at least 95% counting accuracy compared to manual count for clear images with distinct similar objects
- **SC-004**: System achieves at least 90% counting accuracy for challenging scenarios including overlapping objects, mixed lighting, and cluttered backgrounds
- **SC-005**: App maintains memory usage below 150MB during active processing and below 50MB when idle
- **SC-006**: App crashes occur in less than 0.1% of user sessions across all supported devices
- **SC-007**: Camera capture and live preview operate at minimum 30fps without visible lag or stuttering
- **SC-008**: History view loads 100 entries with thumbnails in under 500ms
- **SC-009**: App storage remains under 20MB for 100 history entries with average 200KB per entry including thumbnail
- **SC-010**: At least 95% of users successfully complete their first counting operation without errors or confusion
- **SC-011**: Processing battery consumption is under 0.1% per image processed
- **SC-012**: Users can perform 10 consecutive counting operations consuming less than 2% total battery

### User Satisfaction Metrics

- **SC-013**: At least 90% of beta testers rate the app as easy to use or very easy to use
- **SC-014**: At least 85% of users successfully understand that the displayed count represents the main object type rather than total objects
- **SC-015**: App Store rating achieves 4.5 or higher stars within 3 months of launch
- **SC-016**: At least 40% of users return to the app after 7 days

## Assumptions

1. **Model Availability**: YOLOX-S and TinyCLIP models can be successfully converted to CoreML format with acceptable accuracy loss (<2%)
2. **Model Size**: Combined CoreML models (.mlpackage) total ~33MB, acceptable for App Store distribution
3. **Performance**: iPhone X (A11 Bionic) and newer have sufficient Neural Engine capability for <2s processing
4. **Clustering Library**: Swift implementation of agglomerative clustering (cosine distance, average linkage) can be implemented or imported without GPL dependencies
5. **Image Quality**: Users capture images with reasonable lighting and focus for detection to work effectively
6. **Object Separation**: Most use cases involve objects with some spatial separation (not completely stacked/hidden)
7. **Network**: App operates fully offline; no internet connection required for any functionality
8. **Storage**: Users have at least 100MB free storage for app installation and history data
9. **Permissions**: Most users will grant camera and photo library permissions when prompted with clear explanation
10. **UI Framework**: SwiftUI is mature enough for production use on iOS 15+ without significant limitations
11. **Background Processing**: iOS allows sufficient background execution time for completing in-progress counting when app is backgrounded
12. **Core Data**: Core Data schema migrations work reliably for future feature additions without data loss

## Dependencies

- **iOS Platform**: iOS 15.0+ (AVFoundation, CoreML, Vision, Core Data, SwiftUI)
- **Hardware**: iPhone X or newer (A11 Bionic chip minimum for Neural Engine support)
- **CoreML Models**: 
  - YOLOX-S (yolox_s.mlpackage, ~17MB) - object detection
  - TinyCLIP Vision (tinyclip_vision.mlpackage, ~16MB) - embedding extraction
- **Apple Frameworks**:
  - AVFoundation (camera capture)
  - CoreML (model inference)
  - Vision (image preprocessing)
  - Core Data (persistence)
  - SwiftUI (UI framework)
  - Accelerate (vector operations, cosine similarity)
  - PhotosUI (PHPickerViewController)
- **Third-party**: None (agglomerative clustering implemented in Swift, no external dependencies)

## Constraints

1. **No Real-Time Detection**: Camera viewfinder must NOT run inference on live frames (battery optimization)
2. **No Cloud Services**: All processing must happen on-device (privacy requirement)
3. **Standard Xcode Structure**: Must use standard project structure, NOT Swift Package Manager for app architecture (per project constitution)
4. **Memory Budget**: Peak memory usage must stay below 200MB to avoid termination on older devices
5. **Processing Time**: Total pipeline must complete in <3s maximum on iPhone 12+ (target: 2s)
6. **Model Size**: Total app size should stay under 50MB for initial download (over-the-air limit)
7. **History Limit**: Maximum 100 entries stored (enforced cleanup to prevent unbounded growth)
8. **Thumbnail Storage**: Thumbnail size limited to 200×200px to balance quality and storage
9. **Image Resolution**: Input images resized to 640×640 for YOLOX, 224×224 for TinyCLIP (model requirements)
10. **Clustering Algorithm**: Must use agglomerative clustering with cosine distance and average linkage to match Python implementation
11. **iOS Version Support**: Must support iOS 15.0+ for maximum device compatibility
12. **Offline Operation**: App must function without internet connection (no analytics, no cloud models)

## Migration Mapping (Python to Swift)

| Python Component | Swift Component | Implementation Notes |
|-----------------|-----------------|----------------------|
| `count_objects_yolox.py::count_objects_with_yolox()` | `AICounter.swift::count(image:)` | Main orchestration layer, matches pipeline stages |
| `YOLOXDetector` (ONNX Runtime) | `YOLOXDetector.swift` (CoreML) | Replace ONNX with CoreML inference, same preprocessing |
| `TinyCLIPEmbedder::get_visual_embedding()` | `TinyCLIPEmbedder.swift::getEmbedding()` | CoreML model for CLIP vision encoder, L2 normalization |
| `remove_size_outliers()` | `OutlierFilter.swift::filterBySize()` | Median ± std deviation logic, matches exact algorithm |
| `remove_aspect_ratio_outliers()` | `OutlierFilter.swift::filterByAspectRatio()` | Same statistical filtering approach |
| `filter_by_area_consistency()` | `OutlierFilter.swift::filterByAreaConsistency()` | Pre-clustering crop filtering |
| `cluster_by_similarity()` (sklearn) | `SimilarityClusterer.swift::cluster()` | Implement agglomerative clustering in Swift (cosine, average linkage) |
| `calculate_iou()` | `GeometryUtils.swift::calculateIoU()` | Standard IoU calculation, identical formula |
| `filter_contained_across()` | `NMSFilter.swift::mergeOverlapping()` | IoU-based merging, keep highest confidence |
| `visualize_detections_custom()` (OpenCV) | `OverlayRenderer.swift::renderBoundingBoxes()` | Core Graphics for drawing boxes |
| NumPy array operations | Swift Array + Accelerate framework | Use vDSP for vector operations |
| PIL Image processing | Core Graphics / Core Image | Native image manipulation |
| Torch tensor operations | MLMultiArray / Array | CoreML output handling |

## Testing Strategy

### Accuracy Validation
- **Test Set**: Create 50 test images matching Python output exactly
- **Acceptance**: Swift implementation must produce identical counts (±1 for edge cases) compared to Python
- **Metrics**: Per-stage comparison (detection count, filtering results, cluster assignments, final count)

### Performance Benchmarks
- **Detection Stage**: <100ms per image (YOLOX inference)
- **Embedding Stage**: <20ms per crop (TinyCLIP inference)
- **Clustering Stage**: <200ms for 50 embeddings
- **Total Pipeline**: <2s end-to-end for 50-object image

### Edge Case Coverage
- Empty images (no detections)
- Single object detection
- High object counts (100+)
- Overlapping objects (>80% IoU)
- Mixed object types (verify largest cluster selection)
- Extreme lighting conditions
- Low resolution images (<640px)
- High resolution images (4K)

### History & Persistence Tests
- Add 101 entries, verify FIFO deletion
- App termination during save operation
- Core Data migration simulation
- Thumbnail generation quality validation
- History view performance with 100 entries
- Concurrent access handling

## Out of Scope

1. **Real-time video counting**: Only static images processed (battery optimization)
2. **Multi-model support**: YOLOX-S only (no model selection)
3. **Custom model training**: Pre-trained models only
4. **Cloud sync**: All data local only
5. **Batch processing**: Single image at a time for MVP (future feature)
6. **Export functionality**: No CSV/PDF export in MVP (future feature)
7. **AR overlays**: No augmented reality features
8. **Annotation editing**: Users cannot manually adjust bounding boxes
9. **Video input**: No video file processing
10. **iPad optimization**: iPhone-first design (iPad support post-MVP)
11. **Accessibility features**: Standard iOS accessibility (no custom features in MVP)
12. **Internationalization**: English only for MVP

## Success Validation

### Pre-Launch Checklist
- [ ] All 32 functional requirements implemented and tested
- [ ] 12 success criteria metrics measured and meet targets
- [ ] 50-image test set shows ≥95% accuracy match with Python
- [ ] Performance benchmarks met on iPhone 12 and iPhone X
- [ ] Memory usage stays below 150MB during processing
- [ ] Zero crashes in 100-operation stress test
- [ ] Camera permissions handled gracefully
- [ ] Core Data migrations tested
- [ ] History cleanup (100-entry limit) verified
- [ ] Thumbnail generation quality validated
- [ ] App Store guidelines compliance verified
- [ ] Privacy policy drafted (on-device processing)
- [ ] TestFlight beta with 20+ users completed

### Post-Launch Monitoring
- Track crash rate (target: <0.1%)
- Monitor performance metrics (processing time, memory)
- Collect accuracy feedback from users
- Measure D1, D7, D30 retention rates
- Monitor App Store ratings and reviews
- Track feature usage (camera vs library, settings adjustments)
