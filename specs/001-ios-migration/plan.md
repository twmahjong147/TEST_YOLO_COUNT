# Implementation Plan: iOS AICounter Migration

**Branch**: `001-ios-migration` | **Date**: December 9, 2025 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-ios-migration/spec.md`

**Note**: This plan provides comprehensive technical guidance for migrating the Python YOLOX object counting system to a native iOS application.

## Summary

Migrate the Python-based `count_objects_yolox.py` object counting system to a native iOS application using Swift, CoreML, and standard iOS frameworks. The app will implement the complete counting pipeline: YOLOX-S object detection → size/aspect ratio outlier filtering → TinyCLIP embedding extraction → agglomerative clustering → IoU-based merging → main cluster counting. Target architecture follows MVVM pattern with standard Xcode project structure (no SPM per constitution), supporting camera capture, photo library input, Core Data persistence with thumbnail generation (cropped objects from main cluster), and settings management.

## Technical Context

**Language/Version**: Swift 5.9+ (iOS 15.0+ deployment target)  
**Primary Dependencies**: 
- CoreML (ML model inference)
- Vision (image preprocessing, model integration)
- AVFoundation (camera capture)
- Core Data (persistence layer)
- SwiftUI (UI framework)
- Accelerate (vector operations, cosine similarity)
- PhotosUI (PHPickerViewController for photo library)

**Storage**: Core Data for counting history (max 100 entries, FIFO cleanup), UserDefaults for settings  
**Testing**: XCTest (unit, integration, UI tests), XCUITest (end-to-end UI flows)  
**Target Platform**: iOS 15.0+, iPhone X or newer (A11 Bionic minimum for Neural Engine)  
**Project Type**: Native iOS mobile app (single target, standard Xcode project structure)  
**Performance Goals**: 
- Total processing: <2s per image (iPhone 12+)
- YOLOX inference: <100ms
- TinyCLIP inference: <20ms per crop
- UI: 60fps minimum, 30fps camera preview
- Memory: <150MB during processing, <50MB idle

**Constraints**: 
- Offline-only operation (no internet required)
- Standard Xcode project structure (NO Swift Package Manager architecture per constitution)
- App size: <50MB (over-the-air limit)
- No real-time video inference (battery optimization)
- Maximum 100 history entries with auto-cleanup

**Scale/Scope**: 
- 8 primary screens (Main, Camera, History, Settings, Detail views)
- ~15 Swift classes/structs (core components)
- 2 CoreML models (~33MB total: YOLOX-S 17MB + TinyCLIP 16MB)
- Target: 1,000+ initial downloads, 40% D7 retention

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. Code Quality Principles
- ✅ **Clean code practices**: Component-based architecture with single responsibility
- ✅ **Function length**: All functions <50 lines (Swift extension methods for complex algorithms)
- ✅ **Naming**: Self-documenting (e.g., `YOLOXDetector`, `filterSizeOutliers`, `generateThumbnail`)
- ✅ **Nesting depth**: Max 3 levels (Swift guard statements prevent deep nesting)
- ✅ **Documentation**: Inline comments for ML algorithms (outlier filtering, clustering, IoU)
- ✅ **Platform practices**: Swift naming conventions, Codable for serialization, Combine for async

### II. Testing Standards
- ✅ **80% coverage**: Critical paths (detection, filtering, clustering, persistence)
- ✅ **TDD approach**: Unit tests before implementation for algorithms
- ✅ **Integration tests**: CoreML model loading, full pipeline, Core Data CRUD
- ✅ **Edge cases**: Empty images, single object, 100+ objects, overlapping detections
- ✅ **Mocking**: Protocol-based design for testability (DetectorProtocol, EmbedderProtocol)
- ✅ **ML performance**: YOLOX <100ms, TinyCLIP <20ms per crop benchmarks

### III. User Experience Consistency
- ✅ **iOS HIG compliance**: Native SwiftUI components, standard navigation patterns
- ✅ **Visual feedback**: Progress indicators during ML processing (3-stage: Detecting → Clustering → Complete)
- ✅ **Error handling**: Clear messages for permission denied, no objects detected, processing failures
- ✅ **Accessibility**: VoiceOver labels, Dynamic Type support, minimum contrast ratios
- ✅ **60fps animations**: SwiftUI transitions, smooth camera preview
- ✅ **Responsive design**: Adaptive layouts for different iPhone sizes

### IV. Performance Requirements
- ✅ **App launch**: <2s cold start (lazy model loading)
- ✅ **ML inference**: <100ms per frame (CoreML with Neural Engine)
- ✅ **Memory**: <150MB processing, <50MB idle (model unloading when backgrounded)
- ✅ **Battery efficiency**: <2% per 10 sessions (no real-time video inference)
- ✅ **Profiling**: Instruments Time Profiler for bottlenecks
- ✅ **Caching**: NSCache for processed embeddings during session

### V. Architecture Constraints
- ✅ **Standard Xcode structure**: Single .xcodeproj at root (NO Swift Package Manager)
- ✅ **Minimal dependencies**: Zero third-party libraries (all built-in frameworks)
- ✅ **Built-in frameworks**: CoreML, Vision, AVFoundation, Core Data, SwiftUI, Accelerate
- ⚠️ **Exception request**: Agglomerative clustering implemented in Swift (no external library available)

**Status**: ✅ PASSED - All constitution requirements met. No external dependencies required. Agglomerative clustering will be implemented directly in Swift using Accelerate framework for distance calculations.

## Project Structure

### Documentation (this feature)

```text
specs/001-ios-migration/
├── plan.md              # This file - comprehensive implementation plan
├── research.md          # Phase 0: Technology decisions, clustering algorithms, CoreML integration
├── data-model.md        # Phase 1: Core Data schema, Swift structs, model protocols
├── quickstart.md        # Phase 1: Developer setup, model conversion, first build
├── contracts/           # Phase 1: Component interfaces (protocols), API contracts
│   ├── detector-protocol.swift
│   ├── embedder-protocol.swift
│   ├── clusterer-protocol.swift
│   └── persistence-protocol.swift
└── tasks.md             # Phase 2: Implementation tasks (NOT created by this plan command)
```

### Source Code (iOS Application)

```text
AICounter/
├── AICounter.xcodeproj              # Standard Xcode project (NO SPM)
├── AICounter/
│   ├── App/
│   │   ├── AICounterApp.swift       # SwiftUI @main entry point
│   │   ├── AppDelegate.swift        # Lifecycle management
│   │   └── Info.plist               # Permissions, capabilities
│   │
│   ├── Models/
│   │   ├── CoreML/
│   │   │   ├── yolox_s.mlpackage    # YOLOX-S detection model (17MB)
│   │   │   └── tinyclip_vision.mlpackage  # TinyCLIP embedder (16MB)
│   │   │
│   │   ├── Detection.swift          # struct Detection { bbox, confidence, classId }
│   │   ├── CountResult.swift        # struct CountResult { count, detections, clusterId }
│   │   ├── ProcessingStage.swift    # enum: detecting, clustering, complete, error
│   │   └── CountingSession+CoreData.swift  # Core Data entity extensions
│   │
│   ├── Core/
│   │   ├── Detectors/
│   │   │   ├── DetectorProtocol.swift       # Protocol for detection engines
│   │   │   ├── YOLOXDetector.swift          # CoreML YOLOX-S wrapper
│   │   │   └── YOLOXOutputParser.swift      # Parse [1, 8400, 85] output
│   │   │
│   │   ├── Embedders/
│   │   │   ├── EmbedderProtocol.swift       # Protocol for embedding extraction
│   │   │   ├── TinyCLIPEmbedder.swift       # CoreML TinyCLIP wrapper
│   │   │   └── EmbeddingNormalizer.swift    # L2 normalization
│   │   │
│   │   ├── Clustering/
│   │   │   ├── ClustererProtocol.swift      # Protocol for clustering algorithms
│   │   │   ├── SimilarityClusterer.swift    # Agglomerative clustering (Swift impl)
│   │   │   ├── CosineSimilarity.swift       # Accelerate-based cosine distance
│   │   │   └── ClusterAnalyzer.swift        # Identify largest cluster
│   │   │
│   │   ├── Filtering/
│   │   │   ├── OutlierFilter.swift          # Size, aspect ratio, area filtering
│   │   │   ├── StatisticalUtils.swift       # Median, std deviation helpers
│   │   │   └── NMSFilter.swift              # IoU-based overlap merging
│   │   │
│   │   └── AICounter.swift          # Main orchestrator (pipeline coordinator)
│   │
│   ├── Services/
│   │   ├── Camera/
│   │   │   ├── CameraManager.swift          # AVCaptureSession management
│   │   │   ├── CameraPreviewLayer.swift     # SwiftUI camera preview
│   │   │   └── PhotoCaptureDelegate.swift   # Capture completion handling
│   │   │
│   │   ├── Persistence/
│   │   │   ├── PersistenceController.swift  # Core Data stack
│   │   │   ├── HistoryManager.swift         # CRUD operations, FIFO cleanup
│   │   │   ├── ThumbnailGenerator.swift     # Crop main object, resize, compress
│   │   │   └── CountingSession.xcdatamodeld # Core Data model definition
│   │   │
│   │   └── Settings/
│   │       ├── SettingsManager.swift        # UserDefaults wrapper
│   │       └── AppSettings.swift            # struct: thresholds, toggles
│   │
│   ├── Utils/
│   │   ├── ImageProcessor.swift     # Resize, format conversion, pixel buffer creation
│   │   ├── GeometryUtils.swift      # IoU calculation, bbox operations
│   │   ├── MathUtils.swift          # Statistical functions, vector operations
│   │   └── Extensions/
│   │       ├── CGImage+Extensions.swift
│   │       ├── UIImage+Extensions.swift
│   │       └── Array+Statistics.swift
│   │
│   ├── Views/
│   │   ├── Main/
│   │   │   ├── ContentView.swift            # Main screen (image, buttons, count)
│   │   │   └── ProcessingStatusView.swift   # Progress indicator overlay
│   │   │
│   │   ├── Camera/
│   │   │   ├── CameraView.swift             # Clean camera viewfinder
│   │   │   └── CameraControlsView.swift     # Shutter, flash, focus controls
│   │   │
│   │   ├── History/
│   │   │   ├── HistoryView.swift            # List of sessions with thumbnails
│   │   │   ├── HistoryRowView.swift         # Single entry: thumbnail, count, date
│   │   │   └── SessionDetailView.swift      # Full session details
│   │   │
│   │   ├── Settings/
│   │   │   ├── SettingsView.swift           # Threshold sliders, toggles
│   │   │   └── ThresholdSliderView.swift    # Reusable slider component
│   │   │
│   │   └── Components/
│   │       ├── CountDisplayView.swift       # Large number display
│   │       ├── ImagePreviewView.swift       # Zoomable image viewer
│   │       └── ErrorAlertView.swift         # Error message dialogs
│   │
│   └── Resources/
│       ├── Assets.xcassets          # App icons, colors, images
│       └── Localizable.strings      # English strings (future i18n)
│
└── AICounterTests/
    ├── UnitTests/
    │   ├── DetectorTests.swift              # YOLOX output parsing
    │   ├── EmbedderTests.swift              # TinyCLIP normalization
    │   ├── ClustererTests.swift             # Clustering algorithm correctness
    │   ├── OutlierFilterTests.swift         # Size/aspect ratio filtering
    │   ├── GeometryUtilsTests.swift         # IoU calculation
    │   ├── ThumbnailGeneratorTests.swift    # Cropping, resizing, compression
    │   └── HistoryManagerTests.swift        # Core Data CRUD, FIFO cleanup
    │
    ├── IntegrationTests/
    │   ├── PipelineTests.swift              # End-to-end counting flow
    │   ├── CoreMLModelTests.swift           # Model loading and inference
    │   └── PersistenceTests.swift           # Core Data migrations
    │
    ├── UITests/
    │   ├── CameraCaptureUITests.swift       # Camera flow
    │   ├── PhotoLibraryUITests.swift        # Library selection
    │   ├── HistoryUITests.swift             # History navigation, deletion
    │   └── SettingsUITests.swift            # Settings adjustments
    │
    ├── PerformanceTests/
    │   ├── InferenceBenchmarks.swift        # YOLOX <100ms, TinyCLIP <20ms
    │   └── MemoryBenchmarks.swift           # <150MB during processing
    │
    └── Mocks/
        ├── MockDetector.swift               # Fake detector for testing
        ├── MockEmbedder.swift               # Fake embedder for testing
        └── MockPersistence.swift            # In-memory Core Data for tests
```

**Structure Decision**: Standard iOS single-target Xcode project. No Swift Package Manager architecture per constitution. All dependencies are built-in Apple frameworks. Core logic organized into protocols for testability. MVVM pattern with SwiftUI for views and ViewModels embedded in services. Clear separation between ML components (Core/), UI (Views/), infrastructure (Services/), and utilities.

## Complexity Tracking

> **No violations - no justifications needed**

All constitution requirements are met without exceptions. The project uses:
- Standard Xcode project structure (no SPM)
- Zero external dependencies (all Apple frameworks)
- Protocol-based design for testability
- Single responsibility components
- Clear architectural boundaries

## High-Level Architecture

### System Overview

```
┌───────────────────────────────────────────────────────────────────┐
│                         AICounter iOS App                          │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌─────────────────── Presentation Layer ─────────────────────┐  │
│  │                                                              │  │
│  │  ContentView  ────▶  CameraView  ────▶  HistoryView         │  │
│  │       │                  │                    │              │  │
│  │       │                  │                    │              │  │
│  │       ▼                  ▼                    ▼              │  │
│  │  ProcessingStatusView  CameraControlsView  SessionDetailView│  │
│  │                                                              │  │
│  └──────────────────────┬───────────────────────────────────────┘  │
│                         │                                          │
│  ┌──────────────────────▼────── Service Layer ──────────────────┐ │
│  │                                                                │ │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐    │ │
│  │  │ CameraManager│  │HistoryManager│  │ SettingsManager  │    │ │
│  │  │             │  │              │  │                  │    │ │
│  │  │ • AVCapture │  │ • Core Data  │  │ • UserDefaults   │    │ │
│  │  │ • Permissions│  │ • FIFO cleanup│  │ • Thresholds     │    │ │
│  │  │ • Photo lib │  │ • Thumbnails │  │ • Toggles        │    │ │
│  │  └─────────────┘  └──────────────┘  └──────────────────┘    │ │
│  │                                                                │ │
│  └──────────────────────┬─────────────────────────────────────────┘ │
│                         │                                          │
│  ┌──────────────────────▼──────── Core Layer ────────────────────┐ │
│  │                                                                │ │
│  │                      AICounter                                │ │
│  │               (Pipeline Orchestrator)                         │ │
│  │                         │                                      │ │
│  │         ┌───────────────┼───────────────┐                     │ │
│  │         │               │               │                     │ │
│  │         ▼               ▼               ▼                     │ │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────────┐             │ │
│  │  │  YOLOX   │   │ TinyCLIP │   │  Similarity  │             │ │
│  │  │ Detector │   │ Embedder │   │  Clusterer   │             │ │
│  │  │          │   │          │   │              │             │ │
│  │  │ • Parse  │   │ • Extract│   │ • Cosine sim │             │ │
│  │  │ • Filter │   │ • L2 norm│   │ • Agg. clust │             │ │
│  │  │ • NMS    │   │          │   │ • Main clust │             │ │
│  │  └──────────┘   └──────────┘   └──────────────┘             │ │
│  │         │               │               │                     │ │
│  │         └───────────────┴───────────────┘                     │ │
│  │                         │                                      │ │
│  │                         ▼                                      │ │
│  │              ┌─────────────────┐                              │ │
│  │              │ Outlier Filters │                              │ │
│  │              │ • Size          │                              │ │
│  │              │ • Aspect ratio  │                              │ │
│  │              │ • Area consist. │                              │ │
│  │              │ • IoU merging   │                              │ │
│  │              └─────────────────┘                              │ │
│  │                                                                │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌────────────────────── Model Layer ─────────────────────────┐  │
│  │                                                              │  │
│  │  ┌──────────────────┐         ┌──────────────────┐         │  │
│  │  │ yolox_s.mlpackage│         │tinyclip_vision   │         │  │
│  │  │                  │         │  .mlpackage      │         │  │
│  │  │ • Input: 640×640 │         │ • Input: 224×224 │         │  │
│  │  │ • Output: 8400×85│         │ • Output: 1×256  │         │  │
│  │  │ • Neural Engine  │         │ • L2 normalized  │         │  │
│  │  └──────────────────┘         └──────────────────┘         │  │
│  │                                                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
User Action: Capture Photo / Select from Library
          │
          ▼
┌─────────────────────┐
│   Image Input       │
│   (UIImage/CGImage) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  AICounter.count()  │ ◀── Main orchestration method
└──────────┬──────────┘
           │
           ├─── Stage 1: Object Detection ───────────────────────┐
           │                                                       │
           ▼                                                       │
    ┌──────────────┐                                             │
    │ YOLOXDetector │                                             │
    │ • Resize 640×640                                            │
    │ • CoreML inference                                          │
    │ • Parse [1,8400,85]                                         │
    │ • Apply conf threshold                                      │
    └──────┬───────┘                                             │
           │ Detections: [(bbox, conf, class)]                   │
           │                                                       │
           ├─── Stage 1.5: Size Outlier Filtering ──────────────┤
           │                                                       │
           ▼                                                       │
    ┌──────────────┐                                             │
    │OutlierFilter │                                             │
    │ • Calc areas                                                │
    │ • Median ± 1.0σ                                             │
    │ • Keep valid boxes                                          │
    └──────┬───────┘                                             │
           │ Filtered detections                                 │
           │                                                       │
           ├─── Stage 1.6: Aspect Ratio Filtering ──────────────┤
           │                                                       │
           ▼                                                       │
    ┌──────────────┐                                             │
    │OutlierFilter │                                             │
    │ • Calc ratios (w/h)                                         │
    │ • Median ± 0.5σ                                             │
    │ • Keep consistent shapes                                    │
    └──────┬───────┘                                             │
           │ Shape-filtered detections                           │
           │                                                       │
           ├─── Stage 2: Embedding Extraction ──────────────────┤
           │                                                       │
           ▼                                                       │
    ┌──────────────┐                                             │
    │ Crop Images  │                                             │
    │ • Extract bbox regions                                      │
    │ • Filter by area consistency (median ± 1.0σ)               │
    └──────┬───────┘                                             │
           │ Cropped images                                      │
           ▼                                                       │
    ┌──────────────┐                                             │
    │TinyCLIPEmbedder│                                           │
    │ • Resize 224×224                                            │
    │ • CoreML inference                                          │
    │ • L2 normalize embeddings                                   │
    └──────┬───────┘                                             │
           │ Embeddings: [[Float]] (256-dim each)                │
           │                                                       │
           ├─── Stage 3: Similarity Clustering ─────────────────┤
           │                                                       │
           ▼                                                       │
    ┌──────────────┐                                             │
    │SimilarityClusterer│                                        │
    │ • Agglomerative clustering                                  │
    │ • Cosine distance metric                                    │
    │ • Average linkage                                           │
    │ • Distance threshold = 1 - similarity (0.20 default)       │
    └──────┬───────┘                                             │
           │ Cluster labels: [Int]                               │
           │                                                       │
           ├─── Stage 4: IoU-Based Merging ─────────────────────┤
           │                                                       │
           ▼                                                       │
    ┌──────────────┐                                             │
    │  NMSFilter   │                                             │
    │ • Sort by confidence                                        │
    │ • Merge overlaps (IoU > 0.5)                               │
    │ • Keep highest confidence                                   │
    └──────┬───────┘                                             │
           │ Final detections with cluster IDs                   │
           │                                                       │
           ├─── Stage 5: Result Generation ─────────────────────┤
           │                                                       │
           ▼                                                       │
    ┌──────────────┐                                             │
    │ClusterAnalyzer│                                            │
    │ • Count per cluster                                         │
    │ • Identify largest cluster                                  │
    │ • Extract main cluster count                               │
    └──────┬───────┘                                             │
           │                                                       │
           ▼                                                       │
    ┌──────────────┐                                             │
    │ CountResult  │                                             │
    │ • count: Int (main cluster)                                │
    │ • detections: [Detection]                                   │
    │ • largestClusterId: Int                                     │
    └──────┬───────┘                                             │
           │                                                       │
           ├─── Thumbnail Generation (for History) ──────────────┤
           │                                                       │
           ▼                                                       │
    ┌──────────────┐                                             │
    │ThumbnailGenerator│                                         │
    │ • Filter main cluster detections                            │
    │ • Select representative bbox                                │
    │ • Crop from original image                                  │
    │ • Resize to 200×200                                         │
    │ • Compress to JPEG (80%)                                    │
    └──────┬───────┘                                             │
           │ Thumbnail: Data                                      │
           │                                                       │
           ├─── Save to History ─────────────────────────────────┤
           │                                                       │
           ▼                                                       │
    ┌──────────────┐                                             │
    │HistoryManager│                                             │
    │ • Create CountingSession entity                             │
    │ • Save thumbnail + count + timestamp                        │
    │ • FIFO cleanup if >100 entries                             │
    │ • Persist to Core Data                                      │
    └──────────────┘                                             │
           │                                                       │
           ▼                                                       │
    Display Result to User                                       │
```

### Component Responsibilities

#### 1. AICounter (Main Orchestrator)
**Purpose**: Coordinate the entire counting pipeline from image input to result generation.

**Key Methods**:
```swift
func count(image: UIImage, 
           confThreshold: Float = 0.001, 
           similarityThreshold: Float = 0.80,
           enableSizeFilter: Bool = true,
           enableAspectFilter: Bool = true) async throws -> CountResult

private func filterSizeOutliers(_ detections: [Detection], stdThreshold: Float = 1.0) -> [Detection]
private func filterAspectRatioOutliers(_ detections: [Detection], stdThreshold: Float = 0.5) -> [Detection]
private func cropDetections(_ image: CGImage, detections: [Detection]) -> [CropInfo]
private func filterByAreaConsistency(_ crops: [CropInfo], stdThreshold: Float = 1.0) -> [CropInfo]
```

**Dependencies**:
- `YOLOXDetector` (object detection)
- `TinyCLIPEmbedder` (embedding extraction)
- `SimilarityClusterer` (clustering)
- `OutlierFilter` (statistical filtering)
- `NMSFilter` (IoU merging)

#### 2. YOLOXDetector
**Purpose**: Run object detection using YOLOX-S CoreML model and parse outputs.

**Key Methods**:
```swift
func detect(image: CGImage, confThreshold: Float) async throws -> [Detection]
private func parseYOLOXOutput(_ output: MLMultiArray, 
                              imageSize: CGSize, 
                              confThreshold: Float) -> [Detection]
```

**Input**: CGImage (will be resized to 640×640 internally)
**Output**: Array of Detection structs with bbox, confidence, classId

#### 3. TinyCLIPEmbedder
**Purpose**: Extract 256-dimensional visual embeddings from cropped images.

**Key Methods**:
```swift
func getEmbedding(for image: CGImage) async throws -> [Float]
private func normalizeL2(_ embedding: [Float]) -> [Float]
```

**Input**: CGImage (will be resized to 224×224 internally)
**Output**: L2-normalized 256-dimensional embedding vector

#### 4. SimilarityClusterer
**Purpose**: Group embeddings into clusters using agglomerative clustering with cosine distance.

**Key Methods**:
```swift
func cluster(embeddings: [[Float]], 
             similarityThreshold: Float) -> ([Int], [Int: Int])
private func cosineSimilarity(_ vec1: [Float], _ vec2: [Float]) -> Float
private func buildDistanceMatrix(_ embeddings: [[Float]]) -> [[Float]]
private func agglomerativeClustering(distanceMatrix: [[Float]], 
                                     threshold: Float) -> [Int]
```

**Input**: Array of embeddings
**Output**: Tuple of (cluster labels, cluster counts)

**Algorithm**: Bottom-up hierarchical clustering
- Start with each point as its own cluster
- Iteratively merge closest clusters using average linkage
- Stop when all cluster distances exceed threshold
- Use Accelerate framework for efficient distance calculations

#### 5. OutlierFilter
**Purpose**: Statistical filtering to remove anomalous detections.

**Key Methods**:
```swift
func filterBySize(_ detections: [Detection], stdThreshold: Float) -> [Detection]
func filterByAspectRatio(_ detections: [Detection], stdThreshold: Float) -> [Detection]
func filterByAreaConsistency(_ crops: [CropInfo], stdThreshold: Float) -> [CropInfo]
```

**Algorithm**: Median ± N standard deviations
- Calculate metric (area, aspect ratio, etc.)
- Compute median and std deviation
- Keep only items within [median - N×σ, median + N×σ]

#### 6. NMSFilter
**Purpose**: Merge overlapping detections using IoU-based Non-Maximum Suppression.

**Key Methods**:
```swift
func mergeOverlapping(_ detections: [Detection], 
                      clusterLabels: [Int], 
                      iouThreshold: Float) -> ([Detection], [Int])
private func calculateIoU(_ box1: CGRect, _ box2: CGRect) -> Float
```

**Algorithm**: Greedy NMS
- Sort detections by confidence (descending)
- Keep highest confidence detection
- Suppress all overlapping boxes (IoU > threshold)
- Repeat for remaining detections

#### 7. CameraManager
**Purpose**: Manage AVCaptureSession for clean camera viewfinder and photo capture.

**Key Methods**:
```swift
func startSession() async throws
func stopSession()
func capturePhoto() async throws -> UIImage
func requestPermissions() async -> Bool
```

**Features**:
- Clean viewfinder (no real-time overlays)
- Flash control
- Focus control
- Orientation handling

#### 8. HistoryManager
**Purpose**: Persist counting sessions to Core Data with FIFO cleanup.

**Key Methods**:
```swift
func saveSession(result: CountResult, 
                 thumbnailData: Data, 
                 settings: AppSettings) async throws
func fetchHistory(limit: Int = 100) -> [CountingSession]
func deleteSession(id: UUID) async throws
func clearAll() async throws
private func enforceLimit() async throws  // Keep max 100 entries
```

**Core Data Entity**:
```swift
class CountingSession: NSManagedObject {
    @NSManaged var id: UUID
    @NSManaged var thumbnailData: Data       // Cropped object JPEG (200×200)
    @NSManaged var count: Int16              // Main cluster count only
    @NSManaged var timestamp: Date
    @NSManaged var confidenceThreshold: Float
    @NSManaged var similarityThreshold: Float
    @NSManaged var isFavorite: Bool
}
```

#### 9. ThumbnailGenerator
**Purpose**: Generate thumbnails by cropping representative objects from main cluster.

**Key Methods**:
```swift
func generateThumbnail(from image: CGImage, 
                       detections: [Detection], 
                       clusterLabels: [Int], 
                       mainClusterId: Int) -> Data?
private func selectRepresentative(_ detections: [Detection]) -> Detection
private func cropAndResize(_ image: CGImage, bbox: CGRect, size: CGSize) -> CGImage?
private func compressToJPEG(_ image: CGImage, quality: CGFloat = 0.8) -> Data?
```

**Algorithm**:
1. Filter detections to main cluster
2. Select representative (first/centered/highest confidence)
3. Crop bbox region from original image
4. Resize to 200×200 with aspect ratio preserved (padding if needed)
5. Compress to JPEG (quality 80%, target ~10KB)

#### 10. SettingsManager
**Purpose**: Persist user preferences in UserDefaults.

**Key Methods**:
```swift
func getSettings() -> AppSettings
func updateSettings(_ settings: AppSettings)
func resetToDefaults()
```

**AppSettings struct**:
```swift
struct AppSettings: Codable {
    var confidenceThreshold: Float = 0.001
    var similarityThreshold: Float = 0.80
    var enableSizeFilter: Bool = true
    var enableAspectRatioFilter: Bool = true
    var enableAreaFilter: Bool = true
}
```

## Detailed Technical Approach

### Algorithm Implementation Strategies (Python → Swift)

#### 1. Size Outlier Filtering

**Python Reference** (`count_objects_yolox.py:240-273`):
```python
def remove_size_outliers(boxes, std_threshold=3.0):
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    median_area = np.median(areas)
    std_area = np.std(areas)
    min_area = median_area - std_threshold * std_area
    max_area = median_area + std_threshold * std_area
    return [i for i, area in enumerate(areas) if min_area <= area <= max_area]
```

**Swift Implementation Strategy**:
```swift
// OutlierFilter.swift
func filterBySize(_ detections: [Detection], stdThreshold: Float = 1.0) -> [Detection] {
    guard detections.count >= 3 else { return detections }
    
    // Calculate areas
    let areas = detections.map { detection in
        (detection.bbox.width * detection.bbox.height)
    }
    
    // Statistical calculations
    let medianArea = areas.median()  // Extension on Array<Float>
    let stdArea = areas.standardDeviation()
    
    // Bounds
    let minArea = medianArea - stdThreshold * stdArea
    let maxArea = medianArea + stdThreshold * stdArea
    
    // Filter
    return detections.enumerated().compactMap { index, detection in
        (minArea...maxArea).contains(areas[index]) ? detection : nil
    }
}
```

**Key Differences**:
- Python uses NumPy vectorized operations → Swift uses functional methods (map, filter)
- Python threshold: 3.0σ → Swift matches spec: 1.0σ (tighter filtering per spec.md)
- Swift leverages Array extensions for statistical functions

### Performance Optimization Tactics

#### 1. Memory Management

**Strategy**: Lazy model loading and unloading
```swift
class ModelManager {
    private var yoloxModel: yolox_s?
    private var tinyclipModel: tinyclip_vision?
    
    func loadModels() throws {
        guard yoloxModel == nil else { return }
        
        let config = MLModelConfiguration()
        config.computeUnits = .all
        
        yoloxModel = try yolox_s(configuration: config)
        tinyclipModel = try tinyclip_vision(configuration: config)
    }
    
    func unloadModels() {
        yoloxModel = nil
        tinyclipModel = nil
    }
}
```

**Backgrounding behavior**:
- Unload models when app enters background
- Reload on foreground if processing needed
- Reduces idle memory from 150MB to <50MB

#### 2. Batch Processing Optimization

**Problem**: Processing 50 crops individually = 50× TinyCLIP calls
**Solution**: CoreML batch prediction (if supported by model)

```swift
func extractEmbeddings(for crops: [CGImage]) async throws -> [[Float]] {
    // Check if batch prediction is available
    if crops.count > 1 {
        return try await batchExtract(crops)
    } else {
        return try await crops.asyncMap { try await getEmbedding(for: $0) }
    }
}
```

**Expected speedup**: 30-40% for 20+ objects

#### 3. Accelerate Framework for Vector Operations

**Use cases**:
- Cosine similarity (vDSP_dotpr)
- L2 normalization (vDSP_svesq, vDSP_vsmul)
- Statistical operations (vDSP_meanv, vDSP_vari)

**Performance gain**: 3-5× faster than pure Swift loops for large vectors

### Testing Approach for Each Layer

#### Unit Tests (80%+ coverage target)

**Detection Layer**:
```swift
// DetectorTests.swift
func testYOLOXOutputParsing() {
    let mockOutput = createMockMLMultiArray(shape: [1, 8400, 85])
    let detections = parser.parseYOLOXOutput(mockOutput, imageSize: CGSize(width: 640, height: 640))
    
    XCTAssertEqual(detections.count, expectedCount)
    XCTAssert(detections.allSatisfy { $0.confidence >= confThreshold })
}

func testSizeOutlierFiltering() {
    let detections = createMockDetections(areas: [100, 105, 110, 500, 108])  // 500 is outlier
    let filtered = outlierFilter.filterBySize(detections, stdThreshold: 1.0)
    
    XCTAssertEqual(filtered.count, 4)  // Should remove 500
    XCTAssertFalse(filtered.contains { $0.bbox.area == 500 })
}
```

**Clustering Layer**:
```swift
// ClustererTests.swift
func testCosineSimilarity() {
    let vec1: [Float] = [1.0, 0.0, 0.0]
    let vec2: [Float] = [0.0, 1.0, 0.0]
    let similarity = clusterer.cosineSimilarity(vec1, vec2)
    
    XCTAssertEqual(similarity, 0.0, accuracy: 0.001)  // Orthogonal vectors
}

func testAgglomerativeClustering() {
    let embeddings = createSyntheticEmbeddings()  // 3 clusters of 5 embeddings each
    let (labels, counts) = clusterer.cluster(embeddings: embeddings, similarityThreshold: 0.80)
    
    XCTAssertEqual(counts.count, 3)  // Should identify 3 clusters
    XCTAssertEqual(counts.values.reduce(0, +), 15)  // Total 15 points
}
```

**Persistence Layer**:
```swift
// HistoryManagerTests.swift
func testFIFOCleanup() async throws {
    // Add 101 entries
    for i in 0..<101 {
        try await historyManager.saveSession(...)
    }
    
    let history = historyManager.fetchHistory()
    XCTAssertEqual(history.count, 100)  // Should enforce limit
    
    // Verify oldest entry was deleted
    XCTAssertFalse(history.contains { $0.timestamp == oldestTimestamp })
}
```

#### Integration Tests

**Full Pipeline**:
```swift
// PipelineTests.swift
func testEndToEndCounting() async throws {
    let testImage = loadTestImage("bottles_15.jpg")  // Known 15 bottles
    
    let result = try await aiCounter.count(image: testImage)
    
    XCTAssertEqual(result.count, 15, accuracy: 1)  // Allow ±1 tolerance
    XCTAssertGreaterThan(result.detections.count, 10)
    XCTAssert(result.processingTime < 2.0)  // Performance requirement
}
```

#### UI Tests

**Camera Flow**:
```xctest
func testCameraCaptureFlow() {
    let app = XCUIApplication()
    app.launch()
    
    app.buttons["Capture Photo"].tap()
    XCTAssert(app.otherElements["CameraView"].waitForExistence(timeout: 2))
    
    app.buttons["ShutterButton"].tap()
    XCTAssert(app.images["CapturedImage"].waitForExistence(timeout: 1))
    
    app.buttons["Count Objects"].tap()
    XCTAssert(app.staticTexts.matching(identifier: "CountResult").element.waitForExistence(timeout: 3))
}
```

#### Performance Benchmarks

```swift
// InferenceBenchmarks.swift
func testYOLOXInferenceTime() {
    let image = loadTestImage("test_640x640.jpg")
    
    measure(metrics: [XCTClockMetric()]) {
        _ = try? detector.detect(image: image.cgImage!)
    }
    
    // Baseline: <100ms on iPhone 12
}

func testMemoryUsage() {
    let image = loadTestImage("large_image.jpg")
    
    measure(metrics: [XCTMemoryMetric()]) {
        _ = try? aiCounter.count(image: image)
    }
    
    // Baseline: <150MB peak
}
```

### Implementation Timeline and Milestones

#### Phase 0: Project Setup (Week 1)
**Goal**: Development environment ready, models integrated

**Tasks**:
- [ ] Create Xcode project (standard structure, iOS 15+ target)
- [ ] Convert YOLOX-S to CoreML (yolox_s.mlpackage)
- [ ] Convert TinyCLIP to CoreML (tinyclip_vision.mlpackage)
- [ ] Add models to Xcode project
- [ ] Verify model inference works (simple test)
- [ ] Setup Core Data schema
- [ ] Setup SwiftUI app structure

**Deliverables**: 
- Xcode project that builds successfully
- Models integrated and loadable
- Basic ContentView with placeholder UI

**Success Criteria**:
- App launches on simulator/device
- Models load without errors
- Basic UI displays

---

#### Phase 1: Core Detection Pipeline (Weeks 2-3)
**Goal**: YOLOX detection and outlier filtering working

**Tasks**:
- [ ] Implement YOLOXDetector
  - [ ] Model loading and configuration
  - [ ] Image preprocessing (640×640 resize)
  - [ ] Output parsing ([1, 8400, 85])
  - [ ] Detection struct creation
- [ ] Implement OutlierFilter
  - [ ] Size outlier removal (median ± 1.0σ)
  - [ ] Aspect ratio filtering (median ± 0.5σ)
  - [ ] Area consistency filtering (median ± 1.0σ)
  - [ ] Array statistical extensions
- [ ] Unit tests for detection and filtering
- [ ] Integration test: image → detections

**Deliverables**:
- YOLOXDetector.swift (100% tested)
- OutlierFilter.swift (100% tested)
- Detection output visualization (debug)

**Success Criteria**:
- Detection returns valid bboxes for test images
- Outlier filtering reduces false positives
- Unit tests pass with 80%+ coverage
- Performance: <100ms detection time

---

#### Phase 2: Embedding and Clustering (Weeks 4-5)
**Goal**: TinyCLIP embeddings and similarity clustering functional

**Tasks**:
- [ ] Implement TinyCLIPEmbedder
  - [ ] Model loading
  - [ ] Image preprocessing (224×224 resize)
  - [ ] L2 normalization (Accelerate framework)
  - [ ] Embedding extraction
- [ ] Implement SimilarityClusterer
  - [ ] Cosine similarity (Accelerate vDSP_dotpr)
  - [ ] Distance matrix construction
  - [ ] Agglomerative clustering algorithm
  - [ ] Average linkage calculation
  - [ ] Cluster counting and analysis
- [ ] Implement NMSFilter (IoU-based merging)
- [ ] Unit tests for embeddings and clustering
- [ ] Integration test: detections → clusters → count

**Deliverables**:
- TinyCLIPEmbedder.swift (tested)
- SimilarityClusterer.swift (tested with synthetic data)
- NMSFilter.swift (tested)

**Success Criteria**:
- Embeddings match Python output (cosine similarity >0.99)
- Clustering produces consistent results
- IoU merging removes duplicates correctly
- Performance: <20ms per embedding, <200ms clustering

---

#### Phase 3: Pipeline Integration (Week 6)
**Goal**: End-to-end counting pipeline operational

**Tasks**:
- [ ] Implement AICounter orchestrator
  - [ ] Stage 1: Detection
  - [ ] Stage 1.5: Size filtering
  - [ ] Stage 1.6: Aspect ratio filtering
  - [ ] Stage 2: Cropping and embedding
  - [ ] Stage 3: Clustering
  - [ ] Stage 4: IoU merging
  - [ ] Stage 5: Result generation
- [ ] Async/await pipeline with progress reporting
- [ ] Error handling for each stage
- [ ] Integration tests with real images
- [ ] Accuracy validation vs Python

**Deliverables**:
- AICounter.swift (main orchestrator)
- End-to-end integration tests
- Accuracy report (vs Python baseline)

**Success Criteria**:
- Complete pipeline produces valid counts
- Accuracy ≥95% vs Python on test set
- Total processing time <2s (iPhone 12+)
- Memory usage <150MB

---

#### Phase 4: Camera and Input System (Week 7)
**Goal**: Camera capture and photo library working

**Tasks**:
- [ ] Implement CameraManager
  - [ ] AVCaptureSession setup
  - [ ] Permission handling
  - [ ] Photo capture delegate
  - [ ] Clean viewfinder (no overlays)
- [ ] Implement CameraView (SwiftUI)
  - [ ] Live preview layer
  - [ ] Shutter button
  - [ ] Flash/focus controls
- [ ] Implement PHPickerViewController integration
- [ ] Permission request alerts
- [ ] UI tests for camera and library flows

**Deliverables**:
- CameraManager.swift
- CameraView.swift
- Photo library picker integration

**Success Criteria**:
- Camera captures photos successfully
- Library picker selects images correctly
- Permissions handled gracefully
- Clean viewfinder (no real-time inference)

---

#### Phase 5: Persistence and History (Week 8)
**Goal**: Core Data history with thumbnails working

**Tasks**:
- [ ] Implement PersistenceController
- [ ] Implement HistoryManager
  - [ ] Save session (with thumbnail)
  - [ ] Fetch history (reverse chronological)
  - [ ] Delete session
  - [ ] Clear all history
  - [ ] FIFO cleanup (max 100 entries)
- [ ] Implement ThumbnailGenerator
  - [ ] Crop main cluster object
  - [ ] Resize to 200×200
  - [ ] JPEG compression (80%)
- [ ] Implement HistoryView UI
- [ ] Unit tests for persistence
- [ ] Migration tests

**Deliverables**:
- Core Data stack
- HistoryManager.swift (tested)
- ThumbnailGenerator.swift (tested)
- HistoryView.swift

**Success Criteria**:
- Sessions persist across app launches
- FIFO cleanup works correctly
- Thumbnails display correctly
- Storage <20MB for 100 entries

---

#### Phase 6: UI and Settings (Week 9)
**Goal**: Complete UI with settings management

**Tasks**:
- [ ] Implement ContentView (main screen)
  - [ ] Image preview
  - [ ] Count display
  - [ ] Input buttons
  - [ ] Processing status overlay
- [ ] Implement SettingsView
  - [ ] Threshold sliders
  - [ ] Filter toggles
  - [ ] Reset to defaults
- [ ] Implement SettingsManager (UserDefaults)
- [ ] Polish animations and transitions
- [ ] Accessibility support
- [ ] UI tests

**Deliverables**:
- ContentView.swift (complete)
- SettingsView.swift
- SettingsManager.swift
- UI polish

**Success Criteria**:
- All screens functional and polished
- Settings persist correctly
- 60fps UI animations
- Accessibility labels present

---

#### Phase 7: Testing and Optimization (Week 10)
**Goal**: Production-ready quality

**Tasks**:
- [ ] Performance profiling (Instruments)
  - [ ] Identify bottlenecks
  - [ ] Optimize hot paths
  - [ ] Memory leak detection
- [ ] Accuracy validation (50-image test set)
- [ ] Battery consumption testing
- [ ] Edge case testing
- [ ] Stress testing (100+ objects)
- [ ] UI/UX polish
- [ ] Bug fixes

**Deliverables**:
- Performance report
- Accuracy report
- Bug fixes
- Optimization improvements

**Success Criteria**:
- All performance targets met
- 80%+ test coverage achieved
- Zero critical bugs
- Accuracy ≥95% on test set

---

#### Phase 8: Beta and Launch Prep (Week 11)
**Goal**: Ready for TestFlight beta

**Tasks**:
- [ ] App Store assets (icon, screenshots)
- [ ] Privacy policy (on-device processing)
- [ ] App Store description
- [ ] TestFlight setup
- [ ] Beta testing (20+ users)
- [ ] Feedback collection
- [ ] Final bug fixes
- [ ] App Store submission

**Deliverables**:
- TestFlight beta build
- App Store listing
- Privacy policy
- Beta feedback report

**Success Criteria**:
- Beta testers report positive experience
- No showstopper bugs found
- App Store submission approved

---

### Risk Mitigation Strategies

#### Risk 1: Agglomerative Clustering Performance
**Risk**: Custom Swift implementation may be too slow for 50+ objects

**Mitigation**:
1. **Primary**: Optimize with Accelerate framework for distance calculations
2. **Fallback**: Implement approximate clustering (k-means with k=5) if exact clustering >500ms
3. **Monitoring**: Performance benchmarks in CI/CD

**Trigger**: If clustering >200ms for 50 objects in Phase 2 testing

---

#### Risk 2: Model Accuracy Degradation in CoreML
**Risk**: Conversion to CoreML may reduce accuracy vs Python ONNX

**Mitigation**:
1. **Validation**: Compare CoreML vs ONNX outputs layer-by-layer during conversion
2. **Test Set**: Create 50-image validation set with Python ground truth
3. **Acceptance**: Allow ≤2% accuracy drop (95% → 93% acceptable)
4. **Fallback**: If >2% drop, investigate quantization settings or use Float32 instead of Float16

**Trigger**: Accuracy validation after Phase 1 completion

---

#### Risk 3: Memory Constraints on Older Devices
**Risk**: App exceeds 200MB and gets terminated on iPhone X

**Mitigation**:
1. **Lazy Loading**: Unload models when backgrounded
2. **Batch Size Limits**: Process max 100 objects, downsample if more detected
3. **Image Downscaling**: Scale input images if >4K resolution
4. **Testing**: Test on iPhone X (A11 Bionic) throughout development

**Trigger**: Memory profiling in Phase 7

---

#### Risk 4: App Store Rejection
**Risk**: Rejection due to unclear ML functionality or privacy concerns

**Mitigation**:
1. **Clear Description**: Explain on-device ML processing (no cloud, no data collection)
2. **Privacy Policy**: Explicit statement about local-only processing
3. **Permission Justification**: Clear explanations for camera/photo library access
4. **Pre-submission Review**: Test against App Store Review Guidelines

**Trigger**: Before Phase 8 submission

---

#### Risk 5: Real-World Accuracy Below Expectations
**Risk**: User-reported accuracy <90% in production

**Mitigation**:
1. **Adjustable Thresholds**: Allow users to fine-tune sensitivity
2. **Visual Feedback**: Overlay mode helps users understand what was counted
3. **Iterative Improvement**: Collect feedback (opt-in) for future improvements
4. **Presets**: Offer "Precise", "Balanced", "Generous" modes

**Trigger**: Beta testing Phase 8, post-launch monitoring

---

### API Contracts Between Components

#### DetectorProtocol

```swift
protocol DetectorProtocol {
    func detect(image: CGImage, confThreshold: Float) async throws -> [Detection]
}

enum DetectorError: Error {
    case modelLoadFailed
    case inferenceError(String)
    case invalidInput
}
```

#### EmbedderProtocol

```swift
protocol EmbedderProtocol {
    func getEmbedding(for image: CGImage) async throws -> [Float]
}

enum EmbedderError: Error {
    case modelLoadFailed
    case resizeFailed
    case pixelBufferCreationFailed
    case inferenceError(String)
}
```

#### ClustererProtocol

```swift
protocol ClustererProtocol {
    func cluster(embeddings: [[Float]], similarityThreshold: Float) -> (labels: [Int], counts: [Int: Int])
}
```

#### PersistenceProtocol

```swift
protocol PersistenceProtocol {
    func saveSession(_ result: CountResult, thumbnailData: Data, settings: AppSettings) async throws
    func fetchHistory(limit: Int) -> [CountingSession]
    func deleteSession(id: UUID) async throws
    func clearAll() async throws
}

enum PersistenceError: Error {
    case saveFailed(String)
    case fetchFailed(String)
    case deleteFailed(String)
}
```

---

## Phase 0: Research Plan (To Be Executed)

### Research Topics

#### 1. Agglomerative Clustering Implementation in Swift
**Question**: How to implement agglomerative clustering with cosine distance and average linkage without external dependencies?

**Research Tasks**:
- Review scikit-learn's AgglomerativeClustering source code
- Identify Swift-native approaches (no GPL libraries)
- Evaluate Accelerate framework capabilities for distance matrix operations
- Prototype simple implementation with synthetic data

**Output**: `research.md` section on clustering strategy

---

#### 2. CoreML Model Conversion Best Practices
**Question**: What's the optimal process for converting ONNX models to CoreML while preserving accuracy?

**Research Tasks**:
- Review coremltools documentation
- Identify quantization options (Float32 vs Float16)
- Determine compute unit preferences (Neural Engine, GPU, CPU)
- Test conversion with YOLOX-S and TinyCLIP models

**Output**: `research.md` section on model conversion

---

#### 3. Core Data Performance Optimization
**Question**: How to ensure Core Data operations don't block the UI with 100 entries and thumbnails?

**Research Tasks**:
- Review background context best practices
- Evaluate NSFetchedResultsController for live updates
- Determine optimal thumbnail size/quality tradeoff
- Test concurrent access patterns

**Output**: `research.md` section on persistence strategy

---

#### 4. Image Preprocessing for CoreML
**Question**: What's the most efficient way to convert UIImage → CVPixelBuffer for CoreML?

**Research Tasks**:
- Compare Core Graphics vs Accelerate vs Metal
- Identify optimal pixel format (32BGRA, 32RGBA, etc.)
- Determine resize strategies (letterbox, center crop, stretch)
- Benchmark preprocessing overhead

**Output**: `research.md` section on image processing

---

## Phase 1: Design Artifacts (To Be Generated)

### data-model.md
**Contents**:
- Core Data schema (CountingSession entity)
- Swift structs (Detection, CountResult, AppSettings, ProcessingStage)
- Model relationships and constraints
- Migration strategy

### contracts/
**Files**:
- `detector-protocol.swift` - DetectorProtocol definition
- `embedder-protocol.swift` - EmbedderProtocol definition
- `clusterer-protocol.swift` - ClustererProtocol definition
- `persistence-protocol.swift` - PersistenceProtocol definition

### quickstart.md
**Contents**:
- Prerequisites (Xcode, Python for model conversion)
- Model conversion steps (ONNX → CoreML)
- Project setup (clone, open Xcode)
- First build instructions
- Running tests

---

## Summary

This implementation plan provides:

✅ **Complete architectural design** - MVVM with clear component boundaries
✅ **Detailed algorithm migration** - Python → Swift with code examples
✅ **CoreML integration strategy** - Model loading, preprocessing, output parsing
✅ **Performance optimization tactics** - Memory management, Accelerate framework, lazy loading
✅ **Comprehensive testing approach** - Unit, integration, UI, performance tests
✅ **Phase-by-phase timeline** - 11 weeks from setup to launch
✅ **Risk mitigation strategies** - Identified risks with concrete fallback plans
✅ **API contracts** - Protocol-based design for testability
✅ **Constitution compliance** - Zero external dependencies, standard Xcode structure

### Next Steps

1. **Execute Phase 0 research** - Run research tasks, generate `research.md`
2. **Generate Phase 1 design artifacts** - Create `data-model.md`, `contracts/`, `quickstart.md`
3. **Begin Phase 1 implementation** - Start with detection pipeline
4. **Iterate through phases** - Follow timeline with weekly milestones

This plan is ready for development kickoff.
