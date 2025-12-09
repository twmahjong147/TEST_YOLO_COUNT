# AICounter - iOS Migration Project

**Last Updated:** 2025-12-09  
**Project Status:** Specification & Planning Complete

---

## Project Overview

AICounter is an iOS application that migrates Python-based YOLOX object counting functionality to a native iOS app. The app uses computer vision and machine learning to count similar objects in images captured via camera or selected from the photo library.

### Key Technologies
- **Swift 5.9+** for iOS development
- **CoreML** for on-device ML inference (YOLOX-S, TinyCLIP)
- **AVFoundation** for camera capture
- **Core Data** for counting history persistence
- **SwiftUI** for modern iOS UI

---

## Project Documents

### Product Requirements
- **AICounter_PRD.md** - Complete product requirements document
  - 8 features (4 MVP, 4 post-MVP)
  - Detailed algorithm pseudocode from Python reference
  - UI/UX specifications
  - Performance requirements (<2s processing, <150MB memory)
  - 11-week development roadmap

### Python Reference Implementation
- **count_objects_yolox.py** (1,007 lines) - Reference implementation
  - YOLOX object detection
  - TinyCLIP embedding extraction
  - Similarity-based clustering
  - Size/aspect ratio outlier filtering
  - IoU-based detection merging

### Conversion Documentation
- **COREML_CONVERSION_SUMMARY.md** - CoreML model conversion guide
- **MODEL_SIZE_EXPLANATION.md** - Model size optimization details
- **MODEL_TESTING_RESULTS.md** - Validation test results
- **README_COREML.md** - CoreML integration instructions
- **SWIFT_MIGRATION_GUIDE.md** - Swift migration guidelines

### Project Constitution
- **.specify/memory/constitution.md** (v1.0.0, ratified 2025-12-09)
  - **5 Core Principles:**
    1. Code Quality (clean code, max 50 lines/function, max 3 nesting levels)
    2. Testing Standards (80% coverage, TDD, performance tests)
    3. UX Consistency (iOS HIG, accessibility, 60fps animations)
    4. Performance Requirements (<2s launch, <100ms inference, <150MB memory)
    5. Architecture Constraints (standard Xcode, NO Swift Package Manager)

---

## Feature Specification

### Location
`specs/001-ios-migration/`

### Specification Details
- **spec.md** (24KB) - Complete feature specification
  - 5 prioritized user stories (P1-P3)
  - 36 functional requirements
  - 16 success criteria (12 measurable + 4 user satisfaction)
  - 10 edge cases with expected behaviors
  - Complete Python→Swift component mapping
  - Testing strategy and dependencies

### Requirements Checklist
- **checklists/requirements.md** - Quality validation (16/16 items passed ✅)

---

## Implementation Plan

### Location
`specs/001-ios-migration/plan.md` (1,379 lines, 58KB)

### Plan Contents

#### Architecture
- Standard Xcode project structure (constitution compliant)
- MVVM pattern with protocol-based design
- 10 major components:
  1. **CameraManager** - AVCaptureSession, photo capture
  2. **YOLOXDetector** - CoreML detection inference
  3. **TinyCLIPEmbedder** - Visual embedding extraction
  4. **SimilarityClusterer** - Agglomerative clustering
  5. **OutlierFilter** - Size/aspect ratio filtering
  6. **HistoryManager** - Core Data persistence
  7. **AICounter** - Main orchestrator
  8. **ContentView** - Main UI
  9. **CameraView** - Clean viewfinder
  10. **HistoryView** - Session history

#### Algorithm Migration Strategies
- Size outlier filtering: median ± 1.0σ (Swift Accelerate framework)
- Aspect ratio filtering: median ± 0.5σ
- TinyCLIP L2 normalization: `vDSP_svdiv` + `cblas_snrm2`
- Custom agglomerative clustering (no external dependencies)
- IoU calculation and NMS
- Thumbnail generation (crop main object, 200×200 JPEG)

#### CoreML Integration
- **YOLOX-S**: 640×640 input → [1,8400,85] output
- **TinyCLIP**: 224×224 input → [1,256] embedding
- Lazy model loading for memory optimization
- Vision framework preprocessing

#### Core Data Schema
```swift
CountingSession {
  id: UUID
  thumbnailData: Data       // 200×200 JPEG
  count: Int16             // Main cluster count
  timestamp: Date
  confidenceThreshold: Float
  similarityThreshold: Float
  isFavorite: Bool
}
```

#### Performance Targets
- <2s total processing (iPhone 12+)
- <100ms YOLOX inference
- <20ms TinyCLIP per crop
- <150MB memory usage
- 60fps UI responsiveness
- <0.1% battery per image

#### Testing Strategy
- **Unit Tests**: 80%+ coverage (algorithms, filtering, clustering)
- **Integration Tests**: Full pipeline validation
- **UI Tests**: User flow automation
- **Performance Tests**: Instruments profiling
- **Accuracy Tests**: 95%+ vs manual count

---

## Implementation Timeline (11 Weeks)

### Phase 0: Project Setup (Week 1)
- Create Xcode project (standard structure)
- Integrate CoreML models
- Set up Core Data stack
- Configure build settings

### Phase 1: Detection Layer (Weeks 2-3)
- YOLOXDetector implementation
- Input preprocessing
- Output parsing
- Outlier filtering

### Phase 2: Clustering Layer (Weeks 4-5)
- TinyCLIPEmbedder implementation
- Custom agglomerative clustering
- Cosine similarity with Accelerate
- IoU merging

### Phase 3: Pipeline Integration (Week 6)
- AICounter orchestrator
- End-to-end pipeline
- Error handling
- Progress reporting

### Phase 4: Camera System (Week 7)
- AVCaptureSession setup
- Clean viewfinder UI
- Photo capture flow
- Permission handling

### Phase 5: Persistence Layer (Week 8)
- Core Data CRUD operations
- Thumbnail generation
- History management (max 100 entries)
- Auto-cleanup

### Phase 6: UI Implementation (Week 9)
- ContentView (main screen)
- CameraView (viewfinder)
- HistoryView (session list)
- SettingsView (thresholds)

### Phase 7: Optimization (Week 10)
- Performance profiling
- Memory optimization
- Battery efficiency
- Frame rate tuning

### Phase 8: Testing & Launch (Week 11)
- TestFlight beta
- Bug fixes
- App Store submission
- Launch preparation

---

## Key Design Decisions

### 1. "Capture First, Count Second" Workflow
- **Rationale**: Battery efficiency, clean UX, optimal performance
- **Implementation**: No real-time inference on video frames
- **Benefit**: Minimal battery drain, smooth 60fps camera preview

### 2. Main Cluster Only Display
- **Rationale**: Simplified UX, clear user intent
- **Implementation**: Show count of largest cluster only
- **Benefit**: Reduces cognitive load, clearer results

### 3. Thumbnail Storage Strategy
- **Rationale**: Privacy + storage optimization
- **Implementation**: Crop one representative object (200×200 JPEG)
- **Benefit**: <20MB for 100 entries vs. full images (>500MB)

### 4. Zero External Dependencies
- **Rationale**: Constitution compliance, app size, reliability
- **Implementation**: Custom clustering, Accelerate for math
- **Benefit**: <50MB app size, no dependency conflicts

### 5. Standard Xcode Project Structure
- **Rationale**: Constitution principle #5
- **Implementation**: Traditional Xcode layout, no SPM architecture
- **Benefit**: Familiar structure, easier onboarding

---

## Success Criteria

### Technical
- ✅ Feature parity with Python script
- ✅ <2s processing time on iPhone 12+
- ✅ <150MB memory usage
- ✅ 95%+ detection accuracy
- ✅ 60fps UI responsiveness
- ✅ <0.5% crash rate

### User Experience
- ✅ <10s total workflow (capture → count → view)
- ✅ Clean, intuitive iOS interface
- ✅ iOS Human Interface Guidelines compliant
- ✅ Accessibility support (VoiceOver, Dynamic Type)

### Quality
- ✅ 80%+ test coverage
- ✅ Zero critical bugs at launch
- ✅ TestFlight validation with 20+ users
- ✅ 4.5+ App Store rating target

---

## Next Steps

1. **Execute Phase 0**: Create Xcode project and integrate CoreML models
2. **Implement Core Algorithms**: Start with YOLOXDetector and outlier filtering
3. **Build Testing Infrastructure**: Set up XCTest framework
4. **Iterate on Pipeline**: Validate against Python reference
5. **Polish UI/UX**: Refine based on TestFlight feedback

---

## References

- **YOLOX Paper**: https://arxiv.org/abs/2107.08430
- **TinyCLIP**: https://github.com/wkcn/TinyCLIP
- **CoreML Docs**: https://developer.apple.com/documentation/coreml
- **Vision Framework**: https://developer.apple.com/documentation/vision
- **iOS HIG**: https://developer.apple.com/design/human-interface-guidelines/

---

## Contact & Ownership

- **Product Owner**: TBD
- **Tech Lead**: TBD
- **Design Lead**: TBD
- **QA Lead**: TBD

---

## Document History

| Date | Event | Details |
|------|-------|---------|
| 2025-12-09 | Constitution Created | v1.0.0 with 5 core principles |
| 2025-12-09 | Specification Complete | 001-ios-migration spec.md (24KB) |
| 2025-12-09 | Implementation Plan Complete | 001-ios-migration plan.md (58KB) |
| 2025-12-09 | Project Ready | Ready for Phase 0 execution |

---

**Project Status**: 📋 **PLANNING COMPLETE** → Ready for implementation
