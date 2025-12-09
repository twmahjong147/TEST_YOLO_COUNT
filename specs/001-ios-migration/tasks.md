---
description: "Implementation tasks for iOS AICounter migration feature"
---

# Tasks: iOS AICounter Migration

**Input**: Design documents from `/specs/001-ios-migration/`
**Prerequisites**: plan.md (required), spec.md (required for user stories)

**Tests**: OPTIONAL - Tests are NOT included by default unless explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

iOS project structure per plan.md:
- `AICounter/AICounter/` - Main app source
- `AICounter/AICounterTests/` - Test suite
- `specs/001-ios-migration/` - Design documents

---

## Phase 1: Setup (Project Initialization)

**Purpose**: Create Xcode project structure, integrate CoreML models, establish basic app foundation

- [ ] T001 Create standard Xcode project AICounter.xcodeproj (iOS 15.0+, Swift 5.9+, single target)
- [ ] T002 Convert YOLOX-S ONNX model to CoreML (yolox_s.mlpackage) using coremltools
- [ ] T003 Convert TinyCLIP ONNX model to CoreML (tinyclip_vision.mlpackage) using coremltools
- [ ] T004 Add CoreML models to Xcode project in AICounter/Models/CoreML/
- [ ] T005 [P] Create Core Data model CountingSession.xcdatamodeld in AICounter/Services/Persistence/
- [ ] T006 [P] Setup AICounterApp.swift SwiftUI entry point in AICounter/App/
- [ ] T007 [P] Configure Info.plist with camera and photo library permissions in AICounter/App/
- [ ] T008 [P] Create Assets.xcassets with app icon and color assets in AICounter/Resources/
- [ ] T009 Verify project builds successfully on simulator and device

**Checkpoint**: Xcode project builds, models load without errors, basic SwiftUI app launches

---

## Phase 2: Foundational (Core ML Infrastructure - BLOCKS ALL USER STORIES)

**Purpose**: Core ML model wrappers, protocols, and data structures that ALL user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T010 [P] Define Detection struct in AICounter/Models/Detection.swift
- [ ] T011 [P] Define CountResult struct in AICounter/Models/CountResult.swift
- [ ] T012 [P] Define ProcessingStage enum in AICounter/Models/ProcessingStage.swift
- [ ] T013 [P] Define DetectorProtocol in AICounter/Core/Detectors/DetectorProtocol.swift
- [ ] T014 [P] Define EmbedderProtocol in AICounter/Core/Embedders/EmbedderProtocol.swift
- [ ] T015 [P] Define ClustererProtocol in AICounter/Core/Clustering/ClustererProtocol.swift
- [ ] T016 [P] Implement ImageProcessor utility class in AICounter/Utils/ImageProcessor.swift
- [ ] T017 [P] Implement GeometryUtils for IoU calculations in AICounter/Utils/GeometryUtils.swift
- [ ] T018 [P] Implement Array+Statistics extensions in AICounter/Utils/Extensions/Array+Statistics.swift
- [ ] T019 [P] Implement CGImage+Extensions in AICounter/Utils/Extensions/CGImage+Extensions.swift
- [ ] T020 [P] Implement UIImage+Extensions in AICounter/Utils/Extensions/UIImage+Extensions.swift
- [ ] T021 Implement YOLOXDetector CoreML wrapper in AICounter/Core/Detectors/YOLOXDetector.swift
- [ ] T022 Implement YOLOXOutputParser for [1,8400,85] tensor in AICounter/Core/Detectors/YOLOXOutputParser.swift
- [ ] T023 Implement TinyCLIPEmbedder CoreML wrapper in AICounter/Core/Embedders/TinyCLIPEmbedder.swift
- [ ] T024 Implement EmbeddingNormalizer (L2 normalization with Accelerate) in AICounter/Core/Embedders/EmbeddingNormalizer.swift
- [ ] T025 Verify YOLOX model loads and produces detections on test image
- [ ] T026 Verify TinyCLIP model loads and produces embeddings on test crop

**Checkpoint**: Foundation ready - all protocols defined, ML models operational, utilities functional

---

## Phase 3: User Story 1 - Camera-Based Object Counting (Priority: P1) 🎯 MVP

**Goal**: User can capture photo with camera and get accurate count of main object type within 2 seconds

**Independent Test**: Launch app → Capture photo of 10 bottles → Verify count displays "10" within 2 seconds

### Core Counting Pipeline for US1

- [ ] T027 [P] [US1] Implement OutlierFilter size filtering in AICounter/Core/Filtering/OutlierFilter.swift
- [ ] T028 [P] [US1] Implement StatisticalUtils (median, std) in AICounter/Core/Filtering/StatisticalUtils.swift
- [ ] T029 [US1] Implement CosineSimilarity (Accelerate vDSP) in AICounter/Core/Clustering/CosineSimilarity.swift
- [ ] T030 [US1] Implement SimilarityClusterer (agglomerative) in AICounter/Core/Clustering/SimilarityClusterer.swift
- [ ] T031 [US1] Implement ClusterAnalyzer (identify largest) in AICounter/Core/Clustering/ClusterAnalyzer.swift
- [ ] T032 [US1] Implement NMSFilter (IoU-based merging) in AICounter/Core/Filtering/NMSFilter.swift
- [ ] T033 [US1] Implement AICounter orchestrator count() method in AICounter/Core/AICounter.swift
- [ ] T034 [US1] Add filterSizeOutliers() stage to AICounter pipeline
- [ ] T035 [US1] Add filterAspectRatioOutliers() stage to AICounter pipeline
- [ ] T036 [US1] Add cropDetections() and embeddings stage to AICounter pipeline
- [ ] T037 [US1] Add clustering stage to AICounter pipeline
- [ ] T038 [US1] Add IoU merging stage to AICounter pipeline
- [ ] T039 [US1] Add result generation stage to AICounter pipeline
- [ ] T040 [US1] Add async/await with progress reporting to AICounter

### Camera Integration for US1

- [ ] T041 [P] [US1] Implement CameraManager AVCaptureSession setup in AICounter/Services/Camera/CameraManager.swift
- [ ] T042 [P] [US1] Implement PhotoCaptureDelegate in AICounter/Services/Camera/PhotoCaptureDelegate.swift
- [ ] T043 [US1] Add camera permission handling to CameraManager
- [ ] T044 [US1] Add photo capture method to CameraManager
- [ ] T045 [US1] Implement CameraPreviewLayer SwiftUI wrapper in AICounter/Services/Camera/CameraPreviewLayer.swift
- [ ] T046 [US1] Implement CameraView SwiftUI interface in AICounter/Views/Camera/CameraView.swift
- [ ] T047 [US1] Implement CameraControlsView (shutter, flash) in AICounter/Views/Camera/CameraControlsView.swift

### UI for US1

- [ ] T048 [P] [US1] Implement ContentView main screen layout in AICounter/Views/Main/ContentView.swift
- [ ] T049 [P] [US1] Implement ProcessingStatusView overlay in AICounter/Views/Main/ProcessingStatusView.swift
- [ ] T050 [P] [US1] Implement CountDisplayView component in AICounter/Views/Components/CountDisplayView.swift
- [ ] T051 [P] [US1] Implement ErrorAlertView component in AICounter/Views/Components/ErrorAlertView.swift
- [ ] T052 [US1] Wire camera capture flow in ContentView
- [ ] T053 [US1] Wire processing pipeline invocation from ContentView
- [ ] T054 [US1] Wire result display in ContentView
- [ ] T055 [US1] Add error handling UI for camera permission denied
- [ ] T056 [US1] Add error handling UI for no objects detected
- [ ] T057 [US1] Add error handling UI for processing failures

**Checkpoint**: User Story 1 complete - camera capture, object counting, result display all functional

---

## Phase 4: User Story 2 - Photo Library Processing (Priority: P1)

**Goal**: User can select existing photo from library and get accurate count without retaking photo

**Independent Test**: Select pre-existing photo of 20 coins from library → Verify app correctly counts them

### Implementation for US2

- [ ] T058 [P] [US2] Import PhotosUI framework in AICounter project
- [ ] T059 [P] [US2] Add photo library permission to Info.plist
- [ ] T060 [US2] Implement PHPickerViewController integration in ContentView
- [ ] T061 [US2] Add "Select from Library" button to ContentView
- [ ] T062 [US2] Add image loading from PHPicker result
- [ ] T063 [US2] Wire photo library selection to processing pipeline
- [ ] T064 [US2] Add error handling for photo library permission denied
- [ ] T065 [US2] Add image resolution validation (handle 4K images)
- [ ] T066 [US2] Test high object count scenarios (30+ objects)

**Checkpoint**: User Stories 1 and 2 complete - both camera and library inputs work independently

---

## Phase 5: User Story 3 - Review Counting History (Priority: P2)

**Goal**: User can review past 100 counting sessions with thumbnails, counts, and timestamps

**Independent Test**: Perform 5 counting operations → Open History → Verify all 5 entries with thumbnails → Delete one entry

### Persistence Infrastructure for US3

- [ ] T067 [P] [US3] Implement PersistenceController Core Data stack in AICounter/Services/Persistence/PersistenceController.swift
- [ ] T068 [P] [US3] Define CountingSession Core Data entity in CountingSession.xcdatamodeld
- [ ] T069 [P] [US3] Create CountingSession+CoreData extensions in AICounter/Models/CountingSession+CoreData.swift
- [ ] T070 [US3] Implement ThumbnailGenerator cropping logic in AICounter/Services/Persistence/ThumbnailGenerator.swift
- [ ] T071 [US3] Implement ThumbnailGenerator resize to 200×200 in ThumbnailGenerator.swift
- [ ] T072 [US3] Implement ThumbnailGenerator JPEG compression (80%) in ThumbnailGenerator.swift
- [ ] T073 [US3] Implement HistoryManager saveSession() in AICounter/Services/Persistence/HistoryManager.swift
- [ ] T074 [US3] Implement HistoryManager fetchHistory() in HistoryManager.swift
- [ ] T075 [US3] Implement HistoryManager deleteSession() in HistoryManager.swift
- [ ] T076 [US3] Implement HistoryManager clearAll() in HistoryManager.swift
- [ ] T077 [US3] Implement FIFO cleanup (max 100 entries) in HistoryManager.swift
- [ ] T078 [US3] Wire saveSession() call after successful counting in AICounter.swift

### History UI for US3

- [ ] T079 [P] [US3] Implement HistoryView list interface in AICounter/Views/History/HistoryView.swift
- [ ] T080 [P] [US3] Implement HistoryRowView thumbnail + count display in AICounter/Views/History/HistoryRowView.swift
- [ ] T081 [P] [US3] Implement SessionDetailView full details in AICounter/Views/History/SessionDetailView.swift
- [ ] T082 [US3] Add history navigation from ContentView
- [ ] T083 [US3] Wire fetchHistory() to HistoryView data source
- [ ] T084 [US3] Add swipe-to-delete gesture in HistoryView
- [ ] T085 [US3] Add "Clear All History" button with confirmation dialog
- [ ] T086 [US3] Add reverse chronological sorting in HistoryView
- [ ] T087 [US3] Test 100-entry limit and FIFO cleanup

**Checkpoint**: User Stories 1, 2, and 3 complete - full capture, library, and history functionality

---

## Phase 6: User Story 4 - Adjust Detection Settings (Priority: P3)

**Goal**: Power user can fine-tune detection sensitivity for different scenarios

**Independent Test**: Process same image twice - default settings (0.80) shows 15 objects, strict settings (0.90) shows 12 objects

### Settings Infrastructure for US4

- [ ] T088 [P] [US4] Define AppSettings struct in AICounter/Services/Settings/AppSettings.swift
- [ ] T089 [P] [US4] Implement SettingsManager UserDefaults wrapper in AICounter/Services/Settings/SettingsManager.swift
- [ ] T090 [US4] Implement getSettings() in SettingsManager
- [ ] T091 [US4] Implement updateSettings() in SettingsManager
- [ ] T092 [US4] Implement resetToDefaults() in SettingsManager
- [ ] T093 [US4] Wire settings loading in AICounter orchestrator

### Settings UI for US4

- [ ] T094 [P] [US4] Implement SettingsView main interface in AICounter/Views/Settings/SettingsView.swift
- [ ] T095 [P] [US4] Implement ThresholdSliderView reusable component in AICounter/Views/Settings/ThresholdSliderView.swift
- [ ] T096 [US4] Add confidence threshold slider (0.1 - 0.9) to SettingsView
- [ ] T097 [US4] Add similarity threshold slider (0.6 - 0.95) to SettingsView
- [ ] T098 [US4] Add size filter toggle to SettingsView
- [ ] T099 [US4] Add aspect ratio filter toggle to SettingsView
- [ ] T100 [US4] Add area filter toggle to SettingsView
- [ ] T101 [US4] Add "Reset to Defaults" button to SettingsView
- [ ] T102 [US4] Wire settings navigation from ContentView
- [ ] T103 [US4] Test settings persistence across app launches
- [ ] T104 [US4] Test parameter changes affect counting results

**Checkpoint**: User Stories 1-4 complete - full feature set with customization

---

## Phase 7: User Story 5 - Visualize Detection Overlay (Priority: P3)

**Goal**: User can see which objects were detected with bounding boxes overlaid on image

**Independent Test**: Process image with 10 bottles + 5 cans → View overlay showing 10 green boxes (main cluster) and 5 blue boxes (secondary)

### Overlay Implementation for US5

- [ ] T105 [P] [US5] Implement ImagePreviewView zoomable viewer in AICounter/Views/Components/ImagePreviewView.swift
- [ ] T106 [US5] Implement bounding box rendering with Core Graphics in ImagePreviewView
- [ ] T107 [US5] Add color differentiation (green for main cluster, blue for others)
- [ ] T108 [US5] Add confidence score labels on bounding boxes
- [ ] T109 [US5] Add tap gesture to show detection details
- [ ] T110 [US5] Add pinch-to-zoom gesture handling
- [ ] T111 [US5] Add pan gesture for zoomed navigation
- [ ] T112 [US5] Optimize rendering for 50+ bounding boxes (60fps target)
- [ ] T113 [US5] Wire "Show Details" button from ContentView result display
- [ ] T114 [US5] Test overlay with high object counts (50+)
- [ ] T115 [US5] Test zoom and pan performance

**Checkpoint**: All 5 user stories complete - full feature set ready for polish

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories, final quality assurance

- [ ] T116 [P] Run Instruments Time Profiler to identify bottlenecks
- [ ] T117 [P] Run Instruments Memory Graph to detect leaks
- [ ] T118 [P] Optimize YOLOX inference to <100ms target
- [ ] T119 [P] Optimize TinyCLIP inference to <20ms per crop target
- [ ] T120 [P] Optimize clustering to <200ms for 50 objects
- [ ] T121 [P] Verify total pipeline <2s on iPhone 12+
- [ ] T122 [P] Verify memory usage <150MB during processing
- [ ] T123 [P] Verify memory usage <50MB when idle
- [ ] T124 [P] Add VoiceOver accessibility labels to all views
- [ ] T125 [P] Test Dynamic Type support for text scaling
- [ ] T126 [P] Verify minimum contrast ratios for accessibility
- [ ] T127 [P] Polish SwiftUI animations and transitions (60fps)
- [ ] T128 [P] Test camera preview at 30fps minimum
- [ ] T129 Add model lazy loading on app launch
- [ ] T130 Add model unloading when app backgrounds
- [ ] T131 Test accuracy on 50-image validation set (≥95% target)
- [ ] T132 Test edge case: empty image (no detections)
- [ ] T133 Test edge case: single object detection
- [ ] T134 Test edge case: 100+ objects
- [ ] T135 Test edge case: highly overlapping objects (>80% IoU)
- [ ] T136 Test edge case: extreme lighting conditions
- [ ] T137 Test edge case: low resolution images (<640px)
- [ ] T138 Test on iPhone X (A11 Bionic minimum device)
- [ ] T139 Test on iPhone 12+ (target device)
- [ ] T140 Stress test: 100 consecutive counting operations
- [ ] T141 Battery consumption test: <2% for 10 sessions
- [ ] T142 Create App Store screenshots
- [ ] T143 Write App Store description
- [ ] T144 Draft privacy policy (on-device processing statement)
- [ ] T145 Setup TestFlight beta build

**Checkpoint**: Production-ready quality achieved

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational completion
- **User Story 2 (Phase 4)**: Depends on Foundational completion, builds on US1 components
- **User Story 3 (Phase 5)**: Depends on Foundational completion, integrates with US1/US2 counting
- **User Story 4 (Phase 6)**: Depends on Foundational completion, modifies US1/US2/US3 behavior
- **User Story 5 (Phase 7)**: Depends on Foundational completion, adds visualization to US1/US2
- **Polish (Phase 8)**: Depends on desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Depends only on Foundational (Phase 2) - Core counting functionality
- **User Story 2 (P1)**: Depends on Foundational, uses same counting pipeline as US1 - Library input
- **User Story 3 (P2)**: Depends on Foundational, integrates with counting results from US1/US2 - History
- **User Story 4 (P3)**: Depends on Foundational, modifies behavior of US1/US2/US3 - Settings
- **User Story 5 (P3)**: Depends on Foundational, enhances results from US1/US2 - Overlay

### Within Each User Story

- Protocol definitions before implementations
- Models and utilities before services
- Services before UI components
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

**Within Setup (Phase 1):**
- T002, T003 (model conversions) can run in parallel
- T005, T006, T007, T008 can run in parallel after project creation

**Within Foundational (Phase 2):**
- T010-T012 (model structs) can run in parallel
- T013-T015 (protocols) can run in parallel
- T016-T020 (utilities) can run in parallel

**Across User Stories:**
- After Foundational completes, User Stories 1 and 2 can start in parallel (different files)
- User Story 3 can start after Foundational (separate persistence layer)
- User Story 4 can start after Foundational (separate settings layer)
- User Story 5 can start after Foundational (separate overlay layer)

**Within User Story 1:**
- T027, T028 (filtering) can run in parallel
- T041, T042 (camera infrastructure) can run in parallel
- T048, T049, T050, T051 (UI components) can run in parallel

**Within User Story 3:**
- T067, T068, T069 (Core Data) can run in parallel
- T079, T080, T081 (history UI) can run in parallel

**Within User Story 4:**
- T088, T089 (settings infrastructure) can run in parallel
- T094, T095 (settings UI) can run in parallel

**Within Polish (Phase 8):**
- T116-T128 (profiling and optimization) can run in parallel
- T132-T137 (edge case tests) can run in parallel
- T142-T144 (App Store prep) can run in parallel

---

## Parallel Example: User Story 1

```bash
# Launch all filtering components together:
Task: "Implement OutlierFilter size filtering in AICounter/Core/Filtering/OutlierFilter.swift"
Task: "Implement StatisticalUtils (median, std) in AICounter/Core/Filtering/StatisticalUtils.swift"

# Launch all UI components together:
Task: "Implement ContentView main screen layout in AICounter/Views/Main/ContentView.swift"
Task: "Implement ProcessingStatusView overlay in AICounter/Views/Main/ProcessingStatusView.swift"
Task: "Implement CountDisplayView component in AICounter/Views/Components/CountDisplayView.swift"
Task: "Implement ErrorAlertView component in AICounter/Views/Components/ErrorAlertView.swift"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup → Xcode project created, models integrated
2. Complete Phase 2: Foundational → ML infrastructure operational
3. Complete Phase 3: User Story 1 → Camera capture and counting functional
4. **STOP and VALIDATE**: Test camera capture, verify count accuracy on 10 test images
5. Deploy TestFlight build for early feedback

### Incremental Delivery

1. Setup + Foundational → Foundation ready (2-3 days)
2. Add User Story 1 → Test independently → MVP ACHIEVED (3-5 days)
3. Add User Story 2 → Test independently → Library support added (1-2 days)
4. Add User Story 3 → Test independently → History tracking added (2-3 days)
5. Add User Story 4 → Test independently → Power user features (1-2 days)
6. Add User Story 5 → Test independently → Visual verification (1-2 days)
7. Polish Phase → Production quality (3-5 days)

**Total Estimated Timeline**: 13-22 days for complete feature set

### Parallel Team Strategy

With 2 developers:

1. Both complete Setup + Foundational together (2-3 days)
2. Once Foundational is done:
   - Developer A: User Story 1 (camera counting)
   - Developer B: User Story 3 (history infrastructure)
3. Then:
   - Developer A: User Story 2 (library input)
   - Developer B: User Story 4 (settings)
4. Finally:
   - Developer A: User Story 5 (overlay)
   - Developer B: Polish and testing

---

## Summary

- **Total Tasks**: 145 tasks
- **User Stories**: 5 stories (US1-US5)
- **Task Count per Story**:
  - Setup (Phase 1): 9 tasks
  - Foundational (Phase 2): 17 tasks (BLOCKS all stories)
  - User Story 1 (P1): 31 tasks - Camera counting
  - User Story 2 (P1): 9 tasks - Library selection
  - User Story 3 (P2): 21 tasks - History tracking
  - User Story 4 (P3): 17 tasks - Settings customization
  - User Story 5 (P3): 11 tasks - Detection overlay
  - Polish (Phase 8): 30 tasks

- **Parallel Opportunities**: 40+ tasks marked [P] can run in parallel within phases
- **Independent Test Criteria**: Each user story has specific validation method
- **MVP Scope**: User Story 1 only (camera-based counting) = 57 tasks (Setup + Foundational + US1)

---

## Notes

- [P] tasks = different files, no dependencies within phase
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- **Tests NOT included** - only implementation tasks (per spec: no explicit test request)
- All paths assume standard Xcode project structure per plan.md
