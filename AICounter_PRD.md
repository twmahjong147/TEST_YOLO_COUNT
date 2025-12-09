# Product Requirements Document (PRD)
## AICounter - AI-Powered Object Counting iOS Application

**Version:** 1.0  
**Date:** December 9, 2025  
**Status:** Ready for Development  
**Author:** Product Team

---

## Executive Summary

AICounter is an iOS application that uses advanced computer vision and machine learning to accurately count the most prominent objects in images. By combining object detection (YOLOX-S) with visual similarity clustering (TinyCLIP), the app identifies and counts the main object type, providing a simple and clear count result.

### Key Value Proposition
- **No vocabulary needed**: Works with any object type without pre-training
- **Offline operation**: All ML models run on-device
- **High accuracy**: Visual similarity clustering eliminates false positives
- **Fast processing**: Optimized for Apple Neural Engine (< 2 seconds per image)

---

## 1. Product Overview

### 1.1 Problem Statement
Users need to count similar objects in images but face challenges:
- Manual counting is tedious and error-prone
- Traditional object detectors count all objects, not just similar ones
- Cloud-based solutions require internet and raise privacy concerns
- Existing apps require specific object categories (limited vocabulary)

### 1.2 Solution
AICounter uses a two-stage AI pipeline:
1. **Stage 1**: Detect all objects using YOLOX-S (class-agnostic detection)
2. **Stage 2**: Group by visual similarity using TinyCLIP embeddings
3. **Stage 3**: Count the largest cluster of similar objects
4. **Stage 4**: Display only the main cluster count (simplified UX)

This approach works for any object type without requiring pre-defined categories, and presents results in a simple, easy-to-understand format showing only the count of the most prominent object type.

### 1.3 Target Users
- **Inventory managers**: Count products, stock items, parts
- **Retail workers**: Count items on shelves, in boxes
- **Warehouse operators**: Count pallets, packages
- **Researchers**: Count specimens, samples in lab environments
- **Event organizers**: Count attendees, seats, equipment
- **Hobbyists**: Count collectibles, items in photos

---

## 2. Core Features

### 2.1 Essential Features (MVP - 4 Features)

#### Feature 1: Dual Input Support (Camera + Photo Library)
**Description**: Users can capture images directly using the device camera or select existing images from their photo library.

**User Stories**:
- As a user, I want to capture a photo directly in the app so I can count objects immediately
- As a user, I want to select an existing photo from my library so I can count objects in images I've already taken
- As a user, I want a clean camera viewfinder without distracting overlays
- As a user, I want to see a preview of my selected/captured image before processing

**Acceptance Criteria**:

**Camera Capture**:
- ✅ User can tap "Capture Photo" button to open camera view
- ✅ Live camera preview displays cleanly without real-time overlays
- ✅ Camera includes standard shutter button control
- ✅ Flash and focus controls available
- ✅ Captured photo displays for review before processing
- ✅ Works in both portrait and landscape orientations
- ✅ No real-time inference (static inference only)

**Photo Library Selection**:
- ✅ User can tap "Select from Library" button
- ✅ iOS photo picker (PHPickerViewController) opens with access to user's library
- ✅ Selected image displays in the app with proper aspect ratio
- ✅ Images up to 4K resolution are supported
- ✅ Support all iOS image formats (JPEG, PNG, HEIC)

**Workflow Design ("Capture First, Count Second")**:
- ✅ Live camera feed shows clean viewfinder only (no bounding boxes, labels, or AR overlays)
- ✅ Object detection triggered ONLY after shutter button is pressed or image is selected
- ✅ Processing happens on static captured/selected image, not on live video frames
- ✅ This ensures clean UX, minimal battery drain, and optimal performance

**Technical Requirements**:
- Use `AVCaptureSession` for live camera feed
- Use `PHPickerViewController` for photo library access
- Handle camera and photo library permissions gracefully
- No real-time inference on video frames (battery optimization)
- Process only static images after capture/selection

---

#### Feature 2: AI-Powered Object Counting
**Description**: The core counting engine that analyzes images and returns accurate counts.

**User Stories**:
- As a user, I want to tap "Count Objects" and get an accurate count of similar items in my image
- As a user, I want the counting to complete in under 2 seconds on modern devices

**Acceptance Criteria**:
- ✅ Processing completes in < 2 seconds on iPhone 12+
- ✅ Accuracy matches or exceeds Python reference implementation
- ✅ Handles images with 5-100+ objects
- ✅ Works with various lighting conditions and backgrounds

**Technical Requirements**:
- YOLOX-S model for object detection (640×640 input)
- TinyCLIP vision encoder for embeddings (224×224 input)
- Confidence threshold: 0.25 (configurable)
- Similarity threshold: 0.80 (configurable)
- Size outlier filtering: ±3 standard deviations

**Processing Pipeline**:
```
Image Input
    ↓
Stage 1: Object Detection (YOLOX-S)
    ↓
Stage 1.5: Size Outlier Filtering
    ↓
Stage 1.6: Aspect Ratio Filtering
    ↓
Stage 2: Visual Embedding Extraction (TinyCLIP)
    ↓
Stage 3: Similarity Clustering
    ↓
Stage 4: Count Largest Cluster
    ↓
Result Display
```

---

#### Feature 3: Result Display
**Description**: Clear visualization of the main cluster count.

**User Stories**:
- As a user, I want to see the final count of the main object type prominently displayed
- As a user, I want to quickly understand how many similar objects were detected
- As a user, I want clear feedback during processing

**Acceptance Criteria**:
- ✅ Final count of main cluster displayed in large, readable font
- ✅ Count represents only the largest cluster (main object type)
- ✅ Processing status shown during analysis
- ✅ Error messages displayed clearly if processing fails
- ✅ Simple, clean result display without technical details

**UI Components**:
- Large number display for final count (main cluster only)
- Processing indicator (spinner/progress)
- Error alert dialog
- Success/completion indicator

**Note**: The app focuses on counting the most prominent object type in the image. It does not display the number of different clusters/categories detected - only the count of the main cluster.

---

#### Feature 4: Counting History
**Description**: Keep track of past counting sessions for review and reference.

**User Stories**:
- As a user, I want to review my previous counting results so I can track my work
- As a user, I want to see when each count was performed
- As a user, I want to view thumbnails of counted images for quick reference
- As a user, I want to delete old entries I no longer need

**Acceptance Criteria**:
- ✅ History view accessible from main screen (list/grid icon)
- ✅ Shows last 100 counting sessions
- ✅ Each entry displays:
  - Image thumbnail
  - Final count number (main cluster only)
  - Date and time of counting
- ✅ Tap entry to view full details and original result
- ✅ Swipe to delete individual entries
- ✅ "Clear All" option with confirmation dialog
- ✅ Empty state message when no history exists
- ✅ Data persists between app launches

**Data Storage**:
- ✅ Use Core Data for persistent storage
- ✅ Store: thumbnail (cropped main object), count, timestamp, parameters used
- ✅ Automatic cleanup: Keep max 100 entries (oldest deleted first)
- ✅ Thumbnail: 200×200 px cropped image of a representative object from main cluster (compressed JPEG)
- ✅ No full-resolution images stored (privacy + storage optimization)

**Technical Requirements**:
- Core Data stack implementation
- `CountingSession` entity with attributes:
  - `id: UUID`
  - `thumbnailData: Data` (compressed JPEG of cropped main object)
  - `count: Int` (main cluster count only)
  - `timestamp: Date`
  - `confidenceThreshold: Float`
  - `similarityThreshold: Float`
- Migration strategy for schema updates
- Background context for saving (non-blocking)
- **Thumbnail generation algorithm**:
  1. Identify main cluster (largest group of similar objects)
  2. Select one representative detection from main cluster (e.g., first or most centered)
  3. Crop bounding box region from original image
  4. Resize cropped region to 200×200 px (maintaining aspect ratio with padding if needed)
  5. Compress to JPEG format (quality: 80%)
  6. Store as Data in Core Data

---

### 2.2 Advanced Features (Post-MVP - 4 Features)

#### Feature 5: Visual Detection Overlay
**Description**: Show bounding boxes and highlights on detected objects.

**User Stories**:
- As a user, I want to see which objects were detected and counted
- As a user, I want to visually verify that the AI counted the correct objects

**Acceptance Criteria**:
- ✅ Bounding boxes drawn on detected objects
- ✅ Objects in largest cluster highlighted in green
- ✅ Other objects shown in different color (blue)
- ✅ Confidence scores displayed on each box
- ✅ Zoom and pan supported on annotated image

---

#### Feature 6: Adjustable Thresholds
**Description**: Allow users to fine-tune detection and clustering parameters.

**User Stories**:
- As a power user, I want to adjust sensitivity for different scenarios
- As a user, I want to see real-time updates when changing parameters

**Acceptance Criteria**:
- ✅ Confidence threshold slider (0.1 - 0.9)
- ✅ Similarity threshold slider (0.6 - 0.95)
- ✅ Outlier filtering toggle (on/off)
- ✅ Reset to defaults button
- ✅ Parameter presets: "Precise", "Balanced", "Generous"

---

#### Feature 7: Batch Processing
**Description**: Process multiple images and export results.

**User Stories**:
- As a user, I want to select multiple images and count objects in all of them
- As a user, I want to export counting results to CSV for analysis

**Acceptance Criteria**:
- ✅ Select up to 50 images at once
- ✅ Progress indicator shows X of Y processed
- ✅ Results table with image name and count
- ✅ Export to CSV with timestamps
- ✅ Share results via iOS share sheet

---

#### Feature 8: Favorites
**Description**: Mark and save favorite counting sessions for quick access.

**User Stories**:
- As a user, I want to mark important counting sessions as favorites
- As a user, I want to quickly access my favorite counts
- As a user, I want to filter history to show only favorites

**Acceptance Criteria**:
- ✅ Star icon to mark entries as favorite
- ✅ Favorites section in history view
- ✅ Filter toggle: All / Favorites only
- ✅ Favorite status persists
- ✅ Unfavorite with confirmation

**Data Storage**:
- ✅ Add `isFavorite: Bool` to `CountingSession` entity
- ✅ Query support for filtering favorites

---

## 3. Core Algorithm Pseudocode

This section documents the exact algorithms from `count_objects_yolox.py` that must be replicated in Swift.

### 3.1 Main Counting Pipeline

```python
def count_objects(image):
    """
    Main counting pipeline - must be implemented exactly as Python version.
    """
    # Stage 1: Object Detection
    detections = yolox_detect(image, conf_threshold=0.25, nms_threshold=0.45)
    print(f"Stage 1: Detected {len(detections)} objects")
    
    # Stage 1.5: Size Outlier Filtering
    detections = remove_size_outliers(detections, std_threshold=3.0)
    print(f"Stage 1.5: {len(detections)} after size filtering")
    
    # Stage 1.6: Aspect Ratio Filtering
    detections = remove_aspect_ratio_outliers(detections, std_threshold=1.0)
    print(f"Stage 1.6: {len(detections)} after aspect ratio filtering")
    
    # Stage 2: Extract Embeddings
    crops = extract_crops(image, detections)
    crops = filter_by_area_consistency(crops, std_threshold=2.0)
    embeddings = extract_embeddings(crops)  # TinyCLIP
    print(f"Stage 2: Extracted {len(embeddings)} embeddings")
    
    # Stage 3: Similarity Clustering
    cluster_labels = cluster_by_similarity(embeddings, similarity_threshold=0.80)
    cluster_counts = count_clusters(cluster_labels)
    print(f"Stage 3: Found {len(cluster_counts)} clusters")
    
    # Stage 4: IoU Merging
    detections = merge_overlapping_detections(detections, cluster_labels, iou_threshold=0.5)
    
    # Stage 5: Result Generation
    largest_cluster_id = find_largest_cluster(cluster_counts)
    final_count = cluster_counts[largest_cluster_id]
    
    return final_count, detections, cluster_labels
```

### 3.2 Algorithm 1: Size Outlier Removal

```python
def remove_size_outliers(boxes, std_threshold=3.0):
    """
    Remove boxes that are size outliers based on area.
    
    Algorithm:
        1. Calculate area for each box: area = (x2 - x1) * (y2 - y1)
        2. Compute median_area = median(all_areas)
        3. Compute std_area = standard_deviation(all_areas)
        4. Set bounds:
           min_area = median_area - (std_threshold * std_area)
           max_area = median_area + (std_threshold * std_area)
        5. Keep only boxes where min_area <= area <= max_area
    
    Args:
        boxes: Array of shape (N, 4) with format [x1, y1, x2, y2]
        std_threshold: Number of standard deviations from median (default: 3.0)
    
    Returns:
        Filtered list of box indices to keep
    """
    if len(boxes) < 3:
        return list(range(len(boxes)))  # Not enough data for statistics
    
    # Calculate areas
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
    
    # Calculate statistics
    median_area = median(areas)
    std_area = standard_deviation(areas)
    
    # Calculate bounds
    min_area = median_area - (std_threshold * std_area)
    max_area = median_area + (std_threshold * std_area)
    
    # Filter
    keep_indices = []
    for i, area in enumerate(areas):
        if min_area <= area <= max_area:
            keep_indices.append(i)
    
    return keep_indices
```

### 3.3 Algorithm 2: Aspect Ratio Outlier Removal

```python
def remove_aspect_ratio_outliers(boxes, std_threshold=1.0):
    """
    Remove boxes with unusual aspect ratios.
    
    Algorithm:
        1. Calculate aspect ratio for each box: ratio = width / height
        2. Compute median_ratio = median(all_ratios)
        3. Compute std_ratio = standard_deviation(all_ratios)
        4. Set bounds:
           min_ratio = median_ratio - (std_threshold * std_ratio)
           max_ratio = median_ratio + (std_threshold * std_ratio)
        5. Keep only boxes where min_ratio <= ratio <= max_ratio
    
    Args:
        boxes: Array of shape (N, 4) with format [x1, y1, x2, y2]
        std_threshold: Number of standard deviations from median (default: 1.0)
    
    Returns:
        Filtered list of box indices to keep
    """
    if len(boxes) <= 2:
        return list(range(len(boxes)))
    
    # Calculate aspect ratios
    aspect_ratios = []
    valid_indices = []
    
    for i, box in enumerate(boxes):
        width = box[2] - box[0]
        height = box[3] - box[1]
        
        if height > 0:  # Avoid division by zero
            ratio = width / height
            aspect_ratios.append(ratio)
            valid_indices.append(i)
    
    # Calculate statistics
    median_ratio = median(aspect_ratios)
    std_ratio = standard_deviation(aspect_ratios)
    
    if std_ratio == 0:
        return valid_indices
    
    # Calculate bounds
    min_ratio = median_ratio - (std_threshold * std_ratio)
    max_ratio = median_ratio + (std_threshold * std_ratio)
    
    # Filter
    keep_indices = []
    for i, ratio in zip(valid_indices, aspect_ratios):
        if min_ratio <= ratio <= max_ratio:
            keep_indices.append(i)
    
    return keep_indices
```

### 3.4 Algorithm 3: Area Consistency Filtering

```python
def filter_by_area_consistency(crops, std_threshold=2.0):
    """
    Filter crops to keep only those with consistent areas before embedding extraction.
    
    Algorithm:
        1. Extract area from each crop
        2. Compute median_area = median(all_areas)
        3. Compute std_area = standard_deviation(all_areas)
        4. Set bounds:
           min_area = median_area - (std_threshold * std_area)
           max_area = median_area + (std_threshold * std_area)
        5. Keep only crops where min_area <= area <= max_area
    
    Args:
        crops: List of crop dictionaries with 'area' field
        std_threshold: Number of standard deviations from median (default: 2.0)
    
    Returns:
        Filtered list of crops
    """
    if len(crops) <= 2:
        return crops
    
    # Extract areas
    areas = [crop['area'] for crop in crops]
    
    # Calculate statistics
    median_area = median(areas)
    std_area = standard_deviation(areas)
    
    if std_area == 0:
        return crops
    
    # Calculate bounds
    min_area = median_area - (std_threshold * std_area)
    max_area = median_area + (std_threshold * std_area)
    
    # Filter
    filtered_crops = []
    for crop in crops:
        if min_area <= crop['area'] <= max_area:
            filtered_crops.append(crop)
    
    return filtered_crops
```

### 3.5 Algorithm 4: TinyCLIP Embedding Extraction

```python
def extract_visual_embedding(crop_image):
    """
    Extract normalized visual embedding using TinyCLIP vision encoder.
    
    Algorithm:
        1. Convert crop from BGR to RGB: crop_rgb = cvtColor(crop_bgr, BGR2RGB)
        2. Convert to PIL Image: pil_image = Image.fromarray(crop_rgb)
        3. Preprocess with CLIP processor: inputs = processor(images=pil_image)
        4. Extract features: features = model.get_image_features(inputs)
        5. L2 normalize: normalized = features / L2_norm(features)
        6. Return as numpy array
    
    Args:
        crop_image: Cropped image region (numpy array, BGR format)
    
    Returns:
        Normalized embedding vector (256 dimensions for TinyCLIP-ViT-8M-16)
    """
    # Convert BGR to RGB
    crop_rgb = cvtColor(crop_image, COLOR_BGR2RGB)
    
    # Convert to PIL
    pil_image = PIL.Image.fromarray(crop_rgb)
    
    # Preprocess
    inputs = clip_processor(images=pil_image, return_tensors="pt")
    
    # Extract features (no gradients needed)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
        
        # L2 normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # Convert to numpy
    embedding = image_features.cpu().numpy()[0]
    
    return embedding  # Shape: (256,)
```

### 3.6 Algorithm 5: Similarity-Based Clustering

```python
def cluster_by_similarity(embeddings, similarity_threshold=0.80):
    """
    Cluster embeddings using agglomerative clustering with cosine similarity.
    
    Algorithm:
        1. Convert similarity threshold to distance:
           distance_threshold = 1 - similarity_threshold
        2. Use AgglomerativeClustering with:
           - metric='cosine'
           - linkage='average'
           - distance_threshold (not n_clusters)
        3. Fit and predict cluster labels
        4. Count objects per cluster
    
    Args:
        embeddings: Array of shape (N, embedding_dim)
        similarity_threshold: Minimum cosine similarity for same cluster (0-1)
    
    Returns:
        cluster_labels: Array of cluster IDs for each embedding
        cluster_counts: Dictionary mapping cluster_id -> count
    """
    # Convert similarity to distance
    distance_threshold = 1.0 - similarity_threshold
    
    # Create clustering model
    clustering = AgglomerativeClustering(
        n_clusters=None,                    # Determined by distance_threshold
        distance_threshold=distance_threshold,
        metric='cosine',
        linkage='average'
    )
    
    # Fit and predict
    cluster_labels = clustering.fit_predict(embeddings)
    
    # Count objects per cluster
    unique_labels, counts = unique_with_counts(cluster_labels)
    cluster_counts = dict(zip(unique_labels, counts))
    
    return cluster_labels, cluster_counts
```

### 3.7 Algorithm 6: IoU Calculation

```python
def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Algorithm:
        1. Find intersection rectangle:
           x1_inter = max(box1[0], box2[0])
           y1_inter = max(box1[1], box2[1])
           x2_inter = min(box1[2], box2[2])
           y2_inter = min(box1[3], box2[3])
        
        2. Check if boxes intersect:
           if x2_inter <= x1_inter or y2_inter <= y1_inter:
               return 0.0
        
        3. Calculate intersection area:
           inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        4. Calculate union area:
           area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
           area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
           union_area = area1 + area2 - inter_area
        
        5. Calculate IoU:
           iou = inter_area / union_area
    
    Args:
        box1, box2: Bounding boxes in format [x1, y1, x2, y2]
    
    Returns:
        IoU value in range [0, 1]
    """
    # Intersection coordinates
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Check if boxes intersect
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    # Calculate areas
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    # Calculate IoU
    iou = inter_area / union_area if union_area > 0 else 0.0
    
    return iou
```

### 3.8 Algorithm 7: IoU-Based Detection Merging

```python
def merge_overlapping_detections(detections, iou_threshold=0.5):
    """
    Merge detections that overlap significantly (high IoU).
    Keep detection with higher confidence when merging.
    
    Algorithm:
        1. Sort detections by confidence (descending)
        2. Initialize keep list and suppressed set
        3. For each detection (highest confidence first):
           a. If already suppressed, skip
           b. Add to keep list
           c. For all remaining detections:
              - Calculate IoU with current detection
              - If IoU > threshold: mark as suppressed
        4. Return kept detections
    
    Args:
        detections: List of detection dictionaries with 'bbox' and 'confidence'
        iou_threshold: IoU threshold for merging (default: 0.5)
    
    Returns:
        Filtered list of detections (overlaps removed)
    """
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence (descending)
    sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    suppressed = set()
    
    for i, det1 in enumerate(sorted_dets):
        if i in suppressed:
            continue
        
        keep.append(det1)
        
        # Check against remaining detections
        for j in range(i + 1, len(sorted_dets)):
            if j in suppressed:
                continue
            
            det2 = sorted_dets[j]
            iou = calculate_iou(det1['bbox'], det2['bbox'])
            
            if iou > iou_threshold:
                suppressed.add(j)
    
    return keep
```

### 3.9 Algorithm 8: Thumbnail Generation (Crop Main Object)

```python
def generate_thumbnail(image, detections, cluster_labels, main_cluster_id):
    """
    Generate thumbnail by cropping one representative object from main cluster.
    
    Algorithm:
        1. Filter detections to main cluster:
           main_detections = [det for det, label in zip(detections, cluster_labels) 
                             if label == main_cluster_id]
        
        2. Select representative detection:
           Option A: First detection
           Option B: Most centered detection
           Option C: Highest confidence detection
           representative = main_detections[0]  # Simple approach
        
        3. Crop bounding box from original image:
           bbox = representative['bbox']
           crop = image[y1:y2, x1:x2]
        
        4. Resize to 200×200 maintaining aspect ratio:
           - Calculate scale to fit 200×200
           - Resize with high-quality interpolation
           - Add padding if needed to make square
        
        5. Compress to JPEG:
           jpeg_data = compress_to_jpeg(thumbnail, quality=0.8)
        
        6. Return JPEG data for Core Data storage
    
    Args:
        image: Original image
        detections: All detections
        cluster_labels: Cluster assignment for each detection
        main_cluster_id: ID of the largest cluster
    
    Returns:
        JPEG data as Data/bytes
    """
    # Filter to main cluster
    main_detections = []
    for det, label in zip(detections, cluster_labels):
        if label == main_cluster_id:
            main_detections.append(det)
    
    if len(main_detections) == 0:
        return None
    
    # Select representative (first one for simplicity)
    representative = main_detections[0]
    bbox = representative['bbox']
    
    # Crop region
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    crop = image[y1:y2, x1:x2]
    
    # Resize to 200×200 with aspect ratio preserved
    h, w = crop.shape[:2]
    scale = min(200.0 / w, 200.0 / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = resize(crop, (new_w, new_h), interpolation=INTER_AREA)
    
    # Create 200×200 canvas with padding
    thumbnail = np.ones((200, 200, 3), dtype=np.uint8) * 255
    y_offset = (200 - new_h) // 2
    x_offset = (200 - new_w) // 2
    thumbnail[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # Compress to JPEG
    jpeg_data = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
    
    return jpeg_data
```

### 3.10 Key Parameters (Must Match Python)

```python
# Detection parameters
CONF_THRESHOLD = 0.001          # YOLOX confidence threshold
NMS_THRESHOLD = 0.65           # Non-maximum suppression threshold
INPUT_SIZE = (640, 640)        # YOLOX-S input size

# Filtering parameters
SIZE_OUTLIER_STD = 1.0         # Size outlier removal (std deviations)
ASPECT_RATIO_STD = 0.5         # Aspect ratio filtering (std deviations)
AREA_CONSISTENCY_STD = 1.0     # Area consistency filtering (std deviations)

# Clustering parameters
SIMILARITY_THRESHOLD = 0.80    # Cosine similarity threshold
DISTANCE_THRESHOLD = 1.0 - SIMILARITY_THRESHOLD  # = 0.20
CLUSTERING_METRIC = 'cosine'
CLUSTERING_LINKAGE = 'average'

# Merging parameters
IOU_THRESHOLD = 0.5            # IoU threshold for overlap merging

# Thumbnail parameters
THUMBNAIL_SIZE = (200, 200)    # Output thumbnail size
JPEG_QUALITY = 80              # JPEG compression quality (0-100)
```

---

## 4. Technical Architecture

### 3.1 System Components

```
┌─────────────────────────────────────────────────────────────┐
│                         AICounter App                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  ContentView   │  │  DetectionView  │  │  SettingsView│ │
│  │   (SwiftUI)    │  │    (SwiftUI)    │  │   (SwiftUI)  │ │
│  └────────┬───────┘  └────────┬────────┘  └──────┬───────┘ │
│           │                   │                    │         │
│           └───────────────────┼────────────────────┘         │
│                               │                              │
│  ┌────────────────────────────▼────────────────────────────┐ │
│  │              AICounter (Main Controller)                │ │
│  │  • Orchestrates counting pipeline                       │ │
│  │  • Manages model lifecycle                              │ │
│  │  • Handles errors and edge cases                        │ │
│  └────────┬──────────────────────────────────────┬─────────┘ │
│           │                                       │           │
│  ┌────────▼─────────┐  ┌──────────────┐  ┌──────▼─────────┐ │
│  │ YOLOXDetector    │  │   TinyCLIP   │  │  Similarity    │ │
│  │  • Load model    │  │  Embedder    │  │  Clusterer     │ │
│  │  • Run inference │  │  • Extract   │  │  • Cosine sim  │ │
│  │  • Parse output  │  │    features  │  │  • Clustering  │ │
│  └──────────────────┘  └──────────────┘  └────────────────┘ │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            Core ML Models (.mlpackage)               │   │
│  │  • yolox_s.mlpackage (17 MB)                        │   │
│  │  • tinyclip_vision.mlpackage (16 MB)                │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Core Classes

#### CameraManager.swift
- **Responsibility**: Camera capture and session management
- **Methods**:
  - `startSession()` - Initialize AVCaptureSession
  - `capturePhoto() -> UIImage` - Capture still image
  - `stopSession()` - Clean up camera resources
  - `toggleFlash()` - Control flash mode
  - `focusAt(point:)` - Manual focus control

#### AICounter.swift
- **Responsibility**: Main orchestration layer
- **Methods**:
  - `count(image:confThreshold:similarityThreshold:) -> CountResult`
  - `filterSizeOutliers(_ detections:) -> [Detection]`
  - `cropImage(_ image:to:) -> CGImage?`

#### YOLOXDetector.swift
- **Responsibility**: Object detection
- **Methods**:
  - `detect(image:confThreshold:) -> [Detection]`
  - `parseYOLOXOutput(_ output:) -> [Detection]`

#### TinyCLIPEmbedder.swift
- **Responsibility**: Visual embedding extraction
- **Methods**:
  - `getEmbedding(for:) -> [Float]`
  - `resizeImage(_:to:) -> CGImage`
  - `createPixelBuffer(from:) -> CVPixelBuffer?`

#### SimilarityClusterer.swift
- **Responsibility**: Clustering by similarity
- **Methods**:
  - `cosineSimilarity(_:_:) -> Float`
  - `cluster(embeddings:threshold:) -> [Int]`

#### HistoryManager.swift
- **Responsibility**: Manage counting history persistence
- **Methods**:
  - `saveSession(_ result:image:)` - Save counting result with thumbnail
  - `fetchHistory(limit:) -> [CountingSession]` - Retrieve history entries
  - `deleteSession(_ id:)` - Remove specific entry
  - `clearAll()` - Delete all history
  - `generateThumbnail(from:detections:) -> Data` - Crop one object from main cluster and compress to thumbnail

### 3.3 Data Models

```swift
struct Detection {
    let bbox: CGRect
    let confidence: Float
    let classId: Int
}

struct CountResult {
    let count: Int                          // Main cluster count only
    let detections: [Detection]
    let largestClusterId: Int
    let clusterCounts: [Int: Int]           // Internal use only
}

struct ProcessingStats {
    let totalDetections: Int
    let afterFiltering: Int
    let processingTime: TimeInterval
}

// Core Data Entity
class CountingSession: NSManagedObject {
    @NSManaged var id: UUID
    @NSManaged var thumbnailData: Data      // Cropped image of main object (200×200 JPEG)
    @NSManaged var count: Int16             // Main cluster count only
    @NSManaged var timestamp: Date
    @NSManaged var confidenceThreshold: Float
    @NSManaged var similarityThreshold: Float
    @NSManaged var isFavorite: Bool
}
```

---

## 4. User Interface Design

### 4.1 Main Screen (ContentView)

```
┌─────────────────────────────────────┐
│    📋 AICounter              ⚙️     │
├─────────────────────────────────────┤
│                                     │
│  ┌───────────────────────────────┐  │
│  │                               │  │
│  │                               │  │
│  │      [Image Preview]          │  │
│  │      or placeholder           │  │
│  │                               │  │
│  │                               │  │
│  └───────────────────────────────┘  │
│                                     │
│              Count: 24              │
│                                     │
│  ┌───────────────────────────────┐  │
│  │      📸 Capture Photo         │  │
│  └───────────────────────────────┘  │
│                                     │
│  ┌───────────────────────────────┐  │
│  │      📷 Select from Library   │  │
│  └───────────────────────────────┘  │
│                                     │
│  ┌───────────────────────────────┐  │
│  │      🔍 Count Objects         │  │
│  └───────────────────────────────┘  │
│                                     │
└─────────────────────────────────────┘

Note: 📋 icon in top-left opens History view
Note: Only main cluster count displayed (simplified UI)
```

### 4.2 Camera View (Clean Viewfinder)

```
┌─────────────────────────────────────┐
│    ←                            ⚙️  │
├─────────────────────────────────────┤
│                                     │
│                                     │
│                                     │
│         [Live Camera Feed]          │
│         Clean - No Overlays         │
│                                     │
│                                     │
│                                     │
│                                     │
│                                     │
│                                     │
│  🔦                          🔄     │
│                                     │
│              ⭕                     │
│          (Shutter Button)           │
│                                     │
└─────────────────────────────────────┘

Note: No real-time bounding boxes or labels
      Inference triggered only AFTER capture
```

### 4.3 Settings Screen

```
┌─────────────────────────────────────┐
│    ←  Settings                      │
├─────────────────────────────────────┤
│                                     │
│  Detection Settings                 │
│  ─────────────────────────────────  │
│                                     │
│  Confidence Threshold        0.25   │
│  ━━━━━●─────────────────────        │
│  0.1                    0.9         │
│                                     │
│  Similarity Threshold        0.80   │
│  ━━━━━━━━━━━━━━●───────────        │
│  0.6                    0.95        │
│                                     │
│  ─────────────────────────────────  │
│                                     │
│  Filtering Options                  │
│  ─────────────────────────────────  │
│                                     │
│  Size Outlier Filtering      [ON]   │
│  Aspect Ratio Filtering      [ON]   │
│                                     │
│  ─────────────────────────────────  │
│                                     │
│  ┌───────────────────────────────┐  │
│  │      Reset to Defaults        │  │
│  └───────────────────────────────┘  │
│                                     │
└─────────────────────────────────────┘
```

### 4.4 History View

```
┌─────────────────────────────────────┐
│    ←  History                       │
├─────────────────────────────────────┤
│                                     │
│  ┌─────────────────────────────┐   │
│  │ [🍎 Crop] Count: 24         │   │
│  │            Dec 9, 2:30 PM   │   │
│  └─────────────────────────────┘   │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ [📦 Crop] Count: 18         │   │
│  │            Dec 9, 1:15 PM   │   │
│  └─────────────────────────────┘   │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ [🔧 Crop] Count: 32         │   │
│  │            Dec 8, 5:45 PM   │   │
│  └─────────────────────────────┘   │
│                                     │
│  ┌───────────────────────────────┐  │
│  │        Clear All History      │  │
│  └───────────────────────────────┘  │
│                                     │
└─────────────────────────────────────┘

Interactions:
• Tap entry → View full details
• Swipe left → Delete button
• Pull to refresh

Note: Thumbnail shows cropped image of one representative 
      object from the main cluster (not full scene)
```

### 4.5 Processing State

```
┌─────────────────────────────────────┐
│         AICounter                   │
├─────────────────────────────────────┤
│                                     │
│  ┌───────────────────────────────┐  │
│  │                               │  │
│  │      [Image]                  │  │
│  │                               │  │
│  └───────────────────────────────┘  │
│                                     │
│         🔄 Processing...            │
│                                     │
│  ┌───────────────────────────────┐  │
│  │ Stage 1: Detecting objects    │  │
│  │ Stage 2: Extracting features  │  │
│  │ Stage 3: Clustering...    ✓   │  │
│  └───────────────────────────────┘  │
│                                     │
│  ████████████░░░░░░░░░░░░ 60%      │
│                                     │
└─────────────────────────────────────┘
```

---

## 5. Performance Requirements

### 5.1 Response Time
| Operation | Target | Maximum |
|-----------|--------|---------|
| Image selection | < 100ms | 500ms |
| YOLOX inference | < 100ms | 200ms |
| TinyCLIP per crop | < 20ms | 50ms |
| Total processing | < 1.5s | 3s |
| UI update | < 16ms | 33ms |

### 5.2 Device Requirements
- **Minimum**: iPhone X (A11 Bionic), iOS 15.0
- **Recommended**: iPhone 12 (A14 Bionic), iOS 16.0
- **Optimal**: iPhone 14+ (A16 Bionic), iOS 17.0

### 5.3 Memory Usage
- **Idle**: < 50 MB
- **Processing**: < 150 MB
- **Peak**: < 200 MB
- **History storage**: < 20 MB (100 entries with thumbnails)

### 5.4 Battery Impact
- Camera preview (1 minute): < 1% battery
- Processing one image should consume < 0.1% battery
- Continuous use (10 images with camera): < 2% battery
- Note: No real-time inference minimizes battery drain during camera use

---

## 6. Quality Requirements

### 6.1 Accuracy
- **Target**: 95% accuracy compared to manual count
- **Minimum**: 90% accuracy for clear images
- **Challenging scenarios**: 85% accuracy (overlapping objects, poor lighting)

### 6.2 Reliability
- **Crash rate**: < 0.1% of sessions
- **Successful processing**: > 99% of valid images
- **Error handling**: All errors gracefully handled with user feedback

### 6.3 Compatibility
- **iOS versions**: 15.0 - 17.x
- **Devices**: iPhone X and newer
- **Image formats**: JPEG, PNG, HEIC
- **Image sizes**: 100×100 to 4K (3840×2160)

---

## 7. Security & Privacy

### 7.1 Data Privacy
- ✅ All processing happens on-device
- ✅ No data sent to external servers
- ✅ No user tracking or analytics
- ✅ Photos not saved without explicit user action
- ✅ No access to photo metadata (location, timestamps)

### 7.2 Permissions
| Permission | Required | Purpose |
|------------|----------|---------|
| Camera | Yes | Capture photos for counting |
| Photo Library | Yes | Select existing images to process |
| File Storage | Optional | Export results (future) |

### 7.3 Data Storage
- **Core Data** for counting history (MVP)
  - Entity: `CountingSession`
  - Attributes: id, thumbnailData, count (main cluster only), timestamp, confidenceThreshold, similarityThreshold, isFavorite
  - Max 100 entries (auto-cleanup oldest)
  - Thumbnail: Cropped image of one representative object from main cluster (200×200 px compressed JPEG)
  - No full-resolution images or full scenes stored (privacy + storage)
  - Simplified schema: No cluster count stored (UI shows main count only)
- **UserDefaults** for app settings
  - Confidence threshold
  - Similarity threshold
  - Filtering preferences
- **Total storage estimate**: < 20 MB for 100 history entries

---

## 8. Testing Strategy

### 8.1 Unit Tests
- ✅ YOLOXDetector output parsing
- ✅ TinyCLIP embedding extraction
- ✅ Cosine similarity calculation
- ✅ Clustering algorithm correctness
- ✅ Outlier filtering logic
- ✅ Core Data CRUD operations
- ✅ Thumbnail generation and compression
- ✅ History entry limit enforcement (max 100)

### 8.2 Integration Tests
- ✅ Full counting pipeline end-to-end
- ✅ Model loading and inference
- ✅ Image preprocessing
- ✅ Error handling paths

### 8.3 UI Tests
- ✅ Camera capture flow
- ✅ Image selection flow
- ✅ Count button interaction
- ✅ Result display
- ✅ History view navigation
- ✅ History entry deletion
- ✅ Settings adjustments

### 8.4 Performance Tests
- ✅ Processing time under target (< 2s)
- ✅ Memory usage within limits (< 200 MB)
- ✅ Frame rate during processing (UI responsive)

### 8.5 Accuracy Tests
| Test Scenario | Expected Count | Accuracy Target |
|--------------|----------------|-----------------|
| 10 identical bottles | 10 | 100% |
| 25 similar boxes | 25 | 96% (24-26) |
| 50 mixed objects | Largest group | 90% |
| Overlapping items | Best effort | 85% |

---

## 9. Launch Criteria

### 9.1 MVP Launch Requirements
- ✅ All essential features implemented (Features 1-4)
  - Dual input support (camera + photo library)
  - AI-powered object counting
  - Result display
  - Counting history with persistent storage
- ✅ Core counting accuracy ≥ 90%
- ✅ Processing time < 2s on iPhone 12+
- ✅ Clean camera viewfinder (no real-time overlays)
- ✅ "Capture first, count second" workflow implemented
- ✅ History persistence working correctly (Core Data)
- ✅ Thumbnail generation and storage optimized
- ✅ History limit enforcement (max 100 entries)
- ✅ Zero critical bugs
- ✅ Crash rate < 0.5%
- ✅ Successful TestFlight beta with 20+ users
- ✅ App Store review guidelines compliance
- ✅ Camera and photo library permissions handled properly

### 9.2 Success Metrics (3 months post-launch)
- **Adoption**: 1,000+ downloads
- **Retention**: 40% D7 retention
- **Engagement**: 3+ counting sessions per active user per week
- **Rating**: 4.5+ stars on App Store
- **Accuracy**: User-reported accuracy > 90%

---

## 10. Development Roadmap

### Phase 1: MVP (8-10 weeks)
**Week 1-2: Foundation**
- ✅ Xcode project setup
- ✅ Core ML models integration
- ✅ Basic UI scaffolding
- ✅ Core Data stack setup
- ✅ Camera permissions and setup

**Week 3-4: Camera & Input System**
- ✅ AVCaptureSession implementation (clean viewfinder)
- ✅ PHPickerViewController integration
- ✅ Camera controls (shutter, flash, focus)
- ✅ Image capture and preview flow
- ✅ "Capture first, count second" workflow

**Week 5-6: Core Counting Features**
- ✅ YOLOXDetector implementation
- ✅ TinyCLIPEmbedder implementation
- ✅ SimilarityClusterer implementation
- ✅ AICounter orchestration

**Week 7-8: History & Data Persistence**
- ✅ Core Data entity modeling
- ✅ HistoryManager implementation
- ✅ Thumbnail generation: Crop representative object from main cluster
- ✅ Image compression and resizing (200×200 JPEG)
- ✅ History view UI
- ✅ CRUD operations (save, fetch, delete)
- ✅ Auto-cleanup (max 100 entries)

**Week 9-10: Polish & Testing**
- ✅ UI refinement
- ✅ Error handling
- ✅ Unit/integration tests (including Core Data)
- ✅ Performance optimization
- ✅ Migration testing
- ✅ TestFlight beta

### Phase 2: Enhanced Features (4 weeks)
- Visual detection overlay (Feature 5)
- Adjustable thresholds (Feature 6)
- Settings persistence
- Favorites functionality (Feature 8)
- Performance profiling

### Phase 3: Pro Features (6 weeks)
- Batch processing (Feature 7)
- Export functionality (CSV, PDF)
- Advanced filtering and search in history
- iPad support with optimized layouts
- Dark mode optimization

---

## 11. Dependencies & Risks

### 11.1 Dependencies
| Dependency | Type | Risk Level |
|------------|------|------------|
| Core ML | Platform | Low |
| Vision framework | Platform | Low |
| Accelerate | Platform | Low |
| YOLOX model | External | Medium |
| TinyCLIP model | External | Medium |

### 11.2 Risks & Mitigations

**Risk 1: Model accuracy varies across scenarios**
- **Impact**: High
- **Likelihood**: Medium
- **Mitigation**: Extensive testing, adjustable thresholds, user feedback loop

**Risk 2: Performance on older devices**
- **Impact**: Medium
- **Likelihood**: High
- **Mitigation**: Minimum device requirement (iPhone X), optimization, reduced input size for older devices

**Risk 3: App Store rejection**
- **Impact**: High
- **Likelihood**: Low
- **Mitigation**: Follow guidelines, privacy policy, no misleading claims

**Risk 4: Large app size (models = 33 MB)**
- **Impact**: Low
- **Likelihood**: High
- **Mitigation**: On-demand resources, model quantization (future)

---

## 12. Open Questions

1. Should we support iPad with larger canvas for annotation?
2. Should we add AR mode for real-time counting?
3. Should we offer cloud sync for history across devices?
4. Should we monetize via one-time purchase, subscription, or ads?
5. Should we add social sharing features?
6. Should we support video input (count objects in video frames)?

---

## 13. Appendix

### 13.1 Glossary
- **YOLOX**: You Only Look Once X - object detection model
- **TinyCLIP**: Compact vision-language model for embeddings
- **Core ML**: Apple's machine learning framework
- **Neural Engine**: Dedicated ML accelerator in Apple chips
- **Embedding**: Vector representation of visual features
- **Cosine Similarity**: Measure of similarity between vectors
- **Agglomerative Clustering**: Bottom-up clustering algorithm

### 13.2 References
- YOLOX paper: https://arxiv.org/abs/2107.08430
- TinyCLIP: https://github.com/wkcn/TinyCLIP
- Core ML documentation: https://developer.apple.com/documentation/coreml
- Vision framework: https://developer.apple.com/documentation/vision

### 13.3 Contact
- **Product Owner**: TBD
- **Tech Lead**: TBD
- **Design Lead**: TBD

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-09 | Product Team | Initial PRD based on Swift migration guide |

---

**Approval**

- [ ] Product Owner
- [ ] Engineering Lead
- [ ] Design Lead
- [ ] QA Lead

**Status**: Draft - Ready for Review
