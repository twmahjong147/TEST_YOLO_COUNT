#!/usr/bin/env python3
"""
Object Counting with YOLOX-S and TinyCLIP Similarity Clustering
================================================================

A production-ready object counting program using visual similarity clustering.

Processing Pipeline:
    Input Image → Object Detection → Outlier Filtering → Similarity Clustering → Result Generation
    
    Stage 1: Detect all potential objects (class-agnostic YOLOX)
    Stage 1.5: Remove size outliers (boxes too small/large compared to median)
    Stage 1.6: Remove aspect ratio outliers (unusual shapes)
    Stage 2: Filter by area consistency and extract visual embeddings
    Stage 3: Cluster by similarity and merge overlapping detections
    Stage 4: Count largest cluster as final result

Usage:
    # Basic usage
    python3 count_objects_yolox.py --image sample1.JPG --output output/
    
    # Custom thresholds
    python3 count_objects_yolox.py --image sample1.JPG --similarity-threshold 0.85 --iou-threshold 0.6

Features:
    - Offline YOLOX-S model (no internet required)
    - TinyCLIP visual embeddings (wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M)
    - Similarity-based clustering (no vocabulary needed)
    - Area consistency filtering
    - IoU-based overlap merging
    - Rich console output with statistics
    - Annotated visualization images
    - Detection analysis and reports

Author: YOLOX Object Counting System
Date: December 9, 2025
"""

import torch
import cv2
import sys
import os
import numpy as np
from collections import Counter
import time
import argparse
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import AgglomerativeClustering

# Add official YOLOX to path
sys.path.insert(0, 'official_yolox')

from yolox.exp import get_exp
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess




class TinyCLIPEmbedder:
    """Visual embedding extractor using TinyCLIP."""
    
    def __init__(self, model_name='wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M'):
        """Initialize TinyCLIP model."""
        print(f"\n📦 Loading TinyCLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        print(f"✓ TinyCLIP model loaded successfully")
        print(f"  Parameters: 8M (Vision) + 3M (Text)")
        print(f"  Training dataset: YFCC15M")
    
    def get_visual_embedding(self, crop_bgr):
        """
        Extract visual embedding from crop using CLIP vision encoder.
        
        Args:
            crop_bgr: Cropped image in BGR format (numpy array)
        
        Returns:
            Normalized embedding vector (numpy array)
        """
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(crop_rgb)
        
        inputs = self.processor(images=pil_image, return_tensors="pt")
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()[0]


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU value (0-1)
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def filter_contained_across(detections, iou_threshold=0.5):
    """
    Merge detections that are geometrically too close (high IoU).
    Keep the detection with higher confidence.
    
    Args:
        detections: List of detection dictionaries with 'bbox' and 'confidence'
        iou_threshold: IoU threshold for merging (default: 0.5)
    
    Returns:
        List of filtered detections
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


def filter_by_area_consistency(crops_list, area_variance_threshold=2.0):
    """
    Filter crops by area consistency to avoid grouping very different sizes.
    Removes crops whose area is too different from the median.
    
    Args:
        crops_list: List of crop dictionaries with 'area' field
        area_variance_threshold: Standard deviations from median to keep (default: 2.0)
    
    Returns:
        Filtered list of crops
    """
    if len(crops_list) <= 2:
        return crops_list
    
    areas = np.array([c['area'] for c in crops_list])
    median_area = np.median(areas)
    std_area = np.std(areas)
    
    if std_area == 0:
        return crops_list
    
    min_area = median_area - area_variance_threshold * std_area
    max_area = median_area + area_variance_threshold * std_area
    
    filtered = [c for c in crops_list if min_area <= c['area'] <= max_area]
    
    return filtered


def cluster_by_similarity(embeddings, similarity_threshold=0.80):
    """
    Cluster embeddings based on cosine similarity.
    
    Args:
        embeddings: numpy array of shape (n_crops, embedding_dim)
        similarity_threshold: minimum similarity to be in same cluster (0-1)
    
    Returns:
        cluster_labels: array of cluster assignments
        cluster_counts: dict mapping cluster_id to count
    """
    distance_threshold = 1 - similarity_threshold
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='cosine',
        linkage='average'
    )
    
    cluster_labels = clustering.fit_predict(embeddings)
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    
    return cluster_labels, dict(zip(unique_labels, counts))





def remove_size_outliers(boxes, std_threshold=3.0):
    """
    Remove boxes that are size outliers (too small or too large compared to median).

    Args:
        boxes: Tensor or array of shape (N, 4) with bounding boxes [x1, y1, x2, y2]
        std_threshold: Number of standard deviations from median to consider outlier

    Returns:
        List of indices to keep
    """
    if len(boxes) < 3:  # Need enough boxes to calculate statistics
        return list(range(len(boxes)))

    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes)

    # Calculate box areas
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    areas_np = areas.numpy()

    median_area = np.median(areas_np)
    std_area = np.std(areas_np)

    # Keep boxes within threshold standard deviations of median
    min_area = median_area - std_threshold * std_area
    max_area = median_area + std_threshold * std_area

    keep = []
    for i, area in enumerate(areas_np):
        if min_area <= area <= max_area:
            keep.append(i)

    return keep


def remove_aspect_ratio_outliers(boxes, min_ratio: float = 0.45, max_ratio: float = 2.2) -> list:
    """
    Remove boxes with unusual aspect ratios.

    Most objects like blueberries are roughly square. This filter removes
    boxes that are too elongated (likely false positives from shadows/gaps).

    Args:
        boxes: Tensor or array of shape (N, 4) with bounding boxes [x1, y1, x2, y2]
        min_ratio: Minimum width/height ratio
        max_ratio: Maximum width/height ratio

    Returns:
        List of indices to keep
    """
    if len(boxes) == 0:
        return []

    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes)

    keep = []
    for i, box in enumerate(boxes):
        width = box[2] - box[0]
        height = box[3] - box[1]
        if height > 0:
            ratio = width / height
            if min_ratio <= ratio <= max_ratio:
                keep.append(i)

    return keep


def visualize_detections_custom(img, detections, main_class=None):
    """
    Visualize detections with custom labels and highlight main object.
    
    Args:
        img: Input image
        detections: List of detection dictionaries
        main_class: The main object class to highlight
    
    Returns:
        Annotated image
    """
    vis_img = img.copy()

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cls_name = det['class']
        confidence = det['confidence']

        # Color: highlight main object in green, others in blue
        if main_class and cls_name == main_class:
            color = (0, 255, 0)  # Green for main object
            thickness = 3
        else:
            color = (255, 0, 0)  # Blue for others
            thickness = 2

        # Draw bounding box
        cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        # Draw label with background
        label = f"{cls_name}: {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1

        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        # Draw background rectangle
        cv2.rectangle(vis_img, 
                     (int(x1), int(y1) - text_height - 10),
                     (int(x1) + text_width + 10, int(y1)),
                     color, -1)

        # Draw text
        cv2.putText(vis_img, label,
                   (int(x1) + 5, int(y1) - 5),
                   font, font_scale, (255, 255, 255), font_thickness)

    return vis_img


def count_objects_with_yolox(img_path, weight_path, output_dir='output', outlier_threshold=3.0, similarity_threshold=0.80, aspect_ratio_min=0.45, aspect_ratio_max=2.2, area_variance_threshold=2.0, iou_threshold=0.5, **kwargs):
    """
    Count objects in an image using YOLOX-S detection and TinyCLIP similarity clustering.
    
    Processing Pipeline:
        Stage 1: Detect all potential objects (class-agnostic YOLOX)
        Stage 1.5: Remove size outliers (too small/large compared to median)
        Stage 1.6: Remove aspect ratio outliers (unusual shapes)
        Stage 2: Filter by area consistency and extract visual embeddings
        Stage 3: Cluster by similarity and merge overlapping detections
        Stage 4: Count largest cluster and generate results
    
    Args:
        img_path: Path to the input image
        weight_path: Path to YOLOX-S model weights
        output_dir: Directory to save visualization
        outlier_threshold: Standard deviations from median for outlier removal (0 to disable)
        similarity_threshold: Cosine similarity threshold for clustering (0-1)
        aspect_ratio_min: Minimum width/height ratio for aspect ratio filtering
        aspect_ratio_max: Maximum width/height ratio for aspect ratio filtering
        area_variance_threshold: Standard deviations from median for area consistency filtering
        iou_threshold: IoU threshold for merging overlapping detections (0-1)
    
    Returns:
        Dictionary containing counting results and statistics
    """
    model_name = 'yolox_s'
    input_size = (640, 640)

    print(f"\n{'='*70}")
    print(f"YOLOX-S + TinyCLIP SIMILARITY CLUSTERING")
    print(f"{'='*70}")
    print(f"\n🔬 Processing Pipeline:")
    print(f"  Stage 1: Object Detection (YOLOX)")
    if outlier_threshold > 0:
        print(f"  Stage 1.5: Size Outlier Filtering ({outlier_threshold} std)")
        print(f"  Stage 1.6: Aspect Ratio Filtering ({aspect_ratio_min}-{aspect_ratio_max})")
    print(f"  Stage 2: Area Consistency + Visual Embeddings")
    print(f"  Stage 3: Similarity Clustering (threshold={similarity_threshold})")
    print(f"  Stage 4: IoU Merging + Result Generation")

    # Check if weight file exists
    if not os.path.exists(weight_path):
        print(f"❌ Error: Weight file not found: {weight_path}")
        print(f"\nPlease download YOLOX-S weights:")
        print(f"  wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth")
        print(f"  mv yolox_s.pth weights/")
        return None

    # Initialize TinyCLIP embedder
    embedder = TinyCLIPEmbedder()

    # Load YOLOX model
    print(f"\n{'='*70}")
    print(f"STAGE 1: OBJECT DETECTION")
    print(f"{'='*70}")
    print(f"\n📦 Loading YOLOX-S model...")
    exp = get_exp(f'official_yolox/exps/default/{model_name}.py', None)
    model = exp.get_model()
    ckpt = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Override thresholds if provided
    conf_thresh = kwargs.get('conf_threshold', exp.test_conf)
    nms_thresh = kwargs.get('nms_threshold', exp.nmsthre)

    print(f"✓ Model loaded successfully")
    print(f"  Model: {model_name.upper()}")
    print(f"  Input size: {input_size}")
    print(f"  Confidence threshold: {conf_thresh}")
    print(f"  NMS threshold: {nms_thresh}")

    # Load image
    print(f"\n📷 Loading image: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Failed to load image: {img_path}")
        return None

    h, w = img.shape[:2]
    print(f"  Image size: {w}×{h} pixels")
    print(f"  Image shape: {img.shape}")

    # Preprocess
    print(f"\n🔄 Preprocessing image...")
    preproc = ValTransform(legacy=False)
    ratio = min(input_size[0] / h, input_size[1] / w)
    img_processed, _ = preproc(img, None, input_size)
    img_tensor = torch.from_numpy(img_processed).unsqueeze(0).float()

    # Inference
    print(f"\n🚀 Running inference...")
    start_time = time.time()
    with torch.no_grad():
        outputs = model(img_tensor)
        outputs = postprocess(outputs, 80, conf_thresh, nms_thresh, class_agnostic=True)
    inference_time = time.time() - start_time

    print(f"⏱️  Inference time: {inference_time*1000:.2f}ms")

    # Process results
    if outputs[0] is not None:
        output = outputs[0].cpu()
        output[:, :4] /= ratio
        n_initial = len(output)

        print(f"\n✅ Detected {n_initial} objects!")

        # Remove size outliers (if enabled)
        if outlier_threshold > 0:
            print(f"\n🔍 Filtering size outliers (threshold: {outlier_threshold} std)...")
            boxes = output[:, :4]
            keep_indices = remove_size_outliers(boxes, std_threshold=outlier_threshold)
            output = output[keep_indices]
            n = len(output)
            n_removed = n_initial - n

            if n_removed > 0:
                print(f"  Removed {n_removed} outliers ({n_removed/n_initial*100:.1f}%)")
                print(f"  Kept {n} boxes for aspect ratio filtering")
            else:
                print(f"  No outliers detected, kept all {n} boxes")
        else:
            n = n_initial
            print(f"\n⚠️  Outlier filtering disabled")
        
        # Stage 1.6: Remove aspect ratio outliers
        print(f"\n🔍 Filtering aspect ratio outliers (ratio range: {aspect_ratio_min:.2f} - {aspect_ratio_max:.2f})...")
        n_before_aspect = n
        boxes = output[:, :4]
        keep_indices_aspect = remove_aspect_ratio_outliers(boxes, min_ratio=aspect_ratio_min, max_ratio=aspect_ratio_max)
        output = output[keep_indices_aspect]
        n = len(output)
        n_removed_aspect = n_before_aspect - n
        
        if n_removed_aspect > 0:
            print(f"  Removed {n_removed_aspect} aspect ratio outliers ({n_removed_aspect/n_before_aspect*100:.1f}%)")
            print(f"  Kept {n} boxes for classification")
        else:
            print(f"  No aspect ratio outliers detected, kept all {n} boxes")

        # Stage 2: Similarity Clustering
        print(f"\n{'='*70}")
        print(f"STAGE 2: SIMILARITY CLUSTERING")
        print(f"{'='*70}")

        all_detections = []
        crops_list = []

        # Extract all crops first
        for i in range(n):
            x1, y1, x2, y2 = output[i, :4].numpy()
            obj_conf = output[i, 4].item()

            x1_int, y1_int = max(0, int(x1)), max(0, int(y1))
            x2_int, y2_int = min(w, int(x2)), min(h, int(y2))
            crop = img[y1_int:y2_int, x1_int:x2_int]

            if crop.size == 0:
                continue

            area = (x2 - x1) * (y2 - y1)
            crops_list.append({
                'crop': crop,
                'bbox': [x1, y1, x2, y2],
                'obj_conf': obj_conf,
                'area': area,
                'index': i
            })

        # Similarity-based clustering
        print(f"\n🔍 Extracting embeddings for {len(crops_list)} crops...")
        
        # Apply area-based filtering before clustering
        n_before_area_filter = len(crops_list)
        crops_list = filter_by_area_consistency(crops_list, area_variance_threshold=area_variance_threshold)
        n_after_area_filter = len(crops_list)
        
        if n_before_area_filter > n_after_area_filter:
            print(f"  Removed {n_before_area_filter - n_after_area_filter} crops with inconsistent areas")
            print(f"  Kept {n_after_area_filter} crops for embedding extraction")
        
        embeddings = []
        for item in crops_list:
            embedding = embedder.get_visual_embedding(item['crop'])
            embeddings.append(embedding)

        embeddings = np.array(embeddings)

        print(f"🔗 Clustering with similarity threshold: {similarity_threshold}")
        cluster_labels, cluster_counts = cluster_by_similarity(embeddings, similarity_threshold)

        # Find largest cluster
        largest_cluster_id = max(cluster_counts, key=cluster_counts.get)
        main_count = cluster_counts[largest_cluster_id]

        print(f"\n📊 Clustering Results:")
        print(f"  Total clusters: {len(cluster_counts)}")
        print(f"  Largest cluster: {main_count} objects (cluster #{largest_cluster_id})")
        sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"  Distribution: {dict(sorted_clusters[:5])}{'...' if len(sorted_clusters) > 5 else ''}")

        # Build detections with cluster info
        for idx, item in enumerate(crops_list):
            cluster_id = cluster_labels[idx]
            is_main = (cluster_id == largest_cluster_id)

            # DEBUG: Save crop with cluster info
            debug_crop = item['crop'].copy()
            text = f"Cluster {cluster_id}" + (" (MAIN)" if is_main else "")
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

            text_height = text_size[1] + 10
            debug_crop_with_text = np.ones((debug_crop.shape[0] + text_height, debug_crop.shape[1], 3), dtype=np.uint8) * 255
            debug_crop_with_text[:debug_crop.shape[0], :] = debug_crop

            text_x = (debug_crop.shape[1] - text_size[0]) // 2
            text_y = debug_crop.shape[0] + text_size[1] + 5
            color = (0, 128, 0) if is_main else (0, 0, 0)
            cv2.putText(debug_crop_with_text, text, (text_x, text_y), font, font_scale, color, thickness)

            os.makedirs('output/debug_crops', exist_ok=True)
            cv2.imwrite(
                f'output/debug_crops/{cluster_id}_crop_{item["index"]:03d}.jpg', debug_crop_with_text,
            )

            all_detections.append({
                'class': f'cluster_{cluster_id}',
                'confidence': 1.0 if is_main else 0.5,
                'bbox': item['bbox'],
                'obj_conf': item['obj_conf'],
                'cls_conf': 1.0,
                'area': item['area'],
                'cluster_id': int(cluster_id),
                'is_main_cluster': is_main
            })
        
        # Apply IoU-based filtering to merge overlapping detections
        n_before_iou = len(all_detections)
        all_detections = filter_contained_across(all_detections, iou_threshold=iou_threshold)
        n_after_iou = len(all_detections)
        
        if n_before_iou > n_after_iou:
            print(f"\n🔗 IoU filtering: Merged {n_before_iou - n_after_iou} overlapping detections")
            print(f"  Kept {n_after_iou} unique detections")
            
            # Recalculate cluster counts after IoU filtering
            cluster_counts_after_iou = Counter([d['cluster_id'] for d in all_detections])
            largest_cluster_id = max(cluster_counts_after_iou, key=cluster_counts_after_iou.get)
            main_count = cluster_counts_after_iou[largest_cluster_id]
            cluster_counts = dict(cluster_counts_after_iou)

        # Count by class
        class_counts = Counter([d['class'] for d in all_detections])

        # Stage 3: Count Result
        print(f"\n{'='*70}")
        print(f"STAGE 3: COUNT RESULT")
        print(f"{'='*70}")
        main_class = f'cluster_{largest_cluster_id}'
        main_score = main_count
        # Convert cluster_counts keys to strings for consistency
        cluster_counts_str = {f'cluster_{cid}': cnt for cid, cnt in cluster_counts.items()}
        main_details = {
            'count': main_count,
            'total_area': sum([d['area'] for d in all_detections if d['cluster_id'] == largest_cluster_id]),
            'avg_area': np.mean([d['area'] for d in all_detections if d['cluster_id'] == largest_cluster_id]),
            'all_scores': cluster_counts_str,
            'all_stats': {f'cluster_{cid}': {
                'count': cnt,
                'total_area': sum([d['area'] for d in all_detections if d['cluster_id'] == cid])
            } for cid, cnt in cluster_counts.items()}
        }
        print(f"\n✅ Final Count: {main_count} objects")
        print(f"  (Objects in largest similarity cluster)")

        # Print counting summary
        print(f"\n{'='*70}")
        print("📊 OBJECT COUNTING RESULTS")
        print(f"{'='*70}")
        print(f"\nTotal objects detected: {len(all_detections)}")
        print(f"\nObject breakdown by class:")
        print("-" * 70)

        for cls_name, count in class_counts.most_common():
            avg_conf = np.mean([d['confidence'] for d in all_detections if d['class'] == cls_name])
            total_area = sum([d['area'] for d in all_detections if d['class'] == cls_name])
            percentage = (count / len(all_detections)) * 100
            marker = "⭐" if cls_name == main_class else "  "
            print(f" {marker} {cls_name:20s}: {count:3d} objects ({percentage:5.1f}%) - avg conf: {avg_conf:.3f} - area: {total_area:.0f}px²")

        print("-" * 70)
        print(f"\nUnique object classes: {len(class_counts)}")

        # Show top 10 detections
        print(f"\n🎯 Top 10 Detections (by confidence):")
        print("-" * 70)
        sorted_dets = sorted(all_detections, key=lambda x: x['confidence'], reverse=True)
        for i, det in enumerate(sorted_dets[:10], 1):
            bbox_str = f"[{det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}, {det['bbox'][2]:.0f}, {det['bbox'][3]:.0f}]"
            area_str = f"area: {det['area']:.0f}px²"
            print(f"   {i:2d}. {det['class']:15s} - conf: {det['confidence']:.3f} - {area_str} - bbox: {bbox_str}")
        print("-" * 70)

        # Show statistics
        print(f"\n📈 Statistics:")
        all_confidences = [d['confidence'] for d in all_detections]
        print(f"   Average confidence: {np.mean(all_confidences):.3f}")
        print(f"   Median confidence:  {np.median(all_confidences):.3f}")
        print(f"   Min confidence:     {np.min(all_confidences):.3f}")
        print(f"   Max confidence:     {np.max(all_confidences):.3f}")

        # Stage 4: Result Generation
        print(f"\n{'='*70}")
        print(f"STAGE 4: RESULT GENERATION")
        print(f"{'='*70}")
        print(f"\n🎨 Generating visualization...")

        # Generate visualization with cluster highlighting
        vis_img = visualize_detections_custom(img, all_detections, main_class)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save visualization
        img_basename = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(output_dir, f"{img_basename}_similarity_counting.jpg")
        cv2.imwrite(output_path, vis_img)
        print(f"💾 Saved visualization to: {output_path}")

        # Create text report
        report_path = os.path.join(output_dir, f"{img_basename}_counting_report.txt")
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("YOLOX-S + TinyCLIP SIMILARITY CLUSTERING REPORT\n")
            f.write("="*70 + "\n\n")
            f.write("PROCESSING PIPELINE:\n")
            f.write("  Stage 1: Object Detection (YOLOX)\n")
            f.write("  Stage 1.5: Size Outlier Filtering\n")
            f.write("  Stage 1.6: Aspect Ratio Filtering\n")
            f.write("  Stage 2: Area Consistency + Visual Embeddings\n")
            f.write("  Stage 3: Similarity Clustering + IoU Merging\n")
            f.write("  Stage 4: Result Generation\n\n")
            f.write(f"Image: {img_path}\n")
            f.write(f"Image size: {w}×{h} pixels\n")
            f.write(f"Model: YOLOX-S\n")
            f.write(f"Method: SIMILARITY CLUSTERING\n")
            f.write(f"Similarity Threshold: {similarity_threshold}\n")
            f.write(f"IoU Threshold: {iou_threshold}\n")
            f.write(f"Inference time: {inference_time*1000:.2f}ms\n\n")

            # Main Cluster section
            if main_class:
                f.write("="*70 + "\n")
                f.write("LARGEST CLUSTER (Main Count):\n")
                f.write("="*70 + "\n")
                f.write(f"Cluster: {main_class.upper()}\n")
                f.write(f"Count: {main_details['count']}\n")
                f.write(f"Total Area: {main_details['total_area']:.2f} px²\n")
                f.write(f"Average Area: {main_details['avg_area']:.2f} px²\n\n")
                f.write("All Clusters:\n")
                for cls, score in sorted(main_details['all_scores'].items(), key=lambda x: x[1], reverse=True):
                    stats = main_details['all_stats'][cls]
                    f.write(f"  {cls:20s}: Count={score} (Area={stats['total_area']:.2f})\n")
                f.write("\n")

            f.write(f"Total objects detected: {len(all_detections)}\n")
            f.write(f"Unique classes: {len(class_counts)}\n\n")
            f.write("Object breakdown by class:\n")
            f.write("-" * 70 + "\n")
            for cls_name, count in class_counts.most_common():
                avg_conf = np.mean([d['confidence'] for d in all_detections if d['class'] == cls_name])
                total_area = sum([d['area'] for d in all_detections if d['class'] == cls_name])
                percentage = (count / len(all_detections)) * 100
                marker = "⭐" if cls_name == main_class else "  "
                f.write(f"{marker} {cls_name:20s}: {count:3d} objects ({percentage:5.1f}%) - avg conf: {avg_conf:.3f} - area: {total_area:.0f}px²\n")
            f.write("-" * 70 + "\n\n")
            f.write("Top 10 Detections:\n")
            f.write("-" * 70 + "\n")
            for i, det in enumerate(sorted_dets[:10], 1):
                area_str = f"area: {det['area']:.0f}px²"
                f.write(f"   {i:2d}. {det['class']:15s} - conf: {det['confidence']:.3f} - {area_str}\n")
            f.write("-" * 70 + "\n\n")
            f.write("Statistics:\n")
            f.write(f"   Average confidence: {np.mean(all_confidences):.3f}\n")
            f.write(f"   Median confidence:  {np.median(all_confidences):.3f}\n")
            f.write(f"   Min confidence:     {np.min(all_confidences):.3f}\n")
            f.write(f"   Max confidence:     {np.max(all_confidences):.3f}\n")

        print(f"📄 Saved text report to: {report_path}")

        # Print summary
        print(f"\n{'='*70}")
        print("✅ COUNTING COMPLETE!")
        print(f"{'='*70}")

        if main_class:
            print(f"\n⭐ Main Cluster: {main_class.upper()}")
            print(f"   Count: {main_details['count']}")
            print(f"   Total Area: {main_details['total_area']:.2f} px²")

        most_common = class_counts.most_common(1)[0]
        print(f"\nLargest cluster: {most_common[0]} ({most_common[1]} objects)")
        print(f"Output visualization: {output_path}")
        print(f"Output report: {report_path}")
        print(f"{'='*70}")

        # Return results
        return {
            'model': model_name,
            'method': 'similarity',
            'embedder': 'TinyCLIP',
            'total_count': len(all_detections),
            'clusters': class_counts,
            'detections': all_detections,
            'inference_time': inference_time,
            'largest_cluster': most_common,
            'main_cluster': {
                'class': main_class,
                'count': main_score,
                'details': main_details
            } if main_class else None,
            'visualization_path': output_path,
            'report_path': report_path,
            'image_size': (w, h)
        }
    else:
        print("\n❌ No objects detected")
        print(f"{'='*70}")
        return {
            'model': model_name,
            'method': 'similarity',
            'embedder': 'TinyCLIP',
            'total_count': 0,
            'clusters': {},
            'detections': [],
            'inference_time': inference_time,
            'largest_cluster': None,
            'main_cluster': None,
            'visualization_path': None,
            'report_path': None,
            'image_size': (w, h)
        }


def main():
    """Main function to parse arguments and run object counting."""

    parser = argparse.ArgumentParser(
        description="Object Counting with YOLOX-S and TinyCLIP Similarity Clustering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Processing Pipeline:
  Stage 1: Object Detection (YOLOX - class-agnostic)
  Stage 1.5: Size Outlier Filtering (removes boxes too small/large)
  Stage 1.6: Aspect Ratio Filtering (removes unusual shapes)
  Stage 2: Area Consistency + Visual Embeddings (TinyCLIP)
  Stage 3: Similarity Clustering + IoU Merging
  Stage 4: Count Largest Cluster + Result Generation

Examples:
  # Basic usage
  python3 count_objects_yolox.py --image sample1.JPG --output output/
  
  # Custom similarity threshold (stricter grouping)
  python3 count_objects_yolox.py --image sample1.JPG --similarity-threshold 0.85
  
  # Custom IoU threshold for merging overlaps
  python3 count_objects_yolox.py --image sample1.JPG --iou-threshold 0.6
  
  # Adjust outlier filtering
  python3 count_objects_yolox.py --image sample1.JPG --outlier-threshold 2.0
  
  # Custom area variance threshold
  python3 count_objects_yolox.py --image sample1.JPG --area-variance-threshold 1.5
  
Requirements:
  - YOLOX-S weights must be downloaded first
  - Download from: https://github.com/Megvii-BaseDetection/YOLOX/releases/
  - TinyCLIP will be auto-downloaded from HuggingFace on first use
        """
    )

    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image (e.g., sample1.JPG)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Output directory for results (default: output/)'
    )

    parser.add_argument(
        '--weights',
        type=str,
        default='weights/yolox_s.pth',
        help='Path to YOLOX-S weights file (default: weights/yolox_s.pth)'
    )

    parser.add_argument(
        '--similarity-threshold',
        type=float,
        default=0.80,
        help='Similarity threshold for clustering method (0-1, default: 0.80, higher = stricter grouping)'
    )

    parser.add_argument(
        '--outlier-threshold',
        type=float,
        default=3.0,
        help='Standard deviations from median for size outlier removal (default: 3.0, use 0 to disable)'
    )

    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.001,
        help='Confidence threshold for object detection (default: 0.01, higher = fewer detections)'
    )

    parser.add_argument(
        '--nms-threshold',
        type=float,
        default=None,
        help='NMS (Non-Maximum Suppression) threshold (default: 0.65, higher = more overlapping boxes kept)'
    )

    parser.add_argument(
        '--aspect-ratio-min',
        type=float,
        default=0.45,
        help='Minimum width/height ratio for aspect ratio filtering (default: 0.45)'
    )

    parser.add_argument(
        '--aspect-ratio-max',
        type=float,
        default=2.2,
        help='Maximum width/height ratio for aspect ratio filtering (default: 2.2)'
    )

    parser.add_argument(
        '--area-variance-threshold',
        type=float,
        default=2.0,
        help='Area variance threshold for crop size consistency filtering (default: 2.0, standard deviations from median)'
    )

    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.5,
        help='IoU threshold for merging overlapping detections (default: 0.5, higher = more aggressive merging)'
    )

    args = parser.parse_args()

    # Validate image path
    if not os.path.exists(args.image):
        print(f"❌ Error: Image file not found: {args.image}")
        return 1

    # Remove trailing slash from output directory
    output_dir = args.output.rstrip('/')

    # Run counting
    try:
        kwargs = {}
        if args.conf_threshold is not None:
            kwargs['conf_threshold'] = args.conf_threshold
        if args.nms_threshold is not None:
            kwargs['nms_threshold'] = args.nms_threshold

        result = count_objects_with_yolox(
            args.image, 
            args.weights,
            output_dir,
            args.outlier_threshold,
            similarity_threshold=args.similarity_threshold,
            aspect_ratio_min=args.aspect_ratio_min,
            aspect_ratio_max=args.aspect_ratio_max,
            area_variance_threshold=args.area_variance_threshold,
            iou_threshold=args.iou_threshold,
            **kwargs
        )

        if result is None:
            return 1

        return 0

    except Exception as e:
        print(f"\n❌ Error during counting: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
