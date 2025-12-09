#!/usr/bin/env python3
"""
Object Counting with YOLOX-S and TinyCLIP Classification
=========================================================

A production-ready object counting program with zero-shot classification.

Processing Pipeline:
    Input Image → Object Detection → Outlier Filtering → Classification → Main Object Selection → Result Generation
    
    Stage 1: Detect all potential objects (class-agnostic)
    Stage 1.5: Remove size outliers (boxes too small/large compared to median)
    Stage 2: Classify each detection using vocabulary (TinyCLIP)
    Stage 3: Apply "Main Object Algorithm" (Score = Area × Count)
    Stage 4: Generate annotated output image

Usage:
    # Default: Use COCO classes (80 classes)
    python3 count_objects_yolox.py --image sample1.JPG --output output/
    
    # Custom: Specify your own vocabulary
    python3 count_objects_yolox.py --image sample1.JPG --vocabulary "cupcake,muffin,cake"

Features:
    - Offline YOLOX-S model (no internet required)
    - Always uses TinyCLIP zero-shot classification (wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M)
    - Default vocabulary: 80 COCO classes
    - Main Object Algorithm (Score = Total_Area × Count)
    - Custom vocabulary support for domain-specific tasks
    - Detailed object counting by class
    - Rich console output with statistics
    - Annotated visualization images
    - Detection analysis and reports

Author: YOLOX Object Counting System
Date: December 8, 2025
"""

import torch
import cv2
import sys
import os
import numpy as np
from collections import Counter, defaultdict
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

# COCO 80 class names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class TinyCLIPClassifier:
    """Zero-shot image classifier using TinyCLIP."""
    
    def __init__(self, model_name='wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M'):
        """Initialize TinyCLIP model."""
        print(f"\n📦 Loading TinyCLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        print(f"✓ TinyCLIP model loaded successfully")
        print(f"  Parameters: 8M (Vision) + 3M (Text)")
        print(f"  Training dataset: YFCC15M")
    
    def classify_crop(self, crop_bgr, vocabulary):
        """
        Classify a crop using zero-shot classification.
        
        Args:
            crop_bgr: Cropped image in BGR format (numpy array)
            vocabulary: List of class labels
        
        Returns:
            Tuple of (best_label, confidence, all_probs)
        """
        # BGR to RGB conversion
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(crop_rgb)
        
        # Generate text prompts
        text_prompts = [f"a photo of a {label}" for label in vocabulary]
        
        # Process inputs
        inputs = self.processor(
            text=text_prompts,
            images=pil_image,
            return_tensors="pt",
            padding=True
        )
        
        # Get features and compute similarity
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)[0]
        
        # Get best match
        best_idx = probs.argmax().item()
        best_label = vocabulary[best_idx]
        confidence = probs[best_idx].item()
        if best_idx == 54:
            print(f"   🏆 Best Match: {best_label} (confidence: {confidence:.3f})")
            
        return best_label, confidence, probs.cpu().numpy()
    
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


def calculate_main_object(detections):
    """
    Apply Main Object Algorithm: Score = Total_Area × Count
    
    Args:
        detections: List of detection dictionaries with 'class' and 'area'
    
    Returns:
        Tuple of (main_class, score, details)
    """
    class_stats = defaultdict(lambda: {'count': 0, 'total_area': 0, 'areas': []})
    
    for det in detections:
        cls = det['class']
        area = det['area']
        class_stats[cls]['count'] += 1
        class_stats[cls]['total_area'] += area
        class_stats[cls]['areas'].append(area)
    
    # Calculate scores
    scores = {}
    for cls, stats in class_stats.items():
        scores[cls] = stats['total_area'] * stats['count']
    
    # Find main object
    if scores:
        main_class = max(scores.keys(), key=lambda k: scores[k])
        main_score = scores[main_class]
        main_stats = class_stats[main_class]
        
        return main_class, main_score, {
            'count': main_stats['count'],
            'total_area': main_stats['total_area'],
            'avg_area': main_stats['total_area'] / main_stats['count'],
            'all_scores': scores,
            'all_stats': dict(class_stats)
        }
    
    return None, 0, {}


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


def count_objects_with_yolox(img_path, weight_path, vocabulary=None, output_dir='output', outlier_threshold=3.0, method='classification', similarity_threshold=0.80, aspect_ratio_min=0.45, aspect_ratio_max=2.2, **kwargs):
    """
    Count objects in an image using YOLOX-S detection and TinyCLIP classification or similarity clustering.
    
    Processing Pipeline (Classification):
        Stage 1: Detect all potential objects (class-agnostic YOLOX)
        Stage 1.5: Remove size outliers (too small/large compared to median)
        Stage 1.6: Remove aspect ratio outliers (unusual shapes)
        Stage 2: Classify each detection using vocabulary (TinyCLIP)
        Stage 3: Apply "Main Object Algorithm" (Score = Area × Count)
        Stage 4: Generate annotated output image
    
    Processing Pipeline (Similarity):
        Stage 1: Detect all potential objects (class-agnostic YOLOX)
        Stage 1.5: Remove size outliers (too small/large compared to median)
        Stage 1.6: Remove aspect ratio outliers (unusual shapes)
        Stage 2: Extract visual embeddings and cluster by similarity
        Stage 3: Count largest cluster as answer
        Stage 4: Generate annotated output image
    
    Args:
        img_path: Path to the input image
        weight_path: Path to YOLOX-S model weights
        vocabulary: List of class labels for classification (comma-separated string or list)
        output_dir: Directory to save visualization
        outlier_threshold: Standard deviations from median for outlier removal (0 to disable)
        method: 'classification' or 'similarity'
        similarity_threshold: Cosine similarity threshold for clustering (0-1)
        aspect_ratio_min: Minimum width/height ratio for aspect ratio filtering
        aspect_ratio_max: Maximum width/height ratio for aspect ratio filtering
    
    Returns:
        Dictionary containing counting results and statistics
    """
    model_name = 'yolox_s'
    input_size = (640, 640)

    print(f"\n{'='*70}")
    print(f"YOLOX-S + TinyCLIP OBJECT COUNTING")
    print(f"{'='*70}")
    print(f"\n🔬 Processing Pipeline:")
    print(f"  Stage 1: Object Detection (YOLOX)")
    if outlier_threshold > 0:
        print(f"  Stage 1.5: Size Outlier Filtering ({outlier_threshold} std)")

    if method == 'similarity':
        print(f"  Stage 2: Similarity Clustering (threshold={similarity_threshold})")
        print(f"  Stage 3: Count Largest Cluster")
        print(f"  Stage 4: Result Generation")
        vocabulary = None  # Not used in similarity mode
    else:
        print(f"  Stage 2: Classification (TinyCLIP)")
        print(f"  Stage 3: Main Object Algorithm (Score = Area × Count)")
        print(f"  Stage 4: Result Generation")

        # Parse vocabulary - always use TinyCLIP with COCO as default
        if vocabulary is None:
            vocabulary = COCO_CLASSES
            print(f"\n📝 Vocabulary: Using COCO classes (80 classes)")
        else:
            if isinstance(vocabulary, str):
                vocabulary = [v.strip() for v in vocabulary.split(',')]
            print(f"\n📝 Vocabulary: Custom ({len(vocabulary)} classes)")
            print(f"  Classes: {', '.join(vocabulary)}")

    # Always use TinyCLIP (for classification or embedding)
    use_clip = True

    # Check if weight file exists
    if not os.path.exists(weight_path):
        print(f"❌ Error: Weight file not found: {weight_path}")
        print(f"\nPlease download YOLOX-S weights:")
        print(f"  wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth")
        print(f"  mv yolox_s.pth weights/")
        return None

    # Initialize TinyCLIP classifier (always used)
    classifier = TinyCLIPClassifier()

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

        # Stage 2: Classification or Similarity Clustering
        print(f"\n{'='*70}")
        if method == 'similarity':
            print(f"STAGE 2: SIMILARITY CLUSTERING")
        else:
            print(f"STAGE 2: CLASSIFICATION")
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

        if method == 'similarity':
            # Similarity-based clustering
            print(f"\n🔍 Extracting embeddings for {len(crops_list)} crops...")
            embeddings = []
            for item in crops_list:
                embedding = classifier.get_visual_embedding(item['crop'])
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

        else:
            # Classification-based method
            print(f"\n🔍 Classifying {len(crops_list)} detections using TinyCLIP...")

            for item in crops_list:
                cls_name, cls_conf, probs = classifier.classify_crop(item['crop'], vocabulary)

                # DEBUG: Save crop with class name and confidence
                debug_crop = item['crop'].copy()
                text = f"{cls_name} {cls_conf:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

                text_height = text_size[1] + 10
                debug_crop_with_text = np.ones((debug_crop.shape[0] + text_height, debug_crop.shape[1], 3), dtype=np.uint8) * 255
                debug_crop_with_text[:debug_crop.shape[0], :] = debug_crop

                text_x = (debug_crop.shape[1] - text_size[0]) // 2
                text_y = debug_crop.shape[0] + text_size[1] + 5
                cv2.putText(debug_crop_with_text, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

                os.makedirs('output/debug_crops', exist_ok=True)
                cv2.imwrite(f'output/debug_crops/crop_{item["index"]:03d}.jpg', debug_crop_with_text)

                all_detections.append({
                    'class': cls_name,
                    'confidence': cls_conf,
                    'bbox': item['bbox'],
                    'obj_conf': item['obj_conf'],
                    'cls_conf': cls_conf,
                    'area': item['area'],
                    'clip_probs': probs
                })

                idx = crops_list.index(item)
                if (idx + 1) % 10 == 0 or (idx + 1) == len(crops_list):
                    print(f"  Progress: {idx+1}/{len(crops_list)} detections classified")

            print(f"✓ Classification complete!")

        # Count by class
        class_counts = Counter([d['class'] for d in all_detections])

        # Stage 3: Main Object Algorithm or Count Largest Cluster
        print(f"\n{'='*70}")
        if method == 'similarity':
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
        else:
            print(f"STAGE 3: MAIN OBJECT ALGORITHM")
            print(f"{'='*70}")
            print(f"\n🧮 Calculating Main Object (Score = Total_Area × Count)...")

            main_class, main_score, main_details = calculate_main_object(all_detections)

            if main_class:
                print(f"\n🎯 Main Object Detected: {main_class.upper()}")
                print(f"  Score: {main_score:.2f}")
                print(f"  Count: {main_details['count']}")
                print(f"  Total Area: {main_details['total_area']:.2f} px²")
                print(f"  Average Area: {main_details['avg_area']:.2f} px²")
                print(f"\n📊 All Scores:")
                for cls, score in sorted(main_details['all_scores'].items(), key=lambda x: x[1], reverse=True):
                    stats = main_details['all_stats'][cls]
                    print(f"   {cls:20s}: Score={score:.2f} (Count={stats['count']}, Area={stats['total_area']:.2f})")

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

        # Always use custom visualization with TinyCLIP
        vis_img = visualize_detections_custom(img, all_detections, main_class)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save visualization (always TinyCLIP)
        img_basename = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(output_dir, f"{img_basename}_tinyclip_counting.jpg")
        cv2.imwrite(output_path, vis_img)
        print(f"💾 Saved visualization to: {output_path}")

        # Create text report
        report_path = os.path.join(output_dir, f"{img_basename}_counting_report.txt")
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("YOLOX-S + TinyCLIP OBJECT COUNTING REPORT\n")
            f.write("="*70 + "\n\n")
            f.write("PROCESSING PIPELINE:\n")
            f.write("  Stage 1: Object Detection (YOLOX)\n")
            if method == 'similarity':
                f.write("  Stage 2: Similarity Clustering\n")
                f.write("  Stage 3: Count Largest Cluster\n")
            else:
                f.write("  Stage 2: Classification (TinyCLIP)\n")
                f.write("  Stage 3: Main Object Algorithm (Score = Area × Count)\n")
            f.write("  Stage 4: Result Generation\n\n")
            f.write(f"Image: {img_path}\n")
            f.write(f"Image size: {w}×{h} pixels\n")
            f.write(f"Model: YOLOX-S\n")
            f.write(f"Method: {method.upper()}\n")
            if method == 'similarity':
                f.write(f"Similarity Threshold: {similarity_threshold}\n")
            elif vocabulary:
                f.write(f"Vocabulary: {', '.join(vocabulary)}\n")
            f.write(f"Inference time: {inference_time*1000:.2f}ms\n\n")

            # Main Object section
            if main_class:
                f.write("="*70 + "\n")
                f.write("MAIN OBJECT (Score = Total_Area × Count):\n")
                f.write("="*70 + "\n")
                f.write(f"Class: {main_class.upper()}\n")
                f.write(f"Score: {main_score:.2f}\n")
                f.write(f"Count: {main_details['count']}\n")
                f.write(f"Total Area: {main_details['total_area']:.2f} px²\n")
                f.write(f"Average Area: {main_details['avg_area']:.2f} px²\n\n")
                f.write("All Scores:\n")
                for cls, score in sorted(main_details['all_scores'].items(), key=lambda x: x[1], reverse=True):
                    stats = main_details['all_stats'][cls]
                    f.write(f"  {cls:20s}: Score={score:.2f} (Count={stats['count']}, Area={stats['total_area']:.2f})\n")
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
            print(f"\n⭐ Main Object: {main_class.upper()} (Score: {main_score:.2f})")
            print(f"   Count: {main_details['count']}")
            print(f"   Total Area: {main_details['total_area']:.2f} px²")

        most_common = class_counts.most_common(1)[0]
        print(f"\nMost detected object: {most_common[0]} ({most_common[1]} occurrences)")
        print(f"Output visualization: {output_path}")
        print(f"Output report: {report_path}")
        print(f"{'='*70}")

        # Return results (always TinyCLIP)
        return {
            'model': model_name,
            'classifier': 'TinyCLIP',
            'vocabulary': vocabulary,
            'total_count': len(all_detections),
            'classes': class_counts,
            'detections': all_detections,
            'inference_time': inference_time,
            'most_common': most_common,
            'main_object': {
                'class': main_class,
                'score': main_score,
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
            'classifier': 'TinyCLIP',
            'vocabulary': vocabulary,
            'total_count': 0,
            'classes': {},
            'detections': [],
            'inference_time': inference_time,
            'most_common': None,
            'main_object': None,
            'visualization_path': None,
            'report_path': None,
            'image_size': (w, h)
        }


def main():
    """Main function to parse arguments and run object counting."""

    parser = argparse.ArgumentParser(
        description="Object Counting with YOLOX-S and TinyCLIP Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Processing Pipeline:
  Stage 1: Object Detection (YOLOX - class-agnostic)
  Stage 1.5: Size Outlier Filtering (removes boxes too small/large)
  Stage 2: Classification (TinyCLIP - zero-shot)
  Stage 3: Main Object Algorithm (Score = Area × Count)
  Stage 4: Result Generation

Examples:
  # Use default COCO classes (80 classes with TinyCLIP)
  python3 count_objects_yolox.py --image sample1.JPG --output output/
  
  # Use custom vocabulary (TinyCLIP with custom classes)
  python3 count_objects_yolox.py --image sample1.JPG --vocabulary "cupcake,muffin,cake"
  
  # Custom outlier threshold (more aggressive filtering)
  python3 count_objects_yolox.py --image sample1.JPG --outlier-threshold 2.0
  
  # Disable outlier filtering
  python3 count_objects_yolox.py --image sample1.JPG --outlier-threshold 0
  
  # Custom weights and vocabulary
  python3 count_objects_yolox.py --image test.jpg --weights weights/yolox_s.pth --vocabulary "person,car,bike"
  
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
        '--vocabulary',
        type=str,
        default=None,
        help='Comma-separated list of class labels for zero-shot classification (e.g., "cupcake,muffin,cake"). If not provided, uses default 80 COCO classes with TinyCLIP.'
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=["classification", "similarity"],
        default="similarity",
        help='Counting method: "classification" uses TinyCLIP with vocabulary, "similarity" clusters visually similar crops (default: classification)',
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
            args.vocabulary, 
            output_dir,
            args.outlier_threshold,
            method=args.method,
            similarity_threshold=args.similarity_threshold,
            aspect_ratio_min=args.aspect_ratio_min,
            aspect_ratio_max=args.aspect_ratio_max,
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
