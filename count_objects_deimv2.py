#!/usr/bin/env python3
"""
Object Counting with DEIMv2 and TinyCLIP Similarity Clustering
A variant of count_objects_yolox.py that uses DEIMv2 for detection (DEIMv2-main).
"""

import sys
import os
import time
import argparse
from collections import Counter

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import AgglomerativeClustering

# Make DEIMv2 package importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DEIMv2-main'))
from engine.core.yaml_config import YAMLConfig


class TinyCLIPEmbedder:
    """Visual embedding extractor using TinyCLIP (same as in YOLOX script)."""
    def __init__(self, model_path='weights/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M'):
        print(f"\n📦 Loading TinyCLIP model from: {model_path}")
        if os.path.exists(model_path):
            self.model = CLIPModel.from_pretrained(model_path, local_files_only=True)
            self.processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        else:
            self.model = CLIPModel.from_pretrained(model_path)
            self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model.eval()

    def get_visual_embedding(self, crop_bgr):
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(crop_rgb)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()[0]


def calculate_iou(box1, box2):
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
    if len(detections) <= 1:
        return detections
    sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    keep = []
    suppressed = set()
    for i, det1 in enumerate(sorted_dets):
        if i in suppressed:
            continue
        keep.append(det1)
        for j in range(i + 1, len(sorted_dets)):
            if j in suppressed:
                continue
            det2 = sorted_dets[j]
            iou = calculate_iou(det1['bbox'], det2['bbox'])
            if iou > iou_threshold:
                suppressed.add(j)
    return keep


def filter_by_area_consistency(crops_list, area_variance_threshold=1.0):
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
    if len(boxes) < 3:
        return list(range(len(boxes)))
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    areas_np = areas.numpy()
    median_area = np.median(areas_np)
    std_area = np.std(areas_np)
    min_area = median_area - std_threshold * std_area
    max_area = median_area + std_threshold * std_area
    keep = []
    for i, area in enumerate(areas_np):
        if min_area <= area <= max_area:
            keep.append(i)
    return keep


def remove_aspect_ratio_outliers(boxes, aspect_ratio_variance_threshold: float = 1.0) -> list:
    if len(boxes) == 0:
        return []
    if len(boxes) <= 2:
        return list(range(len(boxes)))
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    valid_mask = heights > 0
    if not valid_mask.any():
        return []
    aspect_ratios = np.zeros(len(boxes))
    aspect_ratios[valid_mask.numpy()] = (widths[valid_mask] / heights[valid_mask]).numpy()
    valid_ratios = aspect_ratios[valid_mask.numpy()]
    median_ratio = np.median(valid_ratios)
    std_ratio = np.std(valid_ratios)
    if std_ratio == 0:
        return list(range(len(boxes)))
    min_ratio = median_ratio - aspect_ratio_variance_threshold * std_ratio
    max_ratio = median_ratio + aspect_ratio_variance_threshold * std_ratio
    keep = []
    for i, ratio in enumerate(aspect_ratios):
        if valid_mask[i] and min_ratio <= ratio <= max_ratio:
            keep.append(i)
    return keep


def visualize_detections_custom(img, detections, main_class=None):
    vis_img = img.copy()
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cls_name = det['class']
        confidence = det['confidence']
        if main_class and cls_name == main_class:
            color = (0, 255, 0)
            thickness = 3
        else:
            color = (255, 0, 0)
            thickness = 2
        cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        label = f"{cls_name}: {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        cv2.rectangle(vis_img,
                     (int(x1), int(y1) - text_height - 10),
                     (int(x1) + text_width + 10, int(y1)),
                     color, -1)
        cv2.putText(vis_img, label, (int(x1) + 5, int(y1) - 5), font, font_scale, (255, 255, 255), font_thickness)
    return vis_img


def count_objects_with_deimv2(img_path, config_path, resume_checkpoint, device='cpu', output_dir='output',
                              outlier_threshold=3.0, similarity_threshold=0.80,
                              aspect_ratio_variance_threshold=1.0, area_variance_threshold=2.0,
                              iou_threshold=0.5, conf_threshold=0.001):
    print(f"\nDEIMv2 + TinyCLIP SIMILARITY CLUSTERING\nConfig: {config_path}\nCheckpoint: {resume_checkpoint}\nDevice: {device}")

    # Load DEIMv2 config and weights
    cfg = YAMLConfig(config_path)

    # Load checkpoint
    if not os.path.exists(resume_checkpoint):
        print(f"❌ Checkpoint not found: {resume_checkpoint}")
        return None
    ckpt = torch.load(resume_checkpoint, map_location='cpu')
    if 'ema' in ckpt:
        state = ckpt['ema']['module']
    elif 'model' in ckpt:
        state = ckpt['model']
    else:
        state = ckpt

    # Load state into cfg.model
    try:
        cfg.model.load_state_dict(state)
    except Exception:
        # sometimes state is already the deploy modules, try direct load
        try:
            cfg.model.load_state_dict(state, strict=False)
        except Exception as e:
            print(f"Warning: failed to load full state: {e}")

    # Build deploy model and postprocessor
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device_t = torch.device(device)
    model = Model().to(device_t)

    # Image transforms (follow tools/inference/torch_inf.py)
    img_size = cfg.yaml_cfg.get('eval_spatial_size', [640, 640])
    vit_backbone = 'DINOv3STAs' in cfg.yaml_cfg
    transforms = T.Compose([
        T.Resize(tuple(img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if vit_backbone else T.Lambda(lambda x: x)
    ])

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Failed to load image: {img_path}")
        return None
    h, w = img.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    orig_size = torch.tensor([[w, h]]).to(device_t)
    im_tensor = transforms(pil).unsqueeze(0).to(device_t)

    print("\n🚀 Running DEIMv2 inference...")
    start_time = time.time()
    with torch.no_grad():
        outputs = model(im_tensor, orig_size)
    inference_time = time.time() - start_time
    print(f"⏱️  Inference time: {inference_time*1000:.2f}ms")

    # outputs expected: (labels, boxes, scores)
    try:
        labels, boxes, scores = outputs
    except Exception:
        # If model returned list or tuple, try first element
        if isinstance(outputs, (list, tuple)) and len(outputs) >= 1:
            labels, boxes, scores = outputs[0]
        else:
            print("❌ Unexpected model output format")
            return None

    # Filter by confidence threshold
    boxes_np = boxes.cpu().numpy()
    scores_np = scores.cpu().numpy()
    labels_np = labels.cpu().numpy()

    keep_idxs = [i for i, s in enumerate(scores_np) if s > conf_threshold]
    if len(keep_idxs) == 0:
        print("❌ No detections above confidence threshold")
        return {
            'model': 'deimv2', 'method': 'similarity', 'total_count': 0,
            'detections': [], 'inference_time': inference_time
        }

    detections_raw = []
    for i in keep_idxs:
        box = boxes_np[i].tolist()  # expected [x1,y1,x2,y2]
        score = float(scores_np[i])
        lbl = int(labels_np[i]) if labels_np is not None else -1
        detections_raw.append({'bbox': box, 'confidence': score, 'label': lbl})

    # Optionally remove size outliers
    boxes_for_stats = np.array([d['bbox'] for d in detections_raw])
    if outlier_threshold > 0 and len(boxes_for_stats) >= 3:
        keep_idx_stat = remove_size_outliers(boxes_for_stats, std_threshold=outlier_threshold)
        detections_raw = [detections_raw[i] for i in keep_idx_stat]

    # Aspect ratio filtering
    boxes_for_stats = np.array([d['bbox'] for d in detections_raw])
    keep_idx_aspect = remove_aspect_ratio_outliers(boxes_for_stats, aspect_ratio_variance_threshold=aspect_ratio_variance_threshold)
    detections_raw = [detections_raw[i] for i in keep_idx_aspect]

    # Build crops
    crops_list = []
    for idx, det in enumerate(detections_raw):
        x1, y1, x2, y2 = det['bbox']
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i, y2i = min(w, int(x2)), min(h, int(y2))
        crop = img[y1i:y2i, x1i:x2i]
        if crop.size == 0:
            continue
        area = (x2 - x1) * (y2 - y1)
        crops_list.append({'crop': crop, 'bbox': det['bbox'], 'obj_conf': det['confidence'], 'area': area, 'index': idx})

    # Area-based filtering
    n_before_area = len(crops_list)
    crops_list = filter_by_area_consistency(crops_list, area_variance_threshold=area_variance_threshold)
    n_after_area = len(crops_list)
    if n_before_area > n_after_area:
        print(f"  Removed {n_before_area - n_after_area} crops with inconsistent areas; kept {n_after_area} crops")

    # Embeddings
    embedder = TinyCLIPEmbedder()
    embeddings = []
    for item in crops_list:
        embeddings.append(embedder.get_visual_embedding(item['crop']))
    if len(embeddings) == 0:
        print("❌ No valid crops for embedding extraction")
        return None
    embeddings = np.array(embeddings)

    # Clustering
    cluster_labels, cluster_counts = cluster_by_similarity(embeddings, similarity_threshold)
    largest_cluster_id = max(cluster_counts, key=cluster_counts.get)
    main_count = cluster_counts[largest_cluster_id]

    all_detections = []
    os.makedirs(os.path.join(output_dir, 'debug_crops'), exist_ok=True)
    for idx, item in enumerate(crops_list):
        cluster_id = int(cluster_labels[idx])
        is_main = (cluster_id == largest_cluster_id)
        all_detections.append({
            'class': f'cluster_{cluster_id}',
            'confidence': 1.0 if is_main else 0.5,
            'bbox': item['bbox'],
            'obj_conf': item['obj_conf'],
            'area': item['area'],
            'cluster_id': cluster_id,
            'is_main_cluster': is_main
        })

        # debug crop save
        debug_crop = item['crop'].copy()
        text = f"Cluster {cluster_id}" + (" (MAIN)" if is_main else "")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(debug_crop, text, (5, 20), font, 0.6, (0, 255, 0) if is_main else (0, 0, 0), 1)
        cv2.imwrite(os.path.join(output_dir, 'debug_crops', f'{cluster_id}_crop_{item["index"]:03d}.jpg'), debug_crop)

    # IoU-based filtering
    n_before_iou = len(all_detections)
    all_detections = filter_contained_across(all_detections, iou_threshold=iou_threshold)
    n_after_iou = len(all_detections)
    if n_before_iou > n_after_iou:
        cluster_counts_after = Counter([d['cluster_id'] for d in all_detections])
        largest_cluster_id = max(cluster_counts_after, key=cluster_counts_after.get)
        main_count = cluster_counts_after[largest_cluster_id]
        cluster_counts = dict(cluster_counts_after)

    # Summaries
    class_counts = Counter([d['class'] for d in all_detections])
    main_class = f'cluster_{largest_cluster_id}'

    print(f"\n✅ Final Count: {main_count} objects (largest similarity cluster)")

    # Visualization
    os.makedirs(output_dir, exist_ok=True)
    vis_img = visualize_detections_custom(img, all_detections, main_class)
    img_basename = os.path.splitext(os.path.basename(img_path))[0]
    output_path = os.path.join(output_dir, f"{img_basename}_deimv2_similarity_counting.jpg")
    cv2.imwrite(output_path, vis_img)
    print(f"💾 Saved visualization to: {output_path}")

    # Report
    report_path = os.path.join(output_dir, f"{img_basename}_deimv2_counting_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"DEIMv2 + TinyCLIP SIMILARITY CLUSTERING REPORT\nConfig: {config_path}\nCheckpoint: {resume_checkpoint}\n\n")
        f.write(f"Final count (largest cluster): {main_count}\nTotal detections: {len(all_detections)}\n")

    return {
        'model': 'deimv2',
        'method': 'similarity',
        'total_count': main_count,
        'clusters': cluster_counts,
        'detections': all_detections,
        'inference_time': inference_time,
        'visualization_path': output_path,
        'report_path': report_path,
        'image_size': (w, h)
    }


def main():
    parser = argparse.ArgumentParser(description='Object Counting with DEIMv2 + TinyCLIP')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--config', required=True, help='DEIMv2 config yml (from DEIMv2-main/configs)')
    parser.add_argument('--resume', required=True, help='Path to DEIMv2 checkpoint (.pth)')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--device', default='cpu', help='Torch device (cpu or cuda:0)')

    parser.add_argument('--similarity-threshold', type=float, default=0.80)
    parser.add_argument('--outlier-threshold', type=float, default=3.0)
    parser.add_argument('--aspect-ratio-variance-threshold', type=float, default=1.0)
    parser.add_argument('--area-variance-threshold', type=float, default=2.0)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--conf-threshold', type=float, default=0.001)

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        return 1
    if not os.path.exists(args.config):
        print(f"❌ Config not found: {args.config}")
        return 1
    if not os.path.exists(args.resume):
        print(f"❌ Checkpoint not found: {args.resume}")
        return 1

    res = count_objects_with_deimv2(
        args.image, args.config, args.resume, device=args.device, output_dir=args.output,
        outlier_threshold=args.outlier_threshold, similarity_threshold=args.similarity_threshold,
        aspect_ratio_variance_threshold=args.aspect_ratio_variance_threshold,
        area_variance_threshold=args.area_variance_threshold,
        iou_threshold=args.iou_threshold, conf_threshold=args.conf_threshold
    )

    if res is None:
        return 1
    return 0


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation interrupted by user")
        sys.exit(130)
