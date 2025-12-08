"""
Postprocessing utilities for YOLOX object detection.
Handles NMS, confidence filtering, and coordinate transformation.
"""

import torch
import numpy as np


def apply_nms(boxes, scores, iou_threshold=0.45):
    """
    Apply Non-Maximum Suppression (NMS) to filter overlapping boxes.
    
    Args:
        boxes: Bounding boxes (N, 4) in [x1, y1, x2, y2] format
        scores: Confidence scores (N,)
        iou_threshold: IoU threshold for NMS
    
    Returns:
        keep_indices: Indices of boxes to keep
    """
    if len(boxes) == 0:
        return []
    
    # Convert to tensor if needed
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    
    # Apply torchvision NMS
    from torchvision.ops import nms
    keep = nms(boxes, scores, iou_threshold)
    
    return keep.cpu().numpy()


def postprocess_outputs(outputs, img_info, conf_threshold=0.25, nms_threshold=0.45, num_classes=80):
    """
    Postprocess YOLOX model outputs to get final detections.
    
    Args:
        outputs: Raw model outputs (1, N, 85) where N is number of predictions
                 Format: [x_center, y_center, w, h, obj_conf, cls1, cls2, ..., cls80]
        img_info: Dictionary with original image info (ratio, padding, original_shape)
        conf_threshold: Confidence threshold for filtering
        nms_threshold: IoU threshold for NMS
        num_classes: Number of object classes (80 for COCO)
    
    Returns:
        detections: Numpy array (M, 7) with [x1, y1, x2, y2, obj_conf, class_conf, class_id]
                    or None if no detections
    """
    if outputs is None or len(outputs) == 0:
        return None
    
    # Move to CPU if needed
    if isinstance(outputs, torch.Tensor):
        if outputs.is_cuda:
            outputs = outputs.cpu()
    
    # Get first batch item
    if len(outputs.shape) == 3:
        outputs = outputs[0]  # (N, 85)
    
    # Parse outputs: [x_center, y_center, w, h, obj_conf, cls1, cls2, ..., cls80]
    boxes_xywh = outputs[:, :4]  # (N, 4)
    obj_conf = outputs[:, 4]  # (N,)
    class_scores = outputs[:, 5:5+num_classes]  # (N, 80)
    
    # Get best class for each prediction
    class_conf, class_pred = torch.max(class_scores, dim=-1)
    
    # Combine object confidence and class confidence
    conf = obj_conf * class_conf
    
    # Filter by confidence threshold
    mask = conf >= conf_threshold
    
    if mask.sum() == 0:
        return None
    
    # Apply mask
    boxes_xywh = boxes_xywh[mask]
    obj_conf = obj_conf[mask]
    class_conf = class_conf[mask]
    class_pred = class_pred[mask]
    conf = conf[mask]
    
    # Convert from center format (x_center, y_center, w, h) to corner format (x1, y1, x2, y2)
    boxes_xyxy = torch.zeros_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2
    
    # Scale boxes back to original image size
    ratio = img_info['ratio']
    pad_w, pad_h = img_info['padding']
    
    # Remove padding and scale back
    boxes_xyxy[:, 0] = (boxes_xyxy[:, 0] - pad_w) / ratio  # x1
    boxes_xyxy[:, 1] = (boxes_xyxy[:, 1] - pad_h) / ratio  # y1
    boxes_xyxy[:, 2] = (boxes_xyxy[:, 2] - pad_w) / ratio  # x2
    boxes_xyxy[:, 3] = (boxes_xyxy[:, 3] - pad_h) / ratio  # y2
    
    # Clip to image boundaries
    original_h, original_w = img_info['original_shape']
    boxes_xyxy[:, 0] = torch.clamp(boxes_xyxy[:, 0], 0, original_w)
    boxes_xyxy[:, 1] = torch.clamp(boxes_xyxy[:, 1], 0, original_h)
    boxes_xyxy[:, 2] = torch.clamp(boxes_xyxy[:, 2], 0, original_w)
    boxes_xyxy[:, 3] = torch.clamp(boxes_xyxy[:, 3], 0, original_h)
    
    # Apply NMS
    keep_indices = apply_nms(boxes_xyxy, conf, nms_threshold)
    
    if len(keep_indices) == 0:
        return None
    
    # Filter by NMS results
    boxes_xyxy = boxes_xyxy[keep_indices]
    obj_conf = obj_conf[keep_indices]
    class_conf = class_conf[keep_indices]
    class_pred = class_pred[keep_indices]
    
    # Combine into final format [x1, y1, x2, y2, obj_conf, class_conf, class_id]
    detections = torch.cat([
        boxes_xyxy,
        obj_conf.unsqueeze(1),
        class_conf.unsqueeze(1),
        class_pred.unsqueeze(1).float()
    ], dim=1).numpy()
    
    return detections


def filter_by_class(detections, class_ids):
    """
    Filter detections by specific class IDs.
    
    Args:
        detections: Array of detections (M, 7)
        class_ids: List of class IDs to keep
    
    Returns:
        Filtered detections
    """
    if detections is None:
        return None
    
    mask = np.isin(detections[:, 6], class_ids)
    return detections[mask] if mask.any() else None
