"""
Visualization utilities for YOLOX object detection.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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


def get_color(class_id, num_classes=80):
    """
    Generate a unique color for each class.
    
    Args:
        class_id: Class ID
        num_classes: Total number of classes
    
    Returns:
        color: BGR color tuple
    """
    hue = (class_id * 360 // num_classes) % 360
    
    # Convert HSV to RGB
    import colorsys
    rgb = colorsys.hsv_to_rgb(hue / 360, 0.8, 0.95)
    bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
    
    return bgr


def visualize_detections(image, detections, class_names=COCO_CLASSES, 
                        conf_threshold=0.3, thickness=2, font_scale=0.6):
    """
    Draw bounding boxes and labels on image.
    
    Args:
        image: Input image (numpy array or path)
        detections: Detections array (N, 7) [x1, y1, x2, y2, obj_conf, class_conf, class_id]
        class_names: List of class names
        conf_threshold: Minimum confidence to display
        thickness: Box line thickness
        font_scale: Font size scale
    
    Returns:
        annotated_image: Image with drawn bounding boxes
    """
    # Load image if path provided
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image.copy()
    
    if detections is None or len(detections) == 0:
        return img
    
    # Draw each detection
    for det in detections:
        x1, y1, x2, y2, obj_conf, class_conf, class_id = det
        
        # Calculate final confidence
        confidence = obj_conf * class_conf
        
        if confidence < conf_threshold:
            continue
        
        # Get class info
        class_id = int(class_id)
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        
        # Get color
        color = get_color(class_id)
        
        # Draw bounding box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label
        label = f"{class_name}: {confidence:.2f}"
        
        # Get label size
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # Draw label background
        cv2.rectangle(
            img,
            (x1, y1 - label_h - baseline - 5),
            (x1 + label_w, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            img,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )
    
    return img


def save_visualization(image, detections, output_path, **kwargs):
    """
    Save visualization to file.
    
    Args:
        image: Input image
        detections: Detections array
        output_path: Output file path
        **kwargs: Additional arguments for visualize_detections
    """
    annotated = visualize_detections(image, detections, **kwargs)
    cv2.imwrite(output_path, annotated)
    return annotated


def print_detection_summary(detections, class_names=COCO_CLASSES):
    """
    Print a summary of detected objects.
    
    Args:
        detections: Detections array
        class_names: List of class names
    """
    if detections is None or len(detections) == 0:
        print("No objects detected.")
        return
    
    print(f"\nDetected {len(detections)} objects:")
    print("-" * 60)
    
    # Count objects by class
    from collections import Counter
    class_ids = detections[:, 6].astype(int)
    class_counts = Counter(class_ids)
    
    for class_id, count in sorted(class_counts.items()):
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        avg_conf = detections[class_ids == class_id][:, 4:6].prod(axis=1).mean()
        print(f"  {class_name}: {count} (avg conf: {avg_conf:.3f})")
    
    print("-" * 60)


def create_detection_grid(images, detections_list, output_path=None, grid_size=None):
    """
    Create a grid visualization of multiple images with detections.
    
    Args:
        images: List of images
        detections_list: List of detection arrays
        output_path: Optional output path
        grid_size: Tuple (rows, cols), auto-calculated if None
    
    Returns:
        grid_image: Grid visualization
    """
    n = len(images)
    
    if grid_size is None:
        # Calculate grid size
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    else:
        rows, cols = grid_size
    
    # Visualize each image
    vis_images = []
    for img, dets in zip(images, detections_list):
        vis_img = visualize_detections(img, dets)
        vis_images.append(vis_img)
    
    # Get max dimensions
    max_h = max(img.shape[0] for img in vis_images)
    max_w = max(img.shape[1] for img in vis_images)
    
    # Create grid
    grid = np.zeros((rows * max_h, cols * max_w, 3), dtype=np.uint8)
    
    for idx, img in enumerate(vis_images):
        row = idx // cols
        col = idx % cols
        h, w = img.shape[:2]
        grid[row*max_h:row*max_h+h, col*max_w:col*max_w+w] = img
    
    if output_path:
        cv2.imwrite(output_path, grid)
    
    return grid
