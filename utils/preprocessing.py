"""
Image preprocessing utilities for YOLOX object detection.
"""

import cv2
import numpy as np
import torch


def letterbox_resize(image, target_size=(640, 640), color=(114, 114, 114)):
    """
    Resize image with letterbox padding to maintain aspect ratio.
    
    Args:
        image: Input image (numpy array, HWC format)
        target_size: Target size (height, width)
        color: Padding color (BGR format)
    
    Returns:
        resized_image: Padded and resized image
        ratio: Resize ratio (used for postprocessing)
        (pad_w, pad_h): Padding applied (left/right, top/bottom)
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scaling ratio
    ratio = min(target_h / h, target_w / w)
    
    # Calculate new dimensions
    new_h, new_w = int(h * ratio), int(w * ratio)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create canvas with padding color
    canvas = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    
    # Calculate padding
    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2
    
    # Place resized image on canvas
    canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    
    return canvas, ratio, (pad_w, pad_h)


def preprocess_image(image_path, input_size=(640, 640), device='cpu'):
    """
    Complete preprocessing pipeline for YOLOX inference.
    
    Args:
        image_path: Path to input image or numpy array
        input_size: Model input size (height, width)
        device: Device to put tensor on ('cpu' or 'cuda')
    
    Returns:
        img_tensor: Preprocessed image tensor (1, 3, H, W)
        img_info: Dictionary with original image info
    """
    # Load image
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image from {image_path}")
    else:
        img = image_path
    
    # Store original info
    original_h, original_w = img.shape[:2]
    img_info = {
        'original_shape': (original_h, original_w),
        'raw_img': img.copy()
    }
    
    # Letterbox resize
    img_resized, ratio, padding = letterbox_resize(img, input_size)
    img_info['ratio'] = ratio
    img_info['padding'] = padding
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # Convert to CHW format
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    
    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0).float()
    
    # Move to device
    img_tensor = img_tensor.to(device)
    
    return img_tensor, img_info


def preprocess_batch(image_paths, input_size=(640, 640), device='cpu'):
    """
    Preprocess multiple images for batch inference.
    
    Args:
        image_paths: List of image paths or numpy arrays
        input_size: Model input size
        device: Device to put tensors on
    
    Returns:
        batch_tensor: Batched tensor (N, 3, H, W)
        batch_info: List of image info dictionaries
    """
    tensors = []
    infos = []
    
    for img_path in image_paths:
        tensor, info = preprocess_image(img_path, input_size, device)
        tensors.append(tensor)
        infos.append(info)
    
    # Stack into batch
    batch_tensor = torch.cat(tensors, dim=0)
    
    return batch_tensor, infos
