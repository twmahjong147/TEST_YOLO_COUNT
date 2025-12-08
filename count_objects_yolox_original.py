#!/usr/bin/env python3
"""
Object Counting with YOLOX-S
=============================

A production-ready object counting program using offline YOLOX-S model.
Based on test_sample1.py with enhanced counting functionality.

Usage:
    python3 count_objects_yolox.py --image sample1.JPG --output output/

Features:
    - Offline YOLOX-S model (no internet required)
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
from collections import Counter
import time
import argparse
from pathlib import Path

# Add official YOLOX to path
sys.path.insert(0, "official_yolox")

from yolox.exp import get_exp
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess
from utils.visualizer import COCO_CLASSES, visualize_detections


def count_objects_with_yolox(img_path, weight_path, output_dir="output"):
    """
    Count objects in an image using YOLOX-S model.

    Args:
        img_path: Path to the input image
        weight_path: Path to YOLOX-S model weights
        output_dir: Directory to save visualization

    Returns:
        Dictionary containing counting results and statistics
    """
    model_name = "yolox_s"
    input_size = (640, 640)

    print(f"\n{'='*70}")
    print(f"YOLOX-S OBJECT COUNTING")
    print(f"{'='*70}")

    # Check if weight file exists
    if not os.path.exists(weight_path):
        print(f"❌ Error: Weight file not found: {weight_path}")
        print(f"\nPlease download YOLOX-S weights:")
        print(
            f"  wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth"
        )
        print(f"  mv yolox_s.pth weights/")
        return None

    # Load model
    print(f"\n📦 Loading YOLOX-S model...")
    exp = get_exp(f"official_yolox/exps/default/{model_name}.py", None)
    model = exp.get_model()
    ckpt = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"✓ Model loaded successfully")
    print(f"  Model: {model_name.upper()}")
    print(f"  Input size: {input_size}")
    print(f"  Confidence threshold: {exp.test_conf}")
    print(f"  NMS threshold: {exp.nmsthre}")

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
        outputs = postprocess(
            outputs, 80, exp.test_conf, exp.nmsthre, class_agnostic=True
        )
    inference_time = time.time() - start_time

    print(f"⏱️  Inference time: {inference_time*1000:.2f}ms")

    # Process results
    if outputs[0] is not None:
        output = outputs[0].cpu()
        output[:, :4] /= ratio
        n = len(output)

        print(f"\n✅ Detected {n} objects!")

        # Analyze detections
        class_counts = Counter()
        all_detections = []

        for i in range(n):
            x1, y1, x2, y2 = output[i, :4].numpy()
            obj_conf = output[i, 4].item()
            cls_conf = output[i, 5].item()
            cls_id = int(output[i, 6].item())
            cls_name = COCO_CLASSES[cls_id]

            combined_conf = obj_conf * cls_conf
            class_counts[cls_name] += 1

            all_detections.append(
                {
                    "class": cls_name,
                    "confidence": combined_conf,
                    "bbox": [x1, y1, x2, y2],
                    "obj_conf": obj_conf,
                    "cls_conf": cls_conf,
                }
            )

        # Print counting summary
        print(f"\n{'='*70}")
        print("📊 OBJECT COUNTING RESULTS")
        print(f"{'='*70}")
        print(f"\nTotal objects detected: {n}")
        print(f"\nObject breakdown by class:")
        print("-" * 70)

        for cls_name, count in class_counts.most_common():
            avg_conf = np.mean(
                [d["confidence"] for d in all_detections if d["class"] == cls_name]
            )
            percentage = (count / n) * 100
            print(
                f"   {cls_name:20s}: {count:3d} objects ({percentage:5.1f}%) - avg conf: {avg_conf:.3f}"
            )

        print("-" * 70)
        print(f"\nUnique object classes: {len(class_counts)}")

        # Show top 10 detections
        print(f"\n🎯 Top 10 Detections (by confidence):")
        print("-" * 70)
        sorted_dets = sorted(
            all_detections, key=lambda x: x["confidence"], reverse=True
        )
        for i, det in enumerate(sorted_dets[:10], 1):
            bbox_str = f"[{det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}, {det['bbox'][2]:.0f}, {det['bbox'][3]:.0f}]"
            print(
                f"   {i:2d}. {det['class']:15s} - conf: {det['confidence']:.3f} - bbox: {bbox_str}"
            )
        print("-" * 70)

        # Show statistics
        print(f"\n📈 Statistics:")
        all_confidences = [d["confidence"] for d in all_detections]
        print(f"   Average confidence: {np.mean(all_confidences):.3f}")
        print(f"   Median confidence:  {np.median(all_confidences):.3f}")
        print(f"   Min confidence:     {np.min(all_confidences):.3f}")
        print(f"   Max confidence:     {np.max(all_confidences):.3f}")

        # Visualize
        print(f"\n🎨 Generating visualization...")
        vis_img = visualize_detections(
            img, output.numpy(), conf_threshold=exp.test_conf
        )

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save visualization
        img_basename = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(output_dir, f"{img_basename}_yolox_s_counting.jpg")
        cv2.imwrite(output_path, vis_img)
        print(f"💾 Saved visualization to: {output_path}")

        # Create text report
        report_path = os.path.join(output_dir, f"{img_basename}_counting_report.txt")
        with open(report_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("YOLOX-S OBJECT COUNTING REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Image: {img_path}\n")
            f.write(f"Image size: {w}×{h} pixels\n")
            f.write(f"Model: YOLOX-S\n")
            f.write(f"Inference time: {inference_time*1000:.2f}ms\n\n")
            f.write(f"Total objects detected: {n}\n")
            f.write(f"Unique classes: {len(class_counts)}\n\n")
            f.write("Object breakdown by class:\n")
            f.write("-" * 70 + "\n")
            for cls_name, count in class_counts.most_common():
                avg_conf = np.mean(
                    [d["confidence"] for d in all_detections if d["class"] == cls_name]
                )
                percentage = (count / n) * 100
                f.write(
                    f"   {cls_name:20s}: {count:3d} objects ({percentage:5.1f}%) - avg conf: {avg_conf:.3f}\n"
                )
            f.write("-" * 70 + "\n\n")
            f.write("Top 10 Detections:\n")
            f.write("-" * 70 + "\n")
            for i, det in enumerate(sorted_dets[:10], 1):
                f.write(
                    f"   {i:2d}. {det['class']:15s} - conf: {det['confidence']:.3f}\n"
                )
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

        most_common = class_counts.most_common(1)[0]
        print(
            f"\nMost detected object: {most_common[0]} ({most_common[1]} occurrences)"
        )
        print(f"Output visualization: {output_path}")
        print(f"Output report: {report_path}")
        print(f"{'='*70}")

        # Return results
        return {
            "model": model_name,
            "total_count": n,
            "classes": class_counts,
            "detections": all_detections,
            "inference_time": inference_time,
            "most_common": most_common,
            "visualization_path": output_path,
            "report_path": report_path,
            "image_size": (w, h),
        }
    else:
        print("\n❌ No objects detected")
        print(f"{'='*70}")
        return {
            "model": model_name,
            "total_count": 0,
            "classes": {},
            "detections": [],
            "inference_time": inference_time,
            "most_common": None,
            "visualization_path": None,
            "report_path": None,
            "image_size": (w, h),
        }


def main():
    """Main function to parse arguments and run object counting."""

    parser = argparse.ArgumentParser(
        description="Object Counting with YOLOX-S",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 count_objects_yolox.py --image sample1.JPG --output output/
  python3 count_objects_yolox.py --image test.jpg --output results/ --weights weights/yolox_s.pth
  
Requirements:
  - YOLOX-S weights must be downloaded first
  - Download from: https://github.com/Megvii-BaseDetection/YOLOX/releases/
        """,
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image (e.g., sample1.JPG)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for results (default: output/)",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="weights/yolox_s.pth",
        help="Path to YOLOX-S weights file (default: weights/yolox_s.pth)",
    )

    args = parser.parse_args()

    # Validate image path
    if not os.path.exists(args.image):
        print(f"❌ Error: Image file not found: {args.image}")
        return 1

    # Remove trailing slash from output directory
    output_dir = args.output.rstrip("/")

    # Run counting
    try:
        result = count_objects_with_yolox(args.image, args.weights, output_dir)

        if result is None:
            return 1

        return 0

    except Exception as e:
        print(f"\n❌ Error during counting: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
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
