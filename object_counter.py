#!/usr/bin/env python3
"""
AICounter - Object Counting Application
Based on PRD requirements for zero-shot object detection and counting
Uses real AI models: YOLOX-Nano and TinyCLIP
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import json
import torch
import torchvision
from PIL import Image, ImageDraw
from transformers import CLIPProcessor, CLIPModel
import sys
sys.path.insert(0, str(Path.home() / '.cache/torch/hub/Megvii-BaseDetection_YOLOX_main'))
from yolox.utils import postprocess


def create_ground_truth_image() -> Tuple[Image.Image, List[Dict]]:
    """
    Create a sample image with 5 known objects.
    Returns: (image, ground_truth_objects)
    """
    width, height = 800, 600
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Define 5 objects with known positions and labels
    # These represent realistic objects that CLIP can classify
    objects = [
        {'name': 'person', 'color': (200, 50, 50), 'bbox': (100, 100, 200, 250)},
        {'name': 'car', 'color': (50, 100, 200), 'bbox': (300, 150, 500, 280)},
        {'name': 'dog', 'color': (150, 100, 50), 'bbox': (520, 120, 650, 240)},
        {'name': 'cup', 'color': (100, 200, 100), 'bbox': (150, 350, 230, 480)},
        {'name': 'book', 'color': (150, 50, 150), 'bbox': (450, 400, 600, 520)}
    ]
    
    # Draw objects with some visual features
    for obj in objects:
        # Main rectangle
        draw.rectangle(obj['bbox'], fill=obj['color'])
        
        # Add border for definition
        draw.rectangle(obj['bbox'], outline=(0, 0, 0), width=2)
    
    return image, objects


class HybridVocabulary:
    """Manages the hybrid vocabulary system with LIFO dynamic list and static base list"""
    
    def __init__(self, max_recent_items: int = 10):
        self.max_recent_items = max_recent_items
        self.recent_items = []  # LIFO list
        self.base_vocabulary = [
            # Bakery items
            "cookie", "bun", "bread", "pastry",
            # Hardware items
            "bolt", "screw", "nut", "washer", "nail",
            # Small items
            "button", "coin", "pill", "candy",
            # Common objects for testing
            "person", "car", "dog", "cat", "bird",
            "cup", "mug", "glass", "bottle",
            "book", "phone", "laptop", "keyboard",
            "chair", "table", "pen", "pencil"
        ]
    
    def get_vocabulary(self) -> List[str]:
        """Returns combined vocabulary: recent items first, then base vocabulary"""
        combined = self.recent_items.copy()
        for item in self.base_vocabulary:
            if item not in combined:
                combined.append(item)
        return combined
    
    def add_recent_item(self, item_name: str):
        """Adds an item to recent history (LIFO)"""
        item_name = item_name.lower().strip()
        if item_name in self.recent_items:
            self.recent_items.remove(item_name)
        self.recent_items.insert(0, item_name)
        if len(self.recent_items) > self.max_recent_items:
            self.recent_items = self.recent_items[:self.max_recent_items]


class ObjectDetector:
    """YOLOX-Nano for class-agnostic object detection"""
    
    def __init__(self):
        self.confidence_threshold = 0.000005  # Tuned threshold for cupcake detection
        self.nms_threshold = 0.3  # Moderate NMS to merge overlapping detections
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load YOLOX-Nano model (required - no fallback)
        print("Loading YOLOX-Nano model...")
        self.model = torch.hub.load('Megvii-BaseDetection/YOLOX', 'yolox_nano', pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        self.input_size = (640, 640)  # Larger input size for better small object detection
        print("YOLOX-Nano model loaded successfully")
    
    def detect_objects(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detects objects in image using YOLOX-Nano
        Returns: List of (x1, y1, x2, y2) tuples
        """
        return self._detect_with_yolox(image)
    
    def _detect_with_yolox(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect objects using YOLOX-Nano"""
        h, w = image.shape[:2]
        
        # Preprocess image
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, self.input_size)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
        
        # Use YOLOX's postprocess function
        # num_classes=80 (COCO), conf_thre and nms_thre, class_agnostic=True for counting
        processed_outputs = postprocess(
            outputs,
            num_classes=80,
            conf_thre=self.confidence_threshold,
            nms_thre=self.nms_threshold,
            class_agnostic=True  # We don't care about class, just objects
        )
        
        bounding_boxes = []
        
        if processed_outputs is not None and len(processed_outputs) > 0:
            result = processed_outputs[0]  # Get first image results
            
            if result is not None and len(result) > 0:
                print(f"  Detected {len(result)} objects after postprocessing")
                
                # Result format: [x1, y1, x2, y2, objectness, class_conf, class_id]
                for det in result:
                    x1, y1, x2, y2 = det[:4]
                    
                    # Scale to original image size
                    x1 = int(x1 * w / self.input_size[0])
                    y1 = int(y1 * h / self.input_size[1])
                    x2 = int(x2 * w / self.input_size[0])
                    y2 = int(y2 * h / self.input_size[1])
                    
                    # Clip to image bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        bounding_boxes.append((x1, y1, x2, y2))
            else:
                print(f"  No objects detected")
        
        return bounding_boxes
    
    def _apply_nms(self, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Apply Non-Maximum Suppression"""
        if len(boxes) == 0:
            return []
        
        boxes_array = np.array(boxes)
        x1 = boxes_array[:, 0]
        y1 = boxes_array[:, 1]
        x2 = boxes_array[:, 2]
        y2 = boxes_array[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = areas.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= self.nms_threshold)[0]
            order = order[inds + 1]
        
        return [boxes[i] for i in keep]


class ObjectClassifier:
    """TinyCLIP for zero-shot classification"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading TinyCLIP model...")
        
        try:
            # Load TinyCLIP model from wkcn
            # TinyCLIP-ViT-8M-16-Text-3M-YFCC15M is the smallest and fastest variant
            model_name = "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.eval()
            print(f"TinyCLIP model loaded successfully: {model_name}")
        except Exception as e:
            print(f"Error loading TinyCLIP: {e}")
            print("Please install: pip install transformers")
            raise
    
    def classify_crop(self, image_crop: np.ndarray, vocabulary: List[str]) -> str:
        """
        Classifies an image crop against vocabulary using TinyCLIP
        Returns: Best matching label
        """
        if len(vocabulary) == 0:
            return "object"
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Create text prompts
        text_prompts = [f"a photo of a {label}" for label in vocabulary]
        
        # Process inputs
        inputs = self.processor(
            text=text_prompts,
            images=pil_image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Calculate features and similarity
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Return best match
        best_idx = probs[0].argmax().item()
        return vocabulary[best_idx]


class AICounter:
    """Main application class implementing the object counting logic"""
    
    def __init__(self):
        self.detector = ObjectDetector()
        self.classifier = ObjectClassifier()
        self.vocabulary = HybridVocabulary()
        self.history = []
        self.history_file = Path("count_history.json")
        self._load_history()
    
    def _load_history(self):
        """Load counting history from file"""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                self.history = data.get('history', [])
                self.vocabulary.recent_items = data.get('recent_items', [])
    
    def _save_history(self):
        """Save counting history to file"""
        data = {
            'history': self.history,
            'recent_items': self.vocabulary.recent_items
        }
        with open(self.history_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def count_objects(self, image_path: str, manual_object: Optional[str] = None) -> Dict:
        """
        Main counting function implementing the "Capture first, Count second" workflow
        
        Args:
            image_path: Path to input image
            manual_object: Optional manual object name for precision mode
        
        Returns:
            Dictionary with counting results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"Processing image: {image_path}")
        
        # Stage 1: Detect all objects (YOLOX-Nano simulation)
        print("Stage 1: Detecting objects...")
        bounding_boxes = self.detector.detect_objects(image)
        print(f"Found {len(bounding_boxes)} potential objects")
        
        if len(bounding_boxes) == 0:
            return {
                'count': 0,
                'name': 'None',
                'image_path': 'N/A',
                'message': 'No objects detected',
                'timestamp': datetime.now().isoformat()
            }
        
        # Stage 2: Classify each detected object (TinyCLIP simulation)
        print("Stage 2: Classifying objects...")
        
        # Determine vocabulary to use
        if manual_object:
            vocabulary = [manual_object.lower()]
            print(f"Using manual precision mode: '{manual_object}'")
        else:
            vocabulary = self.vocabulary.get_vocabulary()
            print(f"Using auto-detection with vocabulary: {vocabulary[:5]}...")
        
        # Classify each bounding box
        detections = {}  # {class_name: [list of boxes]}
        
        for box in bounding_boxes:
            x1, y1, x2, y2 = box
            crop = image[y1:y2, x1:x2]
            
            if crop.size > 0:
                label = self.classifier.classify_crop(crop, vocabulary)
                if label not in detections:
                    detections[label] = []
                detections[label].append(box)
        
        # Stage 3: Apply "Main" Object Algorithm
        print("Stage 3: Determining main object...")
        main_class, main_boxes = self._determine_main_object(detections)
        
        count = len(main_boxes)
        print(f"Main object: '{main_class}' with count: {count}")
        
        # Generate result image with overlay
        output_path = self._burn_overlay(image, main_boxes, main_class, count)
        
        # Create result record
        result = {
            'name': main_class,
            'count': count,
            'image_path': str(output_path),
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def _determine_main_object(self, detections: Dict[str, List]) -> Tuple[str, List]:
        """
        Determines the "main" object using the formula: Score = Area × Count
        
        Args:
            detections: Dictionary mapping class names to list of bounding boxes
        
        Returns:
            Tuple of (main_class_name, list_of_boxes)
        """
        best_score = -1
        best_class = None
        best_boxes = []
        
        for class_name, boxes in detections.items():
            # Calculate total coverage area
            total_area = 0
            for box in boxes:
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                total_area += area
            
            # Calculate score: Area × Count
            count = len(boxes)
            score = total_area * count
            
            if score > best_score:
                best_score = score
                best_class = class_name
                best_boxes = boxes
        
        return best_class or "unknown", best_boxes
    
    def _burn_overlay(self, image: np.ndarray, boxes: List, object_name: str, 
                      count: int) -> Path:
        """
        Burns bounding boxes and count onto image
        
        Returns:
            Path to saved overlay image
        """
        overlay = image.copy()
        
        # Draw bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw center dot
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(overlay, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Draw count text
        text = f"Count: {count} | {object_name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        # Draw background rectangle
        cv2.rectangle(overlay, (10, 10), 
                     (20 + text_width, 20 + text_height + baseline),
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(overlay, text, (15, 15 + text_height),
                   font, font_scale, (255, 255, 255), thickness)
        
        # Save overlay image
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"count_{timestamp}.jpg"
        cv2.imwrite(str(output_path), overlay)
        
        return output_path
    
    def save_result(self, result: Dict) -> None:
        """Saves counting result to history"""
        self.history.insert(0, result)
        self.vocabulary.add_recent_item(result['name'])
        self._save_history()
        print(f"Result saved to history")
    
    def get_history(self) -> List[Dict]:
        """Returns counting history sorted by newest first"""
        return self.history
    
    def print_history(self):
        """Prints formatted history"""
        if not self.history:
            print("No history records found.")
            return
        
        print("\n" + "=" * 70)
        print("COUNTING HISTORY")
        print("=" * 70)
        
        for i, record in enumerate(self.history, 1):
            timestamp = datetime.fromisoformat(record['timestamp'])
            print(f"\n{i}. {record['name'].upper()}")
            print(f"   Count: {record['count']}")
            print(f"   Date: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Image: {record['image_path']}")


def main():
    """CLI interface for AICounter"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AICounter - Object Counting Application",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'command',
        choices=['count', 'history', 'test'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '-i', '--image',
        help='Path to input image (required for count command)'
    )
    
    parser.add_argument(
        '-o', '--object',
        help='Manual object name for precision mode (optional)'
    )
    
    args = parser.parse_args()
    
    counter = AICounter()
    
    if args.command == 'count':
        if not args.image:
            parser.error("count command requires --image argument")
        
        try:
            result = counter.count_objects(args.image, args.object)
            
            print("\n" + "=" * 70)
            print("COUNTING RESULT")
            print("=" * 70)
            print(f"Object: {result['name']}")
            print(f"Count: {result['count']}")
            print(f"Output Image: {result['image_path']}")
            print("=" * 70)
            
            # Ask user if they want to save
            response = input("\nSave this result? (y/n): ").strip().lower()
            if response == 'y':
                counter.save_result(result)
                print("✓ Result saved!")
            
        except Exception as e:
            print(f"Error: {e}")
            return 1
    
    elif args.command == 'history':
        counter.print_history()
    
    elif args.command == 'test':
        print("Generating test image with 5 known objects...")
        test_image, ground_truth = create_ground_truth_image()
        
        # Save the test image
        test_image_path = "test_image.png"
        test_image.save(test_image_path)
        print(f"Test image saved to: {test_image_path}")
        
        # Print ground truth
        print("\n" + "=" * 70)
        print("GROUND TRUTH")
        print("=" * 70)
        for i, obj in enumerate(ground_truth, 1):
            print(f"{i}. {obj['name'].upper()}: bbox={obj['bbox']}")
        print("=" * 70)
        
        # Run counting on the test image
        print("\nRunning object counting on test image...")
        try:
            result = counter.count_objects(test_image_path)
            
            print("\n" + "=" * 70)
            print("TEST RESULT")
            print("=" * 70)
            print(f"Detected Object: {result['name']}")
            print(f"Count: {result['count']}")
            print(f"Output Image: {result['image_path']}")
            print("=" * 70)
            
            # Evaluate accuracy
            print("\nEVALUATION:")
            print(f"Expected: 5 objects of various types")
            print(f"Detected: {result['count']} objects classified as '{result['name']}'")
            
            # Count ground truth objects
            gt_counts = {}
            for obj in ground_truth:
                gt_counts[obj['name']] = gt_counts.get(obj['name'], 0) + 1
            
            print(f"\nGround Truth Distribution:")
            for name, count in gt_counts.items():
                print(f"  - {name}: {count}")
            
        except Exception as e:
            print(f"Error during test: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
