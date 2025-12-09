#!/usr/bin/env python3
"""
Convert YOLOX and TinyCLIP models to Core ML format (.mlmodelc)

This script converts:
1. YOLOX-S PyTorch model (yolox_s.pth) → yolox_s.mlmodel → yolox_s.mlmodelc
2. TinyCLIP Vision encoder (Transformers) → tinyclip_vision.mlmodel → tinyclip_vision.mlmodelc

Usage:
    python3 convert_to_coreml.py
"""

import torch
import coremltools as ct
import numpy as np
import sys
import os
from transformers import CLIPModel, CLIPProcessor

# Add official YOLOX to path
sys.path.insert(0, 'official_yolox')

from yolox.exp import get_exp


def convert_yolox_to_coreml(weight_path='weights/yolox_s.pth', 
                             output_path='weights/yolox_s.mlpackage',
                             input_size=(640, 640)):
    """
    Convert YOLOX PyTorch model to Core ML format.
    
    Args:
        weight_path: Path to YOLOX .pth weights
        output_path: Output path for .mlmodel file
        input_size: Input image size (height, width)
    """
    print("\n" + "="*70)
    print("CONVERTING YOLOX-S TO CORE ML")
    print("="*70)
    
    # Detect model type from weight path
    weight_basename = os.path.basename(weight_path)
    if 'nano' in weight_basename.lower():
        model_name = 'yolox_nano'
        input_size = (416, 416)
    elif 'tiny' in weight_basename.lower():
        model_name = 'yolox_tiny'
        input_size = (416, 416)
    elif 'yolox_s' in weight_basename.lower() or weight_basename == 'yolox_s.pth':
        model_name = 'yolox_s'
        input_size = (640, 640)
    else:
        model_name = 'yolox_s'
        input_size = (640, 640)
    
    print(f"\n📦 Loading {model_name.upper()} model...")
    print(f"  Weight file: {weight_path}")
    print(f"  Input size: {input_size}")
    
    # Load YOLOX model
    exp = get_exp(f'official_yolox/exps/default/{model_name}.py', None)
    model = exp.get_model()
    
    # Load weights
    ckpt = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    print(f"✓ PyTorch model loaded successfully")
    
    # Create dummy input for tracing
    # Core ML expects (batch, channels, height, width) - RGB image normalized to [0, 1]
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    
    print(f"\n🔄 Tracing PyTorch model...")
    print(f"  Input shape: {dummy_input.shape}")
    
    # Trace the model
    traced_model = torch.jit.trace(model, dummy_input)
    
    print(f"✓ Model traced successfully")
    
    # Convert to Core ML
    print(f"\n🔄 Converting to Core ML format...")
    print(f"  This may take a few minutes...")
    
    # Define input
    image_input = ct.ImageType(
        name="image",
        shape=dummy_input.shape,
        scale=1.0/255.0,  # Normalize from [0, 255] to [0, 1]
        bias=[0, 0, 0],
        color_layout=ct.colorlayout.RGB
    )
    
    # Convert with traced model
    mlmodel = ct.convert(
        traced_model,
        inputs=[image_input],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS15
    )
    
    # Add metadata
    mlmodel.author = "YOLOX Object Detection"
    mlmodel.license = "Apache 2.0"
    mlmodel.short_description = f"{model_name.upper()} object detection model"
    mlmodel.version = "1.0"
    
    # Save the model
    print(f"\n💾 Saving Core ML model...")
    print(f"  Output path: {output_path}")
    
    mlmodel.save(output_path)
    
    print(f"✓ Core ML model saved successfully!")
    print(f"  File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    print(f"\n✅ Model ready for deployment!")
    print(f"  Format: ML Program (.mlpackage)")
    print(f"  Note: .mlmodelc will be auto-generated when compiled on iOS device")
    print(f"  Deployment ready model: {output_path}")
    
    return mlmodel, output_path


def convert_tinyclip_to_coreml(model_path='weights/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M',
                                 output_path='weights/tinyclip_vision.mlpackage',
                                 image_size=224):
    """
    Convert TinyCLIP vision encoder to Core ML format.
    
    Args:
        model_path: Path to TinyCLIP model directory
        output_path: Output path for .mlmodel file
        image_size: Input image size (TinyCLIP uses 224x224)
    """
    print("\n" + "="*70)
    print("CONVERTING TINYCLIP VISION ENCODER TO CORE ML")
    print("="*70)
    
    print(f"\n📦 Loading TinyCLIP model...")
    print(f"  Model path: {model_path}")
    
    # Load TinyCLIP model
    model = CLIPModel.from_pretrained(model_path, local_files_only=True)
    processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
    
    # Extract vision model
    vision_model = model.vision_model
    vision_model.eval()
    
    print(f"✓ TinyCLIP vision model loaded successfully")
    print(f"  Parameters: ~8M")
    print(f"  Input size: {image_size}x{image_size}")
    
    # Create wrapper class for Core ML conversion
    class TinyCLIPVisionWrapper(torch.nn.Module):
        def __init__(self, vision_model):
            super().__init__()
            self.vision_model = vision_model
        
        def forward(self, pixel_values):
            # Get vision features
            outputs = self.vision_model(pixel_values=pixel_values)
            # Get pooled output (CLS token)
            pooled_output = outputs.pooler_output
            # Normalize
            normalized = pooled_output / pooled_output.norm(dim=-1, keepdim=True)
            return normalized
    
    wrapped_model = TinyCLIPVisionWrapper(vision_model)
    wrapped_model.eval()
    
    # Create dummy input - TinyCLIP expects normalized images
    # Shape: (batch, channels, height, width)
    dummy_input = torch.randn(1, 3, image_size, image_size)
    
    print(f"\n🔄 Tracing PyTorch model...")
    print(f"  Input shape: {dummy_input.shape}")
    
    # Trace the model
    traced_model = torch.jit.trace(wrapped_model, dummy_input)
    
    print(f"✓ Model traced successfully")
    
    # Convert to Core ML
    print(f"\n🔄 Converting to Core ML format...")
    print(f"  This may take a few minutes...")
    
    # TinyCLIP preprocessing: 
    # 1. Resize to 224x224
    # 2. Normalize with mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
    
    # Define input with preprocessing
    image_input = ct.ImageType(
        name="image",
        shape=dummy_input.shape,
        scale=1.0/255.0,  # Scale from [0, 255] to [0, 1]
        bias=[0, 0, 0],
        color_layout=ct.colorlayout.RGB
    )
    
    # Convert
    mlmodel = ct.convert(
        traced_model,
        inputs=[image_input],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS15
    )
    
    # Add metadata
    mlmodel.author = "TinyCLIP Vision Encoder"
    mlmodel.license = "MIT"
    mlmodel.short_description = "TinyCLIP-ViT-8M-16 vision encoder for visual embeddings"
    mlmodel.version = "1.0"
    
    # Save the model
    print(f"\n💾 Saving Core ML model...")
    print(f"  Output path: {output_path}")
    
    mlmodel.save(output_path)
    
    print(f"✓ Core ML model saved successfully!")
    print(f"  File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    print(f"\n✅ Model ready for deployment!")
    print(f"  Format: ML Program (.mlpackage)")
    print(f"  Note: .mlmodelc will be auto-generated when compiled on iOS device")
    print(f"  Deployment ready model: {output_path}")
    
    return mlmodel, output_path


def main():
    """Main conversion function."""
    print("\n" + "="*70)
    print("MODEL CONVERSION TO CORE ML")
    print("="*70)
    print("\nThis script will convert:")
    print("  1. YOLOX-S PyTorch model → Core ML")
    print("  2. TinyCLIP Vision encoder → Core ML")
    print("\nCore ML models can be used in Swift/iOS applications")
    print("="*70)
    
    # Check dependencies
    print("\n📋 Checking dependencies...")
    try:
        import coremltools
        print(f"  ✓ coremltools version: {coremltools.__version__}")
    except ImportError:
        print("  ❌ coremltools not found!")
        print("\n  Install with: pip install coremltools")
        sys.exit(1)
    
    try:
        import transformers
        print(f"  ✓ transformers version: {transformers.__version__}")
    except ImportError:
        print("  ❌ transformers not found!")
        print("\n  Install with: pip install transformers")
        sys.exit(1)
    
    # Create output directory
    os.makedirs('weights', exist_ok=True)
    
    # Convert YOLOX
    try:
        yolox_model, yolox_path = convert_yolox_to_coreml(
            weight_path='weights/yolox_s.pth',
            output_path='weights/yolox_s.mlpackage'
        )
        print("\n✅ YOLOX conversion completed successfully!")
    except Exception as e:
        print(f"\n❌ YOLOX conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Convert TinyCLIP
    try:
        tinyclip_model, tinyclip_path = convert_tinyclip_to_coreml(
            model_path='weights/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M',
            output_path='weights/tinyclip_vision.mlpackage'
        )
        print("\n✅ TinyCLIP conversion completed successfully!")
    except Exception as e:
        print(f"\n❌ TinyCLIP conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Summary
    print("\n" + "="*70)
    print("CONVERSION SUMMARY")
    print("="*70)
    print("\n✅ All models converted successfully!")
    print(f"\nGenerated files:")
    print(f"  1. {yolox_path}")
    print(f"  2. {tinyclip_path}")
    print(f"\n📱 These models are ready for iOS/Swift deployment!")
    print(f"\nNext steps:")
    print(f"  1. Add .mlpackage files to your Xcode project")
    print(f"  2. Xcode will automatically compile them to .mlmodelc")
    print(f"  3. Use Vision framework for YOLOX inference")
    print(f"  4. Use Core ML directly for TinyCLIP embeddings")
    print("="*70)


if __name__ == '__main__':
    main()
