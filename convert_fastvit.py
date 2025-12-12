#!/usr/bin/env python3
"""
Convert FastViT vision encoder to Core ML format (single-model script).
Run: python3 convert_fastvit.py
"""
import os
import sys
import torch
import coremltools as ct


def convert_fastvit_to_coreml(model_path='weights/fastvit_s.pth', output_path='weights/fastvit_s.mlpackage', image_size=224, variant='fastvit_s'):
    print("\n" + "="*70)
    print("CONVERTING FASTVIT VISION ENCODER TO CORE ML")
    print("="*70)
    print(f"\n📦 Model path: {model_path}")
    print(f"  Variant: {variant}")
    print(f"  Input size: {image_size}x{image_size}")

    # Use local ml-fastvit-main implementation (do not rely on timm)
    sys.path.insert(0, 'ml-fastvit-main')
    try:
        import models as fastvit_models
        from models.modules.mobileone import reparameterize_model
    except Exception as e:
        raise RuntimeError('ml-fastvit-main not found or missing dependencies. Ensure the ml-fastvit-main folder is present.') from e

    # Attempt to locate a matching variant factory in the provided models
    model_factory = None
    for name in dir(fastvit_models):
        if name == variant or name.startswith(variant):
            model_factory = getattr(fastvit_models, name)
            break
    if model_factory is None:
        raise RuntimeError(f'Variant "{variant}" not found in ml-fastvit-main.models')

    # Instantiate and reparameterize for inference
    model = model_factory()
    print(f"  ✓ Instantiated FastViT variant: {model_factory.__name__}")
    reparameterized_model = reparameterize_model(model)

    # Load checkpoint into reparameterized model if available
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        if 'state_dict' in ckpt:
            state = ckpt['state_dict']
        elif 'model' in ckpt:
            state = ckpt['model']
        else:
            state = ckpt
        try:
            reparameterized_model.load_state_dict(state)
            print("  ✓ Checkpoint loaded into reparameterized model")
        except Exception as e:
            print(f"  ⚠️ Failed to load checkpoint into model: {e}")
            print("  Proceeding with instantiated weights")
    else:
        print("  ⚠️ No local checkpoint found; proceeding with instantiated model weights")

    model = reparameterized_model
    model.eval()

    class FastViTWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            if hasattr(self.model, 'forward_features'):
                feats = self.model.forward_features(x)
                if feats.ndim == 4:
                    pooled = feats.mean(dim=[2,3])
                elif feats.ndim == 3:
                    pooled = feats[:,0,:]
                else:
                    pooled = feats
            else:
                out = self.model(x)
                if out.ndim == 2:
                    pooled = out
                elif out.ndim == 4:
                    pooled = out.mean(dim=[2,3])
                else:
                    pooled = out.mean(dim=1)
            norm = pooled / pooled.norm(dim=-1, keepdim=True)
            return norm

    wrapped = FastViTWrapper(model)
    wrapped.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size)
    traced_model = torch.jit.trace(wrapped, dummy_input)

    image_input = ct.ImageType(
        name="image",
        shape=dummy_input.shape,
        scale=1.0/255.0,
        color_layout=ct.colorlayout.RGB
    )

    mlmodel = ct.convert(
        traced_model,
        inputs=[image_input],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS15
    )

    mlmodel.author = "FastViT Vision Encoder"
    mlmodel.license = "MIT"
    mlmodel.short_description = "FastViT vision encoder producing L2-normalized embeddings"
    mlmodel.version = "1.0"

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    mlmodel.save(output_path)

    print(f"✓ Core ML model saved: {output_path}")
    return mlmodel, output_path


def main():
    print('Running FastViT-only conversion...')
    try:
        convert_fastvit_to_coreml(
            model_path='weights/fastvit_ma36_reparam.pth.tar',
            output_path='weights/fastvit_ma36.mlpackage',
            image_size=256,
            variant='fastvit_ma36'
        )
    except Exception as e:
        print(f"FastViT conversion failed: {e}")


if __name__ == '__main__':
    main()
