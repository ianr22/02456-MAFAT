"""
Test script to verify exact dtype pipeline between original and new implementation
Run this to compare preprocessing outputs
"""

import numpy as np
import torch
from PIL import Image
from skimage.transform import resize
from pathlib import Path

def original_preprocessing(image_path, picturesize=224):
    """Exact preprocessing from original notebook"""
    print("="*60)
    print("ORIGINAL PREPROCESSING")
    print("="*60)
    
    # Load image
    image = np.asarray(Image.open(image_path))
    print(f"1. After np.asarray(Image.open()): dtype={image.dtype}, shape={image.shape}")
    print(f"   min={image.min()}, max={image.max()}")
    
    # Copy and normalize
    img = image
    img_normalized = np.copy(img)
    print(f"2. After np.copy(): dtype={img_normalized.dtype}")
    
    img_normalized = img_normalized / 255.
    print(f"3. After / 255.: dtype={img_normalized.dtype}")
    print(f"   min={img_normalized.min():.6f}, max={img_normalized.max():.6f}")
    
    # Resize
    img_resized = resize(
        img_normalized,
        output_shape=(picturesize, picturesize),
        mode='reflect',
        anti_aliasing=True
    )[:, :, :3]
    print(f"4. After resize: dtype={img_resized.dtype}, shape={img_resized.shape}")
    print(f"   min={img_resized.min():.6f}, max={img_resized.max():.6f}")
    
    # Convert to tensor
    tensor = torch.from_numpy(img_resized)
    print(f"5. After torch.from_numpy(): dtype={tensor.dtype}")
    
    # Permute
    tensor_permuted = tensor.permute(2, 0, 1)
    print(f"6. After permute(2,0,1): dtype={tensor_permuted.dtype}, shape={tensor_permuted.shape}")
    
    # Convert to float on device
    tensor_float = tensor_permuted.to('cpu', dtype=torch.float)
    print(f"7. After .to(device, dtype=torch.float): dtype={tensor_float.dtype}")
    print(f"   min={tensor_float.min():.6f}, max={tensor_float.max():.6f}")
    
    return tensor_float


def new_preprocessing(image_path, picturesize=224):
    """New preprocessing from updated code"""
    print("\n" + "="*60)
    print("NEW PREPROCESSING")
    print("="*60)
    
    # Load image
    image = np.asarray(Image.open(image_path))
    print(f"1. After np.asarray(Image.open()): dtype={image.dtype}, shape={image.shape}")
    print(f"   min={image.min()}, max={image.max()}")
    
    # Copy and normalize
    img_normalized = np.copy(image)
    print(f"2. After np.copy(): dtype={img_normalized.dtype}")
    
    img_normalized = img_normalized / 255.0
    print(f"3. After / 255.0: dtype={img_normalized.dtype}")
    print(f"   min={img_normalized.min():.6f}, max={img_normalized.max():.6f}")
    
    # Resize
    img_resized = resize(
        img_normalized,
        output_shape=(picturesize, picturesize),
        mode='reflect',
        anti_aliasing=True
    )[:, :, :3]
    print(f"4. After resize: dtype={img_resized.dtype}, shape={img_resized.shape}")
    print(f"   min={img_resized.min():.6f}, max={img_resized.max():.6f}")
    
    # Convert to tensor with .float()
    tensor = torch.from_numpy(img_resized).float()
    print(f"5. After torch.from_numpy().float(): dtype={tensor.dtype}")
    
    # Permute
    tensor_permuted = tensor.permute(2, 0, 1)
    print(f"6. After permute(2,0,1): dtype={tensor_permuted.dtype}, shape={tensor_permuted.shape}")
    
    # Convert to float on device
    tensor_float = tensor_permuted.to('cpu', dtype=torch.float)
    print(f"7. After .to(device, dtype=torch.float): dtype={tensor_float.dtype}")
    print(f"   min={tensor_float.min():.6f}, max={tensor_float.max():.6f}")
    
    return tensor_float


def compare_preprocessing(image_path):
    """Compare both preprocessing pipelines"""
    original = original_preprocessing(image_path)
    new = new_preprocessing(image_path)
    
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    # Check if they're identical
    diff = torch.abs(original - new)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Maximum difference: {max_diff:.10f}")
    print(f"Mean difference: {mean_diff:.10f}")
    print(f"Are they identical? {torch.allclose(original, new, atol=1e-10)}")
    
    if max_diff > 1e-6:
        print(f"\n⚠️  WARNING: Significant difference detected!")
        print(f"This could explain performance degradation.")
    else:
        print(f"\n✓ Preprocessing pipelines are functionally identical")
    
    return original, new


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_preprocessing_dtypes.py <path_to_test_image>")
        print("\nExample: python test_preprocessing_dtypes.py ./dataset_v2/root/train/cropped/1_1.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    compare_preprocessing(image_path)