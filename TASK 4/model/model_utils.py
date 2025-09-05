"""
Model Utilities for Interactive Image Colorization

This module contains utility functions for model management, including
loading, saving, and preprocessing operations.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import pickle

from .colorization_model import ColorizationModel, UserGuidedColorizer


def load_model(model_path: str, device: str = 'cpu') -> UserGuidedColorizer:
    """
    Load a pre-trained colorization model.
    
    Args:
        model_path: Path to the model checkpoint file
        device: Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        Loaded UserGuidedColorizer instance
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # Initialize colorizer
        colorizer = UserGuidedColorizer(device=device)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            colorizer.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume the checkpoint is just the state dict
            colorizer.model.load_state_dict(checkpoint)
        
        print(f"✓ Model loaded successfully from {model_path}")
        return colorizer
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")


def save_model(colorizer: UserGuidedColorizer, 
               save_path: str, 
               metadata: Optional[Dict[str, Any]] = None):
    """
    Save a colorization model to disk.
    
    Args:
        colorizer: UserGuidedColorizer instance to save
        save_path: Path where to save the model
        metadata: Optional metadata to save with the model
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint
    checkpoint = {
        'model_state_dict': colorizer.model.state_dict(),
        'model_config': {
            'input_channels': colorizer.model.input_channels,
            'output_channels': colorizer.model.output_channels,
        },
        'device': str(colorizer.device),
        'metadata': metadata or {}
    }
    
    # Save checkpoint
    torch.save(checkpoint, save_path)
    print(f"✓ Model saved to {save_path}")


def preprocess_image(image: np.ndarray, 
                    target_size: Optional[Tuple[int, int]] = None,
                    normalize: bool = True) -> np.ndarray:
    """
    Preprocess image for colorization model input.
    
    Args:
        image: Input image as numpy array
        target_size: Optional target size (width, height) for resizing
        normalize: Whether to normalize pixel values to [0, 1]
        
    Returns:
        Preprocessed image array
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Resize if target size is specified
    if target_size:
        gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize if requested
    if normalize:
        gray = gray.astype(np.float32) / 255.0
    
    return gray


def postprocess_image(image: np.ndarray, 
                     denormalize: bool = True) -> np.ndarray:
    """
    Postprocess colorized image for display.
    
    Args:
        image: Colorized image array (values in [0, 1])
        denormalize: Whether to denormalize to [0, 255] range
        
    Returns:
        Postprocessed image array ready for display
    """
    # Clip values to valid range
    image = np.clip(image, 0, 1)
    
    # Denormalize if requested
    if denormalize:
        image = (image * 255).astype(np.uint8)
    
    return image


def create_sample_data(image_size: Tuple[int, int] = (256, 256)) -> Dict[str, np.ndarray]:
    """
    Create sample data for testing the model.
    
    Args:
        image_size: Size of the sample image (height, width)
        
    Returns:
        Dictionary containing sample grayscale image and test regions
    """
    # Create a simple test image
    height, width = image_size
    test_image = np.random.rand(height, width).astype(np.float32)
    
    # Add some structure to make it more interesting
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    test_image += 0.5 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    
    # Normalize to [0, 1]
    test_image = (test_image - test_image.min()) / (test_image.max() - test_image.min())
    
    # Create sample regions
    regions = [
        {
            'bbox': [50, 50, 100, 100],  # Rectangle region
            'color': (255, 0, 0)  # Red
        },
        {
            'bbox': [150, 150, 200, 200],  # Another rectangle
            'color': (0, 255, 0)  # Green
        }
    ]
    
    return {
        'image': test_image,
        'regions': regions
    }


def validate_model_input(image: np.ndarray) -> bool:
    """
    Validate that an image is suitable for the colorization model.
    
    Args:
        image: Image array to validate
        
    Returns:
        True if image is valid, False otherwise
    """
    # Check if image is 2D (grayscale) or 3D with 3 channels (RGB)
    if len(image.shape) == 2:
        return True
    elif len(image.shape) == 3 and image.shape[2] == 3:
        return True
    else:
        return False


def get_model_info(colorizer: UserGuidedColorizer) -> Dict[str, Any]:
    """
    Get information about a loaded model.
    
    Args:
        colorizer: UserGuidedColorizer instance
        
    Returns:
        Dictionary containing model information
    """
    model = colorizer.model
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'model_type': 'ColorizationModel',
        'input_channels': model.input_channels,
        'output_channels': model.output_channels,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'device': str(colorizer.device),
        'model_state': 'eval' if model.training == False else 'training'
    }
    
    return info


def export_model_for_inference(colorizer: UserGuidedColorizer, 
                              export_path: str,
                              input_shape: Tuple[int, int, int] = (1, 4, 256, 256)):
    """
    Export model for inference (e.g., to ONNX format).
    
    Args:
        colorizer: UserGuidedColorizer instance
        export_path: Path to save the exported model
        input_shape: Expected input shape (batch_size, channels, height, width)
    """
    try:
        import onnx
        import onnxruntime
    except ImportError:
        print("⚠️  ONNX not available. Install with: pip install onnx onnxruntime")
        return
    
    model = colorizer.model
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape, device=colorizer.device)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    
    print(f"✓ Model exported to ONNX format: {export_path}")


def benchmark_model(colorizer: UserGuidedColorizer, 
                   image_size: Tuple[int, int] = (256, 256),
                   num_runs: int = 10) -> Dict[str, float]:
    """
    Benchmark model performance.
    
    Args:
        colorizer: UserGuidedColorizer instance
        image_size: Size of test image (height, width)
        num_runs: Number of runs for averaging
        
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    # Create test data
    test_data = create_sample_data(image_size)
    image = test_data['image']
    regions = test_data['regions']
    colors = [region['color'] for region in regions]
    
    # Warm up
    for _ in range(3):
        _ = colorizer.colorize_image(image, regions, colors)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = colorizer.colorize_image(image, regions, colors)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    results = {
        'average_inference_time': avg_time,
        'std_inference_time': std_time,
        'fps': 1.0 / avg_time,
        'image_size': image_size,
        'num_runs': num_runs
    }
    
    return results




