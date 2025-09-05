"""
Image Processing Utilities for Interactive Image Colorization

This module contains utility functions for image processing operations
including conversion, resizing, normalization, and color manipulation.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Union
from PIL import Image


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale.
    
    Args:
        image: Input image array (RGB or grayscale)
        
    Returns:
        Grayscale image array
        
    Raises:
        ValueError: If image has unsupported number of channels
    """
    if len(image.shape) == 2:
        # Already grayscale
        return image
    elif len(image.shape) == 3:
        if image.shape[2] == 3:
            # RGB to grayscale
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:
            # RGBA to grayscale
            return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


def resize_image(image: np.ndarray, 
                target_size: Union[int, Tuple[int, int]], 
                maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize an image to the target size.
    
    Args:
        image: Input image array
        target_size: Target size as integer (max dimension) or tuple (width, height)
        maintain_aspect: Whether to maintain aspect ratio when target_size is int
        
    Returns:
        Resized image array
    """
    if isinstance(target_size, int):
        # Resize to maximum dimension while maintaining aspect ratio
        height, width = image.shape[:2]
        
        if height <= target_size and width <= target_size:
            return image
        
        # Calculate scale factor
        scale = min(target_size / height, target_size / width)
        new_size = (int(width * scale), int(height * scale))
        
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    else:
        # Resize to exact dimensions
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray, 
                   target_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """
    Normalize image values to the target range.
    
    Args:
        image: Input image array
        target_range: Target range as (min, max)
        
    Returns:
        Normalized image array
    """
    if image.dtype == np.uint8:
        # Convert from [0, 255] to target range
        min_val, max_val = target_range
        normalized = image.astype(np.float32) / 255.0
        return normalized * (max_val - min_val) + min_val
    else:
        # Assume already in [0, 1] range
        return image.astype(np.float32)


def denormalize_image(image: np.ndarray, 
                     source_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """
    Denormalize image values from the source range to [0, 255].
    
    Args:
        image: Input image array (normalized)
        source_range: Source range as (min, max)
        
    Returns:
        Denormalized image array in [0, 255] range
    """
    min_val, max_val = source_range
    
    # Clip to source range
    clipped = np.clip(image, min_val, max_val)
    
    # Normalize to [0, 1]
    normalized = (clipped - min_val) / (max_val - min_val)
    
    # Convert to [0, 255]
    return (normalized * 255).astype(np.uint8)


def apply_color_mask(image: np.ndarray, 
                    mask: np.ndarray, 
                    color: Tuple[int, int, int],
                    alpha: float = 0.5) -> np.ndarray:
    """
    Apply a color mask to an image.
    
    Args:
        image: Input image array
        mask: Binary mask array (same size as image)
        color: RGB color tuple
        alpha: Transparency of the color overlay
        
    Returns:
        Image with color mask applied
    """
    # Ensure image is RGB
    if len(image.shape) == 2:
        # Convert grayscale to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()
    
    # Create color overlay
    color_overlay = np.zeros_like(image_rgb)
    color_overlay[mask > 0] = color
    
    # Blend images
    result = cv2.addWeighted(image_rgb, 1 - alpha, color_overlay, alpha, 0)
    
    return result


def create_region_mask(image_shape: Tuple[int, int], 
                      regions: list,
                      mask_type: str = 'binary') -> np.ndarray:
    """
    Create a mask from region definitions.
    
    Args:
        image_shape: Shape of the image (height, width)
        regions: List of region dictionaries with 'bbox' or 'mask' keys
        mask_type: Type of mask ('binary' or 'indexed')
        
    Returns:
        Mask array
    """
    height, width = image_shape
    
    if mask_type == 'binary':
        mask = np.zeros((height, width), dtype=np.uint8)
    else:  # indexed
        mask = np.zeros((height, width), dtype=np.int32)
    
    for i, region in enumerate(regions):
        if 'bbox' in region:
            x1, y1, x2, y2 = region['bbox']
            # Ensure coordinates are within bounds
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            
            if mask_type == 'binary':
                mask[y1:y2+1, x1:x2+1] = 255
            else:
                mask[y1:y2+1, x1:x2+1] = i + 1
                
        elif 'mask' in region:
            region_mask = region['mask']
            if region_mask.shape == (height, width):
                if mask_type == 'binary':
                    mask = np.logical_or(mask, region_mask).astype(np.uint8) * 255
                else:
                    mask[region_mask > 0] = i + 1
    
    return mask


def enhance_contrast(image: np.ndarray, 
                    method: str = 'histogram_equalization') -> np.ndarray:
    """
    Enhance image contrast using various methods.
    
    Args:
        image: Input image array
        method: Enhancement method ('histogram_equalization', 'clahe', 'gamma')
        
    Returns:
        Enhanced image array
    """
    if method == 'histogram_equalization':
        if len(image.shape) == 2:
            return cv2.equalizeHist(image)
        else:
            # Apply to each channel separately
            enhanced = np.zeros_like(image)
            for i in range(image.shape[2]):
                enhanced[:, :, i] = cv2.equalizeHist(image[:, :, i])
            return enhanced
    
    elif method == 'clahe':
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        if len(image.shape) == 2:
            return clahe.apply(image)
        else:
            # Apply to each channel separately
            enhanced = np.zeros_like(image)
            for i in range(image.shape[2]):
                enhanced[:, :, i] = clahe.apply(image[:, :, i])
            return enhanced
    
    elif method == 'gamma':
        # Gamma correction
        gamma = 1.2
        look_up_table = np.empty((1, 256), np.uint8)
        for i in range(256):
            look_up_table[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        
        return cv2.LUT(image, look_up_table)
    
    else:
        raise ValueError(f"Unknown enhancement method: {method}")


def remove_noise(image: np.ndarray, 
                method: str = 'gaussian',
                kernel_size: int = 3) -> np.ndarray:
    """
    Remove noise from an image.
    
    Args:
        image: Input image array
        method: Denoising method ('gaussian', 'median', 'bilateral')
        kernel_size: Size of the filter kernel
        
    Returns:
        Denoised image array
    """
    if method == 'gaussian':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    elif method == 'median':
        return cv2.medianBlur(image, kernel_size)
    
    elif method == 'bilateral':
        return cv2.bilateralFilter(image, kernel_size, 75, 75)
    
    else:
        raise ValueError(f"Unknown denoising method: {method}")


def detect_edges(image: np.ndarray, 
                method: str = 'canny',
                low_threshold: int = 50,
                high_threshold: int = 150) -> np.ndarray:
    """
    Detect edges in an image.
    
    Args:
        image: Input image array
        method: Edge detection method ('canny', 'sobel', 'laplacian')
        low_threshold: Lower threshold for Canny edge detection
        high_threshold: Upper threshold for Canny edge detection
        
    Returns:
        Edge map array
    """
    if method == 'canny':
        return cv2.Canny(image, low_threshold, high_threshold)
    
    elif method == 'sobel':
        # Sobel edge detection
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return (magnitude * 255 / magnitude.max()).astype(np.uint8)
    
    elif method == 'laplacian':
        return cv2.Laplacian(image, cv2.CV_64F).astype(np.uint8)
    
    else:
        raise ValueError(f"Unknown edge detection method: {method}")


def save_image(image: np.ndarray, 
               filepath: str, 
               quality: int = 95) -> None:
    """
    Save an image to disk.
    
    Args:
        image: Image array to save
        filepath: Path where to save the image
        quality: JPEG quality (1-100) for JPEG files
    """
    # Convert to PIL Image
    if len(image.shape) == 2:
        pil_image = Image.fromarray(image)
    else:
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)
    
    # Save with appropriate format
    if filepath.lower().endswith('.jpg') or filepath.lower().endswith('.jpeg'):
        pil_image.save(filepath, 'JPEG', quality=quality)
    else:
        pil_image.save(filepath)


def load_image(filepath: str, 
               target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load an image from disk.
    
    Args:
        filepath: Path to the image file
        target_size: Optional target size for resizing
        
    Returns:
        Loaded image array
    """
    # Load with PIL
    pil_image = Image.open(filepath)
    
    # Convert to numpy array
    image = np.array(pil_image)
    
    # Resize if requested
    if target_size:
        image = resize_image(image, target_size)
    
    return image




