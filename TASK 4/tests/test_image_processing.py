"""
Unit Tests for Image Processing Utilities

This module contains unit tests for the image processing functions
in the utils.image_processing module.
"""

import unittest
import numpy as np
import cv2
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.image_processing import (
    convert_to_grayscale,
    resize_image,
    normalize_image,
    denormalize_image,
    apply_color_mask,
    create_region_mask,
    enhance_contrast,
    remove_noise,
    detect_edges
)


class TestImageProcessing(unittest.TestCase):
    """Test cases for image processing utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test images
        self.test_grayscale = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        self.test_rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self.test_rgba = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        
        # Create test mask
        self.test_mask = np.zeros((100, 100), dtype=np.uint8)
        self.test_mask[25:75, 25:75] = 255
        
        # Test color
        self.test_color = (255, 0, 0)  # Red
    
    def test_convert_to_grayscale_rgb(self):
        """Test converting RGB image to grayscale."""
        result = convert_to_grayscale(self.test_rgb)
        
        # Check output shape
        self.assertEqual(result.shape, (100, 100))
        
        # Check data type
        self.assertEqual(result.dtype, np.uint8)
        
        # Check that result is different from input (not just copying)
        self.assertFalse(np.array_equal(result, self.test_rgb[:, :, 0]))
    
    def test_convert_to_grayscale_already_grayscale(self):
        """Test converting already grayscale image."""
        result = convert_to_grayscale(self.test_grayscale)
        
        # Should return the same image
        np.testing.assert_array_equal(result, self.test_grayscale)
    
    def test_convert_to_grayscale_rgba(self):
        """Test converting RGBA image to grayscale."""
        result = convert_to_grayscale(self.test_rgba)
        
        # Check output shape
        self.assertEqual(result.shape, (100, 100))
        
        # Check data type
        self.assertEqual(result.dtype, np.uint8)
    
    def test_convert_to_grayscale_invalid_channels(self):
        """Test converting image with invalid number of channels."""
        invalid_image = np.random.randint(0, 256, (100, 100, 5), dtype=np.uint8)
        
        with self.assertRaises(ValueError):
            convert_to_grayscale(invalid_image)
    
    def test_resize_image_max_dimension(self):
        """Test resizing image to maximum dimension."""
        result = resize_image(self.test_grayscale, 50)
        
        # Check that maximum dimension is 50
        max_dim = max(result.shape)
        self.assertLessEqual(max_dim, 50)
        
        # Check that aspect ratio is maintained
        original_ratio = self.test_grayscale.shape[1] / self.test_grayscale.shape[0]
        result_ratio = result.shape[1] / result.shape[0]
        np.testing.assert_almost_equal(original_ratio, result_ratio, decimal=2)
    
    def test_resize_image_exact_size(self):
        """Test resizing image to exact dimensions."""
        target_size = (80, 60)
        result = resize_image(self.test_grayscale, target_size)
        
        # Check output size
        self.assertEqual(result.shape, (60, 80))
    
    def test_resize_image_no_change_needed(self):
        """Test resizing when no change is needed."""
        result = resize_image(self.test_grayscale, 150)
        
        # Should return the same image
        np.testing.assert_array_equal(result, self.test_grayscale)
    
    def test_normalize_image_uint8(self):
        """Test normalizing uint8 image."""
        result = normalize_image(self.test_grayscale)
        
        # Check data type
        self.assertEqual(result.dtype, np.float32)
        
        # Check value range
        self.assertGreaterEqual(result.min(), 0.0)
        self.assertLessEqual(result.max(), 1.0)
    
    def test_normalize_image_custom_range(self):
        """Test normalizing image to custom range."""
        result = normalize_image(self.test_grayscale, target_range=(-1.0, 1.0))
        
        # Check value range
        self.assertGreaterEqual(result.min(), -1.0)
        self.assertLessEqual(result.max(), 1.0)
    
    def test_denormalize_image(self):
        """Test denormalizing image."""
        # First normalize
        normalized = normalize_image(self.test_grayscale)
        
        # Then denormalize
        result = denormalize_image(normalized)
        
        # Check data type
        self.assertEqual(result.dtype, np.uint8)
        
        # Check value range
        self.assertGreaterEqual(result.min(), 0)
        self.assertLessEqual(result.max(), 255)
    
    def test_apply_color_mask_grayscale(self):
        """Test applying color mask to grayscale image."""
        result = apply_color_mask(self.test_grayscale, self.test_mask, self.test_color)
        
        # Check output shape
        self.assertEqual(result.shape, (100, 100, 3))
        
        # Check data type
        self.assertEqual(result.dtype, np.uint8)
        
        # Check that masked areas have the specified color
        masked_pixels = result[self.test_mask > 0]
        # Should have some red component in masked areas
        self.assertTrue(np.any(masked_pixels[:, 0] > 0))
    
    def test_apply_color_mask_rgb(self):
        """Test applying color mask to RGB image."""
        result = apply_color_mask(self.test_rgb, self.test_mask, self.test_color)
        
        # Check output shape
        self.assertEqual(result.shape, (100, 100, 3))
        
        # Check data type
        self.assertEqual(result.dtype, np.uint8)
    
    def test_create_region_mask_binary(self):
        """Test creating binary region mask."""
        regions = [
            {'bbox': [10, 10, 30, 30]},
            {'bbox': [50, 50, 70, 70]}
        ]
        
        result = create_region_mask((100, 100), regions, mask_type='binary')
        
        # Check output shape
        self.assertEqual(result.shape, (100, 100))
        
        # Check that regions are marked
        self.assertTrue(np.any(result > 0))
    
    def test_create_region_mask_indexed(self):
        """Test creating indexed region mask."""
        regions = [
            {'bbox': [10, 10, 30, 30]},
            {'bbox': [50, 50, 70, 70]}
        ]
        
        result = create_region_mask((100, 100), regions, mask_type='indexed')
        
        # Check output shape
        self.assertEqual(result.shape, (100, 100))
        
        # Check that regions have different indices
        unique_values = np.unique(result)
        self.assertIn(1, unique_values)
        self.assertIn(2, unique_values)
    
    def test_enhance_contrast_histogram_equalization(self):
        """Test histogram equalization contrast enhancement."""
        result = enhance_contrast(self.test_grayscale, method='histogram_equalization')
        
        # Check output shape
        self.assertEqual(result.shape, self.test_grayscale.shape)
        
        # Check data type
        self.assertEqual(result.dtype, np.uint8)
    
    def test_enhance_contrast_clahe(self):
        """Test CLAHE contrast enhancement."""
        result = enhance_contrast(self.test_grayscale, method='clahe')
        
        # Check output shape
        self.assertEqual(result.shape, self.test_grayscale.shape)
        
        # Check data type
        self.assertEqual(result.dtype, np.uint8)
    
    def test_enhance_contrast_gamma(self):
        """Test gamma correction contrast enhancement."""
        result = enhance_contrast(self.test_grayscale, method='gamma')
        
        # Check output shape
        self.assertEqual(result.shape, self.test_grayscale.shape)
        
        # Check data type
        self.assertEqual(result.dtype, np.uint8)
    
    def test_enhance_contrast_invalid_method(self):
        """Test contrast enhancement with invalid method."""
        with self.assertRaises(ValueError):
            enhance_contrast(self.test_grayscale, method='invalid_method')
    
    def test_remove_noise_gaussian(self):
        """Test Gaussian noise removal."""
        result = remove_noise(self.test_grayscale, method='gaussian')
        
        # Check output shape
        self.assertEqual(result.shape, self.test_grayscale.shape)
        
        # Check data type
        self.assertEqual(result.dtype, np.uint8)
    
    def test_remove_noise_median(self):
        """Test median noise removal."""
        result = remove_noise(self.test_grayscale, method='median')
        
        # Check output shape
        self.assertEqual(result.shape, self.test_grayscale.shape)
        
        # Check data type
        self.assertEqual(result.dtype, np.uint8)
    
    def test_remove_noise_bilateral(self):
        """Test bilateral noise removal."""
        result = remove_noise(self.test_grayscale, method='bilateral')
        
        # Check output shape
        self.assertEqual(result.shape, self.test_grayscale.shape)
        
        # Check data type
        self.assertEqual(result.dtype, np.uint8)
    
    def test_remove_noise_invalid_method(self):
        """Test noise removal with invalid method."""
        with self.assertRaises(ValueError):
            remove_noise(self.test_grayscale, method='invalid_method')
    
    def test_detect_edges_canny(self):
        """Test Canny edge detection."""
        result = detect_edges(self.test_grayscale, method='canny')
        
        # Check output shape
        self.assertEqual(result.shape, self.test_grayscale.shape)
        
        # Check data type
        self.assertEqual(result.dtype, np.uint8)
    
    def test_detect_edges_sobel(self):
        """Test Sobel edge detection."""
        result = detect_edges(self.test_grayscale, method='sobel')
        
        # Check output shape
        self.assertEqual(result.shape, self.test_grayscale.shape)
        
        # Check data type
        self.assertEqual(result.dtype, np.uint8)
    
    def test_detect_edges_laplacian(self):
        """Test Laplacian edge detection."""
        result = detect_edges(self.test_grayscale, method='laplacian')
        
        # Check output shape
        self.assertEqual(result.shape, self.test_grayscale.shape)
        
        # Check data type
        self.assertEqual(result.dtype, np.uint8)
    
    def test_detect_edges_invalid_method(self):
        """Test edge detection with invalid method."""
        with self.assertRaises(ValueError):
            detect_edges(self.test_grayscale, method='invalid_method')


if __name__ == '__main__':
    unittest.main()




