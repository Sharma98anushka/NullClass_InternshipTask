"""
Unit Tests for Color Utilities

This module contains unit tests for the color utility functions
in the utils.color_utils module.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.color_utils import (
    rgb_to_hex,
    hex_to_rgb,
    rgb_to_lab,
    lab_to_rgb,
    rgb_to_hsv,
    hsv_to_rgb,
    create_color_palette,
    calculate_color_distance,
    find_closest_color,
    adjust_brightness,
    adjust_saturation,
    blend_colors
)


class TestColorUtils(unittest.TestCase):
    """Test cases for color utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Test colors
        self.test_red = (255, 0, 0)
        self.test_green = (0, 255, 0)
        self.test_blue = (0, 0, 255)
        self.test_white = (255, 255, 255)
        self.test_black = (0, 0, 0)
        
        # Test color palette
        self.test_palette = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
        ]
    
    def test_rgb_to_hex(self):
        """Test converting RGB to hex."""
        # Test red
        self.assertEqual(rgb_to_hex(self.test_red), "#FF0000")
        
        # Test green
        self.assertEqual(rgb_to_hex(self.test_green), "#00FF00")
        
        # Test blue
        self.assertEqual(rgb_to_hex(self.test_blue), "#0000FF")
        
        # Test white
        self.assertEqual(rgb_to_hex(self.test_white), "#FFFFFF")
        
        # Test black
        self.assertEqual(rgb_to_hex(self.test_black), "#000000")
    
    def test_hex_to_rgb(self):
        """Test converting hex to RGB."""
        # Test red
        self.assertEqual(hex_to_rgb("#FF0000"), self.test_red)
        self.assertEqual(hex_to_rgb("FF0000"), self.test_red)
        
        # Test green
        self.assertEqual(hex_to_rgb("#00FF00"), self.test_green)
        
        # Test blue
        self.assertEqual(hex_to_rgb("#0000FF"), self.test_blue)
        
        # Test white
        self.assertEqual(hex_to_rgb("#FFFFFF"), self.test_white)
        
        # Test black
        self.assertEqual(hex_to_rgb("#000000"), self.test_black)
    
    def test_rgb_hex_roundtrip(self):
        """Test roundtrip conversion between RGB and hex."""
        test_colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (128, 64, 32),
            (255, 255, 255),
            (0, 0, 0),
        ]
        
        for rgb_color in test_colors:
            hex_color = rgb_to_hex(rgb_color)
            converted_back = hex_to_rgb(hex_color)
            self.assertEqual(rgb_color, converted_back)
    
    def test_rgb_to_lab(self):
        """Test converting RGB to LAB."""
        # Test red
        lab_red = rgb_to_lab(self.test_red)
        self.assertEqual(len(lab_red), 3)
        self.assertIsInstance(lab_red[0], float)
        self.assertIsInstance(lab_red[1], float)
        self.assertIsInstance(lab_red[2], float)
        
        # Test green
        lab_green = rgb_to_lab(self.test_green)
        self.assertEqual(len(lab_green), 3)
        
        # Test blue
        lab_blue = rgb_to_lab(self.test_blue)
        self.assertEqual(len(lab_blue), 3)
    
    def test_lab_to_rgb(self):
        """Test converting LAB to RGB."""
        # Test red
        lab_red = rgb_to_lab(self.test_red)
        rgb_back = lab_to_rgb(lab_red)
        self.assertEqual(len(rgb_back), 3)
        self.assertIsInstance(rgb_back[0], int)
        self.assertIsInstance(rgb_back[1], int)
        self.assertIsInstance(rgb_back[2], int)
        
        # Check that values are in valid range
        for component in rgb_back:
            self.assertGreaterEqual(component, 0)
            self.assertLessEqual(component, 255)
    
    def test_rgb_lab_roundtrip(self):
        """Test roundtrip conversion between RGB and LAB."""
        test_colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (128, 64, 32),
        ]
        
        for rgb_color in test_colors:
            lab_color = rgb_to_lab(rgb_color)
            converted_back = lab_to_rgb(lab_color)
            
            # Due to color space conversion precision, we check approximate equality
            for i in range(3):
                self.assertAlmostEqual(rgb_color[i], converted_back[i], delta=5)
    
    def test_rgb_to_hsv(self):
        """Test converting RGB to HSV."""
        # Test red
        hsv_red = rgb_to_hsv(self.test_red)
        self.assertEqual(len(hsv_red), 3)
        self.assertIsInstance(hsv_red[0], float)  # Hue
        self.assertIsInstance(hsv_red[1], float)  # Saturation
        self.assertIsInstance(hsv_red[2], float)  # Value
        
        # Check ranges
        self.assertGreaterEqual(hsv_red[0], 0)    # Hue: [0, 360]
        self.assertLessEqual(hsv_red[0], 360)
        self.assertGreaterEqual(hsv_red[1], 0)    # Saturation: [0, 100]
        self.assertLessEqual(hsv_red[1], 100)
        self.assertGreaterEqual(hsv_red[2], 0)    # Value: [0, 100]
        self.assertLessEqual(hsv_red[2], 100)
    
    def test_hsv_to_rgb(self):
        """Test converting HSV to RGB."""
        # Test red (H=0, S=100, V=100)
        hsv_red = (0, 100, 100)
        rgb_back = hsv_to_rgb(hsv_red)
        self.assertEqual(len(rgb_back), 3)
        self.assertIsInstance(rgb_back[0], int)
        self.assertIsInstance(rgb_back[1], int)
        self.assertIsInstance(rgb_back[2], int)
        
        # Check that values are in valid range
        for component in rgb_back:
            self.assertGreaterEqual(component, 0)
            self.assertLessEqual(component, 255)
    
    def test_rgb_hsv_roundtrip(self):
        """Test roundtrip conversion between RGB and HSV."""
        test_colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (128, 64, 32),
        ]
        
        for rgb_color in test_colors:
            hsv_color = rgb_to_hsv(rgb_color)
            converted_back = hsv_to_rgb(hsv_color)
            
            # Due to color space conversion precision, we check approximate equality
            for i in range(3):
                self.assertAlmostEqual(rgb_color[i], converted_back[i], delta=5)
    
    def test_create_color_palette_rainbow(self):
        """Test creating rainbow color palette."""
        palette = create_color_palette('rainbow', 5)
        
        # Check number of colors
        self.assertEqual(len(palette), 5)
        
        # Check that all are valid hex colors
        for color in palette:
            self.assertTrue(color.startswith('#'))
            self.assertEqual(len(color), 7)  # #RRGGBB
    
    def test_create_color_palette_warm(self):
        """Test creating warm color palette."""
        palette = create_color_palette('warm', 6)
        
        # Check number of colors
        self.assertEqual(len(palette), 6)
        
        # Check that all are valid hex colors
        for color in palette:
            self.assertTrue(color.startswith('#'))
            self.assertEqual(len(color), 7)
    
    def test_create_color_palette_cool(self):
        """Test creating cool color palette."""
        palette = create_color_palette('cool', 6)
        
        # Check number of colors
        self.assertEqual(len(palette), 6)
        
        # Check that all are valid hex colors
        for color in palette:
            self.assertTrue(color.startswith('#'))
            self.assertEqual(len(color), 7)
    
    def test_create_color_palette_pastel(self):
        """Test creating pastel color palette."""
        palette = create_color_palette('pastel', 6)
        
        # Check number of colors
        self.assertEqual(len(palette), 6)
        
        # Check that all are valid hex colors
        for color in palette:
            self.assertTrue(color.startswith('#'))
            self.assertEqual(len(color), 7)
    
    def test_create_color_palette_earth(self):
        """Test creating earth color palette."""
        palette = create_color_palette('earth', 6)
        
        # Check number of colors
        self.assertEqual(len(palette), 6)
        
        # Check that all are valid hex colors
        for color in palette:
            self.assertTrue(color.startswith('#'))
            self.assertEqual(len(color), 7)
    
    def test_create_color_palette_invalid_type(self):
        """Test creating color palette with invalid type."""
        with self.assertRaises(ValueError):
            create_color_palette('invalid_type', 5)
    
    def test_calculate_color_distance_euclidean(self):
        """Test calculating Euclidean color distance."""
        distance = calculate_color_distance(self.test_red, self.test_green, 'euclidean')
        
        self.assertIsInstance(distance, float)
        self.assertGreater(distance, 0)
        
        # Distance from red to green should be significant
        self.assertGreater(distance, 100)
    
    def test_calculate_color_distance_manhattan(self):
        """Test calculating Manhattan color distance."""
        distance = calculate_color_distance(self.test_red, self.test_green, 'manhattan')
        
        self.assertIsInstance(distance, float)
        self.assertGreater(distance, 0)
    
    def test_calculate_color_distance_lab(self):
        """Test calculating LAB color distance."""
        distance = calculate_color_distance(self.test_red, self.test_green, 'lab')
        
        self.assertIsInstance(distance, float)
        self.assertGreater(distance, 0)
    
    def test_calculate_color_distance_invalid_metric(self):
        """Test calculating color distance with invalid metric."""
        with self.assertRaises(ValueError):
            calculate_color_distance(self.test_red, self.test_green, 'invalid_metric')
    
    def test_find_closest_color(self):
        """Test finding closest color in palette."""
        target_color = (255, 100, 100)  # Light red
        closest = find_closest_color(target_color, self.test_palette)
        
        # Should return one of the palette colors
        self.assertIn(closest, self.test_palette)
        
        # Should be red (closest to light red)
        self.assertEqual(closest, self.test_red)
    
    def test_find_closest_color_empty_palette(self):
        """Test finding closest color in empty palette."""
        target_color = (255, 0, 0)
        empty_palette = []
        
        with self.assertRaises(IndexError):
            find_closest_color(target_color, empty_palette)
    
    def test_adjust_brightness_brighter(self):
        """Test adjusting brightness to make color brighter."""
        # Make red brighter
        brighter = adjust_brightness(self.test_red, 1.5)
        
        # Should still be red (R component should be max)
        self.assertEqual(brighter[0], 255)
        self.assertEqual(brighter[1], 0)
        self.assertEqual(brighter[2], 0)
    
    def test_adjust_brightness_darker(self):
        """Test adjusting brightness to make color darker."""
        # Make red darker
        darker = adjust_brightness(self.test_red, 0.5)
        
        # Should be darker red
        self.assertEqual(darker[0], 128)  # 255 * 0.5
        self.assertEqual(darker[1], 0)
        self.assertEqual(darker[2], 0)
    
    def test_adjust_brightness_zero_factor(self):
        """Test adjusting brightness with zero factor."""
        # Make color black
        black = adjust_brightness(self.test_red, 0.0)
        
        self.assertEqual(black, (0, 0, 0))
    
    def test_adjust_saturation_more_saturated(self):
        """Test adjusting saturation to make color more saturated."""
        # Make a grayish red more saturated
        grayish_red = (200, 100, 100)
        more_saturated = adjust_saturation(grayish_red, 2.0)
        
        # Should be more saturated (more difference between components)
        original_diff = abs(grayish_red[0] - grayish_red[1])
        saturated_diff = abs(more_saturated[0] - more_saturated[1])
        self.assertGreater(saturated_diff, original_diff)
    
    def test_adjust_saturation_less_saturated(self):
        """Test adjusting saturation to make color less saturated."""
        # Make red less saturated
        less_saturated = adjust_saturation(self.test_red, 0.5)
        
        # Should be less saturated (more gray)
        self.assertLess(less_saturated[0], 255)
        self.assertGreater(less_saturated[1], 0)
        self.assertGreater(less_saturated[2], 0)
    
    def test_blend_colors_equal_ratio(self):
        """Test blending colors with equal ratio."""
        # Blend red and blue equally
        blended = blend_colors(self.test_red, self.test_blue, 0.5)
        
        # Should be purple (red + blue)
        self.assertEqual(blended[0], 128)  # (255 + 0) / 2
        self.assertEqual(blended[1], 0)    # (0 + 0) / 2
        self.assertEqual(blended[2], 128)  # (0 + 255) / 2
    
    def test_blend_colors_all_first_color(self):
        """Test blending colors with ratio 0 (all first color)."""
        blended = blend_colors(self.test_red, self.test_blue, 0.0)
        
        # Should be all red
        self.assertEqual(blended, self.test_red)
    
    def test_blend_colors_all_second_color(self):
        """Test blending colors with ratio 1 (all second color)."""
        blended = blend_colors(self.test_red, self.test_blue, 1.0)
        
        # Should be all blue
        self.assertEqual(blended, self.test_blue)
    
    def test_blend_colors_default_ratio(self):
        """Test blending colors with default ratio (0.5)."""
        blended = blend_colors(self.test_red, self.test_blue)
        
        # Should be equal blend
        self.assertEqual(blended[0], 128)
        self.assertEqual(blended[1], 0)
        self.assertEqual(blended[2], 128)


if __name__ == '__main__':
    unittest.main()




