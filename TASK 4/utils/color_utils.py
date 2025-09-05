"""
Color Utilities for Interactive Image Colorization

This module contains utility functions for color space conversions,
color manipulation, and color palette generation.
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
import colorsys


def rgb_to_hex(rgb_color: Tuple[int, int, int]) -> str:
    """
    Convert RGB color tuple to hex string.
    
    Args:
        rgb_color: RGB color tuple (R, G, B) with values in [0, 255]
        
    Returns:
        Hex color string (e.g., "#FF0000")
    """
    r, g, b = rgb_color
    return f"#{r:02x}{g:02x}{b:02x}".upper()


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert hex color string to RGB tuple.
    
    Args:
        hex_color: Hex color string (e.g., "#FF0000" or "FF0000")
        
    Returns:
        RGB color tuple (R, G, B) with values in [0, 255]
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')
    
    # Convert to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    return (r, g, b)


def rgb_to_lab(rgb_color: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """
    Convert RGB color to LAB color space.
    
    Args:
        rgb_color: RGB color tuple (R, G, B) with values in [0, 255]
        
    Returns:
        LAB color tuple (L, a, b)
    """
    # Normalize RGB to [0, 1]
    r, g, b = [c / 255.0 for c in rgb_color]
    
    # Convert to LAB using OpenCV
    rgb_array = np.array([[[r, g, b]]], dtype=np.float32)
    lab_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
    
    # Extract LAB values
    l, a, b = lab_array[0, 0]
    
    return (l, a, b)


def lab_to_rgb(lab_color: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """
    Convert LAB color to RGB color space.
    
    Args:
        lab_color: LAB color tuple (L, a, b)
        
    Returns:
        RGB color tuple (R, G, B) with values in [0, 255]
    """
    # Convert to RGB using OpenCV
    lab_array = np.array([[[lab_color[0], lab_color[1], lab_color[2]]]], dtype=np.float32)
    rgb_array = cv2.cvtColor(lab_array, cv2.COLOR_LAB2RGB)
    
    # Extract RGB values and convert to [0, 255]
    r, g, b = rgb_array[0, 0]
    r = int(np.clip(r * 255, 0, 255))
    g = int(np.clip(g * 255, 0, 255))
    b = int(np.clip(b * 255, 0, 255))
    
    return (r, g, b)


def rgb_to_hsv(rgb_color: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """
    Convert RGB color to HSV color space.
    
    Args:
        rgb_color: RGB color tuple (R, G, B) with values in [0, 255]
        
    Returns:
        HSV color tuple (H, S, V) with H in [0, 360], S and V in [0, 100]
    """
    r, g, b = [c / 255.0 for c in rgb_color]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    
    # Convert to degrees and percentages
    h = h * 360
    s = s * 100
    v = v * 100
    
    return (h, s, v)


def hsv_to_rgb(hsv_color: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """
    Convert HSV color to RGB color space.
    
    Args:
        hsv_color: HSV color tuple (H, S, V) with H in [0, 360], S and V in [0, 100]
        
    Returns:
        RGB color tuple (R, G, B) with values in [0, 255]
    """
    h, s, v = hsv_color
    
    # Convert to [0, 1] range
    h = h / 360.0
    s = s / 100.0
    v = v / 100.0
    
    # Convert to RGB
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    
    # Convert to [0, 255]
    r = int(np.clip(r * 255, 0, 255))
    g = int(np.clip(g * 255, 0, 255))
    b = int(np.clip(b * 255, 0, 255))
    
    return (r, g, b)


def create_color_palette(palette_type: str = 'rainbow', 
                        num_colors: int = 10) -> List[str]:
    """
    Create a color palette of specified type.
    
    Args:
        palette_type: Type of palette ('rainbow', 'warm', 'cool', 'pastel', 'earth')
        num_colors: Number of colors in the palette
        
    Returns:
        List of hex color strings
    """
    if palette_type == 'rainbow':
        return _create_rainbow_palette(num_colors)
    elif palette_type == 'warm':
        return _create_warm_palette(num_colors)
    elif palette_type == 'cool':
        return _create_cool_palette(num_colors)
    elif palette_type == 'pastel':
        return _create_pastel_palette(num_colors)
    elif palette_type == 'earth':
        return _create_earth_palette(num_colors)
    else:
        raise ValueError(f"Unknown palette type: {palette_type}")


def _create_rainbow_palette(num_colors: int) -> List[str]:
    """Create a rainbow color palette."""
    colors = []
    for i in range(num_colors):
        hue = (i / num_colors) * 360
        rgb = hsv_to_rgb((hue, 100, 100))
        colors.append(rgb_to_hex(rgb))
    return colors


def _create_warm_palette(num_colors: int) -> List[str]:
    """Create a warm color palette (reds, oranges, yellows)."""
    base_colors = [
        (255, 0, 0),      # Red
        (255, 69, 0),     # Red-orange
        (255, 140, 0),    # Dark orange
        (255, 165, 0),    # Orange
        (255, 215, 0),    # Gold
        (255, 255, 0),    # Yellow
    ]
    
    colors = []
    for i in range(num_colors):
        idx = (i / num_colors) * len(base_colors)
        idx1 = int(idx)
        idx2 = min(idx1 + 1, len(base_colors) - 1)
        t = idx - idx1
        
        # Interpolate between colors
        color1 = base_colors[idx1]
        color2 = base_colors[idx2]
        interpolated = tuple(int(c1 + t * (c2 - c1)) for c1, c2 in zip(color1, color2))
        
        colors.append(rgb_to_hex(interpolated))
    
    return colors


def _create_cool_palette(num_colors: int) -> List[str]:
    """Create a cool color palette (blues, greens, cyans)."""
    base_colors = [
        (0, 0, 255),      # Blue
        (0, 128, 255),    # Sky blue
        (0, 255, 255),    # Cyan
        (0, 255, 128),    # Spring green
        (0, 255, 0),      # Green
        (128, 255, 0),    # Lime
    ]
    
    colors = []
    for i in range(num_colors):
        idx = (i / num_colors) * len(base_colors)
        idx1 = int(idx)
        idx2 = min(idx1 + 1, len(base_colors) - 1)
        t = idx - idx1
        
        # Interpolate between colors
        color1 = base_colors[idx1]
        color2 = base_colors[idx2]
        interpolated = tuple(int(c1 + t * (c2 - c1)) for c1, c2 in zip(color1, color2))
        
        colors.append(rgb_to_hex(interpolated))
    
    return colors


def _create_pastel_palette(num_colors: int) -> List[str]:
    """Create a pastel color palette."""
    base_colors = [
        (255, 182, 193),  # Light pink
        (255, 218, 185),  # Peach
        (255, 255, 224),  # Light yellow
        (240, 255, 240),  # Honeydew
        (230, 230, 250),  # Lavender
        (255, 228, 225),  # Misty rose
    ]
    
    colors = []
    for i in range(num_colors):
        idx = (i / num_colors) * len(base_colors)
        idx1 = int(idx)
        idx2 = min(idx1 + 1, len(base_colors) - 1)
        t = idx - idx1
        
        # Interpolate between colors
        color1 = base_colors[idx1]
        color2 = base_colors[idx2]
        interpolated = tuple(int(c1 + t * (c2 - c1)) for c1, c2 in zip(color1, color2))
        
        colors.append(rgb_to_hex(interpolated))
    
    return colors


def _create_earth_palette(num_colors: int) -> List[str]:
    """Create an earth-tone color palette."""
    base_colors = [
        (139, 69, 19),    # Saddle brown
        (160, 82, 45),    # Sienna
        (210, 180, 140),  # Tan
        (244, 164, 96),   # Sandy brown
        (222, 184, 135),  # Burlywood
        (245, 245, 220),  # Beige
    ]
    
    colors = []
    for i in range(num_colors):
        idx = (i / num_colors) * len(base_colors)
        idx1 = int(idx)
        idx2 = min(idx1 + 1, len(base_colors) - 1)
        t = idx - idx1
        
        # Interpolate between colors
        color1 = base_colors[idx1]
        color2 = base_colors[idx2]
        interpolated = tuple(int(c1 + t * (c2 - c1)) for c1, c2 in zip(color1, color2))
        
        colors.append(rgb_to_hex(interpolated))
    
    return colors


def calculate_color_distance(color1: Tuple[int, int, int], 
                           color2: Tuple[int, int, int],
                           metric: str = 'euclidean') -> float:
    """
    Calculate the distance between two colors.
    
    Args:
        color1: First RGB color tuple
        color2: Second RGB color tuple
        metric: Distance metric ('euclidean', 'manhattan', 'lab')
        
    Returns:
        Distance between the colors
    """
    if metric == 'euclidean':
        return np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))
    
    elif metric == 'manhattan':
        return sum(abs(c1 - c2) for c1, c2 in zip(color1, color2))
    
    elif metric == 'lab':
        # Convert to LAB and calculate Euclidean distance
        lab1 = rgb_to_lab(color1)
        lab2 = rgb_to_lab(color2)
        return np.sqrt(sum((l1 - l2) ** 2 for l1, l2 in zip(lab1, lab2)))
    
    else:
        raise ValueError(f"Unknown distance metric: {metric}")


def find_closest_color(target_color: Tuple[int, int, int],
                      color_palette: List[Tuple[int, int, int]],
                      metric: str = 'lab') -> Tuple[int, int, int]:
    """
    Find the closest color in a palette to a target color.
    
    Args:
        target_color: Target RGB color
        color_palette: List of RGB colors to search in
        metric: Distance metric to use
        
    Returns:
        Closest color from the palette
    """
    min_distance = float('inf')
    closest_color = color_palette[0]
    
    for color in color_palette:
        distance = calculate_color_distance(target_color, color, metric)
        if distance < min_distance:
            min_distance = distance
            closest_color = color
    
    return closest_color


def adjust_brightness(color: Tuple[int, int, int], 
                     factor: float) -> Tuple[int, int, int]:
    """
    Adjust the brightness of a color.
    
    Args:
        color: RGB color tuple
        factor: Brightness factor (>1 for brighter, <1 for darker)
        
    Returns:
        Adjusted RGB color tuple
    """
    adjusted = tuple(int(np.clip(c * factor, 0, 255)) for c in color)
    return adjusted


def adjust_saturation(color: Tuple[int, int, int], 
                     factor: float) -> Tuple[int, int, int]:
    """
    Adjust the saturation of a color.
    
    Args:
        color: RGB color tuple
        factor: Saturation factor (>1 for more saturated, <1 for less saturated)
        
    Returns:
        Adjusted RGB color tuple
    """
    # Convert to HSV
    h, s, v = rgb_to_hsv(color)
    
    # Adjust saturation
    s = np.clip(s * factor, 0, 100)
    
    # Convert back to RGB
    return hsv_to_rgb((h, s, v))


def blend_colors(color1: Tuple[int, int, int], 
                color2: Tuple[int, int, int], 
                ratio: float = 0.5) -> Tuple[int, int, int]:
    """
    Blend two colors together.
    
    Args:
        color1: First RGB color tuple
        color2: Second RGB color tuple
        ratio: Blend ratio (0 = all color1, 1 = all color2)
        
    Returns:
        Blended RGB color tuple
    """
    blended = tuple(int(c1 + ratio * (c2 - c1)) for c1, c2 in zip(color1, color2))
    return blended




