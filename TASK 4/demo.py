#!/usr/bin/env python3
"""
Demo Script for Interactive Image Colorization

This script demonstrates how to use the colorization model with sample data.
"""

import numpy as np
import cv2
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from model import UserGuidedColorizer
from utils.image_processing import convert_to_grayscale, resize_image
from utils.color_utils import rgb_to_hex, hex_to_rgb


def create_sample_image(size=(256, 256)):
    """Create a sample grayscale image for demonstration."""
    # Create a simple test image with some structure
    height, width = size
    image = np.zeros((height, width), dtype=np.float32)
    
    # Add some geometric shapes
    # Circle
    center = (width // 4, height // 4)
    radius = min(width, height) // 8
    y, x = np.ogrid[:height, :width]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    image[mask] = 0.8
    
    # Rectangle
    rect_x1, rect_y1 = width // 2, height // 2
    rect_x2, rect_y2 = 3 * width // 4, 3 * height // 4
    image[rect_y1:rect_y2, rect_x1:rect_x2] = 0.6
    
    # Add some noise
    noise = np.random.normal(0, 0.1, image.shape)
    image = np.clip(image + noise, 0, 1)
    
    return image


def main():
    """Main demo function."""
    print("🎨 Interactive Image Colorization Demo")
    print("=" * 50)
    
    # Create sample image
    print("📸 Creating sample grayscale image...")
    sample_image = create_sample_image()
    
    # Initialize colorizer
    print("🤖 Initializing AI colorization model...")
    try:
        colorizer = UserGuidedColorizer()
        print("✅ Model initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        print("Using placeholder model for demonstration")
        return
    
    # Define sample regions and colors
    print("🎯 Setting up sample regions and colors...")
    regions = [
        {
            'bbox': [50, 50, 100, 100],  # Circle region
            'color': (255, 0, 0)  # Red
        },
        {
            'bbox': [128, 128, 192, 192],  # Rectangle region
            'color': (0, 255, 0)  # Green
        }
    ]
    
    colors = [region['color'] for region in regions]
    
    # Perform colorization
    print("🎨 Performing colorization...")
    try:
        colorized_image = colorizer.colorize_image(sample_image, regions, colors)
        print("✅ Colorization completed successfully!")
    except Exception as e:
        print(f"❌ Colorization failed: {e}")
        return
    
    # Display results
    print("\n📊 Results:")
    print(f"   Original image shape: {sample_image.shape}")
    print(f"   Colorized image shape: {colorized_image.shape}")
    print(f"   Number of regions: {len(regions)}")
    
    # Show region information
    for i, (region, color) in enumerate(zip(regions, colors)):
        print(f"   Region {i+1}: {region['bbox']} -> RGB{color} ({rgb_to_hex(color)})")
    
    # Save results
    print("\n💾 Saving results...")
    try:
        # Convert to uint8 for saving
        original_uint8 = (sample_image * 255).astype(np.uint8)
        colorized_uint8 = (colorized_image * 255).astype(np.uint8)
        
        # Save images
        cv2.imwrite('demo_original.png', original_uint8)
        cv2.imwrite('demo_colorized.png', colorized_uint8)
        
        print("✅ Images saved as 'demo_original.png' and 'demo_colorized.png'")
    except Exception as e:
        print(f"❌ Failed to save images: {e}")
    
    print("\n🎉 Demo completed successfully!")
    print("💡 To run the full interactive application, use: python main.py")


if __name__ == '__main__':
    main()




