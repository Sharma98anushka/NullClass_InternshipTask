"""
GUI Components for Interactive Image Colorization

This module contains reusable Streamlit components for the colorization interface,
including image upload, region selection, color picking, and preview functionality.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw
import io
from typing import List, Tuple, Dict, Any, Optional
import json


class ImageUploader:
    """Component for handling image upload and processing."""
    
    def __init__(self, accepted_types: List[str] = None, max_size: int = 512):
        """
        Initialize the image uploader.
        
        Args:
            accepted_types: List of accepted file extensions
            max_size: Maximum image size for processing
        """
        self.accepted_types = accepted_types or ['png', 'jpg', 'jpeg', 'bmp', 'tiff']
        self.max_size = max_size
    
    def upload_image(self, key: str = "image_uploader") -> Optional[np.ndarray]:
        """
        Create an image upload widget.
        
        Args:
            key: Unique key for the uploader widget
            
        Returns:
            Processed image array or None if no image uploaded
        """
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=self.accepted_types,
            key=key,
            help=f"Upload an image (max size: {self.max_size}x{self.max_size})"
        )
        
        if uploaded_file is not None:
            return self._process_uploaded_image(uploaded_file)
        
        return None
    
    def _process_uploaded_image(self, uploaded_file) -> np.ndarray:
        """
        Process the uploaded image file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Processed image array
        """
        # Read image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            # Convert RGB to grayscale
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # Resize if too large
        if gray.shape[0] > self.max_size or gray.shape[1] > self.max_size:
            scale = min(self.max_size / gray.shape[0], self.max_size / gray.shape[1])
            new_size = (int(gray.shape[1] * scale), int(gray.shape[0] * scale))
            gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
        
        return gray


class RegionSelector:
    """Component for interactive region selection on images."""
    
    def __init__(self, tools: List[str] = None):
        """
        Initialize the region selector.
        
        Args:
            tools: List of available selection tools
        """
        self.tools = tools or ["Brush", "Rectangle", "Circle"]
        self.current_tool = "Brush"
        self.brush_size = 10
    
    def create_selection_interface(self, 
                                 image: np.ndarray, 
                                 key_prefix: str = "region_selector") -> Dict[str, Any]:
        """
        Create the region selection interface.
        
        Args:
            image: Image to select regions on
            key_prefix: Prefix for widget keys
            
        Returns:
            Dictionary containing selection data
        """
        # Tool selection
        self.current_tool = st.selectbox(
            "Selection Tool",
            self.tools,
            key=f"{key_prefix}_tool"
        )
        
        # Tool-specific parameters
        if self.current_tool == "Brush":
            self.brush_size = st.slider(
                "Brush Size",
                min_value=1,
                max_value=50,
                value=10,
                key=f"{key_prefix}_brush_size"
            )
        
        # Create canvas for selection
        canvas_data = self._create_canvas(image, key_prefix)
        
        return {
            'tool': self.current_tool,
            'brush_size': self.brush_size,
            'canvas_data': canvas_data
        }
    
    def _create_canvas(self, image: np.ndarray, key_prefix: str) -> Dict[str, Any]:
        """
        Create a canvas for region selection.
        
        Args:
            image: Image to display on canvas
            key_prefix: Prefix for widget keys
            
        Returns:
            Canvas data dictionary
        """
        # For demonstration, we'll create a simple interface
        # In a real implementation, this would use streamlit-drawable-canvas
        
        # Display the image
        st.image(image, caption="Click and drag to select regions", use_column_width=True)
        
        # Create selection controls
        col1, col2 = st.columns(2)
        
        with col1:
            start_x = st.number_input("Start X", 0, image.shape[1]-1, 0, key=f"{key_prefix}_start_x")
            start_y = st.number_input("Start Y", 0, image.shape[0]-1, 0, key=f"{key_prefix}_start_y")
        
        with col2:
            end_x = st.number_input("End X", 0, image.shape[1]-1, min(start_x+50, image.shape[1]-1), key=f"{key_prefix}_end_x")
            end_y = st.number_input("End Y", 0, image.shape[0]-1, min(start_y+50, image.shape[0]-1), key=f"{key_prefix}_end_y")
        
        # Return selection data
        return {
            'tool': self.current_tool,
            'start_point': (start_x, start_y),
            'end_point': (end_x, end_y),
            'bbox': [min(start_x, end_x), min(start_y, end_y), 
                    max(start_x, end_x), max(start_y, end_y)]
        }


class ColorPicker:
    """Component for color selection and management."""
    
    def __init__(self, default_color: str = "#FF0000"):
        """
        Initialize the color picker.
        
        Args:
            default_color: Default color in hex format
        """
        self.default_color = default_color
        self.color_history = []
    
    def create_color_picker(self, key: str = "color_picker") -> str:
        """
        Create a color picker widget.
        
        Args:
            key: Unique key for the color picker
            
        Returns:
            Selected color in hex format
        """
        selected_color = st.color_picker(
            "Choose Color",
            value=self.default_color,
            key=key,
            help="Select a color for the current region"
        )
        
        # Add to history
        if selected_color not in self.color_history:
            self.color_history.append(selected_color)
            if len(self.color_history) > 10:  # Keep last 10 colors
                self.color_history.pop(0)
        
        return selected_color
    
    def create_color_palette(self, colors: List[str] = None, key: str = "color_palette") -> str:
        """
        Create a color palette widget.
        
        Args:
            colors: List of predefined colors
            key: Unique key for the palette
            
        Returns:
            Selected color in hex format
        """
        if colors is None:
            colors = [
                "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF",
                "#00FFFF", "#FFA500", "#800080", "#008000", "#FFC0CB"
            ]
        
        st.subheader("Color Palette")
        
        # Create color buttons in a grid
        cols = st.columns(5)
        selected_color = self.default_color
        
        for i, color in enumerate(colors):
            col_idx = i % 5
            with cols[col_idx]:
                if st.button(
                    "⬜",  # White square placeholder
                    key=f"{key}_{i}",
                    help=f"Select {color}"
                ):
                    selected_color = color
        
        # Show color preview
        if selected_color:
            color_preview = np.zeros((50, 50, 3), dtype=np.uint8)
            rgb_color = self._hex_to_rgb(selected_color)
            color_preview[:, :] = rgb_color
            st.image(color_preview, caption=f"Selected: {selected_color}", width=100)
        
        return selected_color
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def get_color_history(self) -> List[str]:
        """Get the color history."""
        return self.color_history.copy()


class PreviewWindow:
    """Component for displaying image previews and results."""
    
    def __init__(self, max_width: int = 400):
        """
        Initialize the preview window.
        
        Args:
            max_width: Maximum width for preview images
        """
        self.max_width = max_width
    
    def display_image(self, 
                     image: np.ndarray, 
                     caption: str = "Image", 
                     key: str = "preview") -> None:
        """
        Display an image in the preview window.
        
        Args:
            image: Image array to display
            caption: Caption for the image
            key: Unique key for the display
        """
        # Ensure image is in the right format
        if len(image.shape) == 2:
            # Grayscale image
            display_image = image
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # RGB image
            if image.dtype == np.float32 or image.dtype == np.float64:
                # Normalize to [0, 255]
                display_image = (image * 255).astype(np.uint8)
            else:
                display_image = image
        else:
            st.error(f"Unsupported image format: {image.shape}")
            return
        
        # Display the image
        st.image(display_image, caption=caption, use_column_width=True)
    
    def display_comparison(self, 
                          original: np.ndarray, 
                          processed: np.ndarray,
                          original_caption: str = "Original",
                          processed_caption: str = "Processed") -> None:
        """
        Display a side-by-side comparison of two images.
        
        Args:
            original: Original image
            processed: Processed image
            original_caption: Caption for original image
            processed_caption: Caption for processed image
        """
        col1, col2 = st.columns(2)
        
        with col1:
            self.display_image(original, original_caption)
        
        with col2:
            self.display_image(processed, processed_caption)
    
    def display_regions(self, 
                       image: np.ndarray, 
                       regions: List[Dict[str, Any]],
                       colors: List[Tuple[int, int, int]]) -> None:
        """
        Display image with selected regions highlighted.
        
        Args:
            image: Base image
            regions: List of region dictionaries
            colors: List of colors for each region
        """
        # Create a copy of the image for drawing
        if len(image.shape) == 2:
            # Convert grayscale to RGB for drawing
            display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            display_image = image.copy()
        
        # Draw regions
        for region, color in zip(regions, colors):
            if 'bbox' in region:
                x1, y1, x2, y2 = region['bbox']
                cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
                
                # Add color label
                cv2.putText(display_image, f"RGB{color}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Display the result
        self.display_image(display_image, "Image with Selected Regions")
    
    def create_download_button(self, 
                              image: np.ndarray, 
                              filename: str = "image.png",
                              key: str = "download") -> None:
        """
        Create a download button for an image.
        
        Args:
            image: Image to download
            filename: Name of the file to download
            key: Unique key for the button
        """
        # Convert image to PIL format
        if len(image.shape) == 2:
            pil_image = Image.fromarray(image.astype(np.uint8))
        else:
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
        
        # Create download button
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        st.download_button(
            label="📥 Download Image",
            data=img_buffer.getvalue(),
            file_name=filename,
            mime="image/png",
            key=key
        )


class ImageProcessor:
    """Utility class for image processing operations."""
    
    @staticmethod
    def resize_image(image: np.ndarray, max_size: int) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image: Input image
            max_size: Maximum dimension size
            
        Returns:
            Resized image
        """
        height, width = image.shape[:2]
        
        if height <= max_size and width <= max_size:
            return image
        
        # Calculate scale factor
        scale = min(max_size / height, max_size / width)
        new_size = (int(width * scale), int(height * scale))
        
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range.
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        return image
    
    @staticmethod
    def denormalize_image(image: np.ndarray) -> np.ndarray:
        """
        Denormalize image to [0, 255] range.
        
        Args:
            image: Input image (values in [0, 1])
            
        Returns:
            Denormalized image
        """
        if image.dtype == np.float32 or image.dtype == np.float64:
            return (image * 255).astype(np.uint8)
        return image




