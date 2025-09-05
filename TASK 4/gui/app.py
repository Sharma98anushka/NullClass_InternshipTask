"""
Main Streamlit Application for Interactive Image Colorization

This module contains the main Streamlit app that provides a web interface
for user-guided image colorization with interactive tools.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import sys
from pathlib import Path
from streamlit_drawable_canvas import st_canvas

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from model.colorization_model import UserGuidedColorizer
from utils.image_processing import convert_to_grayscale, resize_image
from utils.color_utils import rgb_to_hex, hex_to_rgb
from gui.components import ImageUploader, RegionSelector, ColorPicker, PreviewWindow


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'grayscale_image' not in st.session_state:
        st.session_state.grayscale_image = None
    if 'selected_regions' not in st.session_state:
        st.session_state.selected_regions = []
    if 'region_colors' not in st.session_state:
        st.session_state.region_colors = []
    if 'colorized_image' not in st.session_state:
        st.session_state.colorized_image = None
    if 'colorizer' not in st.session_state:
        st.session_state.colorizer = None


def setup_page_config():
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="Interactive Image Colorization",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def create_sidebar():
    """Create the sidebar with controls and information."""
    st.sidebar.title("🎨 Colorization Controls")
    
    # Model information
    st.sidebar.subheader("Model Info")
    if st.session_state.colorizer:
        model_info = {
            "Device": str(st.session_state.colorizer.device),
            "Status": "Ready" if st.session_state.colorizer.model.training == False else "Training"
        }
        for key, value in model_info.items():
            st.sidebar.text(f"{key}: {value}")
    else:
        st.sidebar.text("Model: Not loaded")
    
    # Color picker
    st.sidebar.subheader("Color Selection")
    selected_color = st.sidebar.color_picker(
        "Choose Color", 
        value="#FF0000",
        help="Select a color for the current region"
    )
    
    # Tool selection
    st.sidebar.subheader("Selection Tool")
    tool = st.sidebar.selectbox(
        "Select Tool",
        ["Brush", "Rectangle", "Circle"],
        help="Choose the selection tool for marking regions"
    )
    
    # Brush size (for brush tool)
    if tool == "Brush":
        brush_size = st.sidebar.slider(
            "Brush Size", 
            min_value=2, 
            max_value=60, 
            value=12,
            help="Adjust brush size for freehand selection"
        )
    else:
        brush_size = 12
    
    # Canvas options
    st.sidebar.subheader("Canvas")
    realtime_update = st.sidebar.checkbox("Realtime update", True)

    # Enhancement controls
    st.sidebar.subheader("Enhancement")
    saturation_boost = st.sidebar.slider(
        "Saturation boost", min_value=0.5, max_value=2.5, value=1.4, step=0.1,
        help="Multiply a/b chroma to increase color vividness"
    )
    blend_with_gray = st.sidebar.slider(
        "Blend with original", min_value=0.0, max_value=1.0, value=0.15, step=0.05,
        help="0 = fully colorized, 1 = closer to original grayscale"
    )
    guidance_strength = st.sidebar.slider(
        "Guidance strength", min_value=0.0, max_value=1.0, value=1.0, step=0.05,
        help="Weight applied to user color hints"
    )
    # Persist in session
    st.session_state['saturation_boost'] = saturation_boost
    st.session_state['blend_with_gray'] = blend_with_gray
    st.session_state['guidance_strength'] = guidance_strength
    
    # Action buttons
    st.sidebar.subheader("Actions")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🔄 Reset", help="Clear all selections and start over"):
            st.session_state.selected_regions = []
            st.session_state.region_colors = []
            st.session_state.colorized_image = None
            st.rerun()
    
    with col2:
        if st.button("💾 Save", help="Save the colorized image"):
            if st.session_state.colorized_image is not None:
                save_colorized_image()
    
    # Always expose download when available
    if st.session_state.get('colorized_image') is not None:
        st.sidebar.markdown("---")
        save_colorized_image()
    
    return selected_color, tool, brush_size, realtime_update, saturation_boost, blend_with_gray, guidance_strength


def save_colorized_image():
    """Save the colorized image to disk."""
    if st.session_state.colorized_image is not None:
        # Convert to PIL Image
        img_array = (np.clip(st.session_state.colorized_image, 0, 1) * 255).astype(np.uint8)
        if img_array.ndim == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        pil_image = Image.fromarray(img_array)
        
        # Create download button
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        st.sidebar.download_button(
            label="📥 Download Colorized Image",
            data=img_buffer.getvalue(),
            file_name="colorized_image.png",
            mime="image/png"
        )


def _to_rgb_uint8(gray: np.ndarray) -> np.ndarray:
    """Ensure grayscale image is 3-channel RGB uint8 for canvas background."""
    if len(gray.shape) == 2:
        rgb = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        rgb = gray
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
    return rgb


def _extract_region_from_canvas(canvas_result, tool: str, img_h: int, img_w: int, selected_color_hex: str):
    """Extract a region dict and color tuple from canvas result based on tool."""
    rgb_color = hex_to_rgb(selected_color_hex)
    if canvas_result is None:
        return None, None
    
    # Brush: build a binary mask from drawn layer alpha
    if tool == "Brush" and canvas_result.image_data is not None:
        overlay = np.array(canvas_result.image_data)  # H x W x 4 RGBA
        if overlay.ndim == 3 and overlay.shape[2] == 4:
            alpha = overlay[:, :, 3]
            mask = (alpha > 0).astype(np.float32)
            if mask.sum() == 0:
                return None, None
            return {'mask': mask}, rgb_color
        return None, None
    
    # Rectangle / Circle: use last object bbox
    if canvas_result.json_data is not None and "objects" in canvas_result.json_data:
        objects = canvas_result.json_data["objects"]
        if len(objects) == 0:
            return None, None
        obj = objects[-1]
        left = int(round(obj.get('left', 0)))
        top = int(round(obj.get('top', 0)))
        width = int(round(obj.get('width', 0)))
        height = int(round(obj.get('height', 0)))
        x1 = max(0, left)
        y1 = max(0, top)
        x2 = min(img_w - 1, left + width)
        y2 = min(img_h - 1, top + height)
        if x2 <= x1 or y2 <= y1:
            return None, None
        return {'bbox': [x1, y1, x2, y2]}, rgb_color
    
    return None, None


def main():
    """Main application function."""
    # Setup
    setup_page_config()
    initialize_session_state()
    
    # Header
    st.title("🎨 Interactive User-Guided Image Colorization")
    st.markdown("""
    Upload a grayscale image, select regions, choose colors, and watch the AI colorize your image in real-time!
    """)
    
    # Initialize colorizer if not already done
    if st.session_state.colorizer is None:
        try:
            st.session_state.colorizer = UserGuidedColorizer()
            st.success("✅ AI model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to load AI model: {e}")
            st.info("Using placeholder model for demonstration")
    
    # Sidebar controls
    selected_color, tool, brush_size, realtime_update, saturation_boost, blend_with_gray, guidance_strength = create_sidebar()
    # Read enhancements from sidebar widgets (they are defined inside create_sidebar now? Actually defined above.)
    # Fetch from session_state if present, else defaults
    sat = st.session_state.get('saturation_boost', None)
    blend = st.session_state.get('blend_with_gray', None)
    guide = st.session_state.get('guidance_strength', None)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📸 Image Upload & Selection")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a grayscale or color image to colorize"
        )
        
        if uploaded_file is not None:
            # Process uploaded image
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            # Convert to grayscale if needed
            if len(image_array.shape) == 3:
                grayscale = convert_to_grayscale(image_array)
            else:
                grayscale = image_array
            
            # Resize if too large
            max_size = 768
            if grayscale.shape[0] > max_size or grayscale.shape[1] > max_size:
                grayscale = resize_image(grayscale, max_size)
            
            st.session_state.grayscale_image = grayscale
            
            # Display original image
            st.image(grayscale, caption="Original Grayscale Image")
            
            # Region selection interface
            st.subheader("🖱️ Region Selection")
            
            # Prepare background image for canvas
            bg_rgb = _to_rgb_uint8(grayscale)
            bg_pil = Image.fromarray(bg_rgb)
            
            drawing_mode = {
                "Brush": "freedraw",
                "Rectangle": "rect",
                "Circle": "circle"
            }[tool]
            
            canvas_result = None
            canvas_ok = True
            try:
                canvas_result = st_canvas(
                    fill_color=selected_color + "20",  # Transparent fill
                    stroke_width=brush_size,
                    stroke_color=selected_color,
                    background_color="#00000000",
                    background_image=bg_pil,
                    update_streamlit=realtime_update,
                    height=bg_rgb.shape[0],
                    width=bg_rgb.shape[1],
                    drawing_mode=drawing_mode,
                    key="colorization_canvas",
                )
            except Exception as canvas_err:
                canvas_ok = False
                st.error(
                    "Interactive canvas failed to load. Please align your environment and restart: \n"
                    "1) Activate venv: source /Users/apple/Downloads/Project_Null1/.venv/bin/activate\n"
                    "2) pip install --upgrade 'numpy==1.26.4' 'pyarrow==16.1.0' streamlit-drawable-canvas\n"
                    "3) Restart: python /Users/apple/Downloads/Project_Null1/main.py"
                )
            
            st.caption("Draw on the canvas, then click 'Add Region' to record the selection.")
            
            # Region selection buttons
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                if st.button("➕ Add Region", help="Add the current selection as a colored region"):
                    if canvas_ok:
                        region, color = _extract_region_from_canvas(
                            canvas_result,
                            tool,
                            img_h=grayscale.shape[0],
                            img_w=grayscale.shape[1],
                            selected_color_hex=selected_color,
                        )
                        if region is not None:
                            st.session_state.selected_regions.append(region)
                            st.session_state.region_colors.append(color)
                            st.success("✅ Region added")
                        else:
                            st.warning("No selection detected. Draw on the canvas first.")
                    else:
                        st.info("Canvas disabled. Please check the error message above for instructions.")
            
            with col_b:
                if st.button("🗑️ Clear Last", help="Remove the last added region"):
                    if st.session_state.selected_regions:
                        st.session_state.selected_regions.pop()
                        st.session_state.region_colors.pop()
                        st.success("✅ Removed last region")
                    else:
                        st.info("No regions to remove.")
            
            with col_c:
                if st.button("🎨 Colorize", help="Apply colorization to the image"):
                    if st.session_state.selected_regions and st.session_state.colorizer:
                        # Keep colors length consistent
                        if len(st.session_state.region_colors) != len(st.session_state.selected_regions):
                            st.session_state.region_colors = st.session_state.region_colors[:len(st.session_state.selected_regions)]
                        try:
                            colorized = st.session_state.colorizer.colorize_image(
                                st.session_state.grayscale_image,
                                st.session_state.selected_regions,
                                st.session_state.region_colors,
                                saturation_boost=saturation_boost,
                                blend_with_gray=blend_with_gray,
                                guidance_strength=guidance_strength,
                            )
                            st.session_state.colorized_image = colorized
                            st.success("✅ Image colorized successfully!")
                        except Exception as e:
                            st.error(f"❌ Colorization failed: {e}")
                    else:
                        st.info("Add at least one region before colorizing.")
    
    with col2:
        st.subheader("🎯 Current Selections")
        
        # Display current regions
        if st.session_state.selected_regions:
            for i, (region, color) in enumerate(zip(st.session_state.selected_regions, st.session_state.region_colors)):
                with st.expander(f"Region {i+1}"):
                    if 'bbox' in region:
                        st.write(f"Type: Rectangle/Circle")
                        st.write(f"Bounds: {region['bbox']}")
                    elif 'mask' in region:
                        st.write("Type: Brush mask")
                        st.write(f"Mask pixels: {int(np.sum(region['mask'] > 0))}")
                    st.write(f"Color: {color}")
        else:
            st.info("No regions selected yet. Use the tools on the left to select areas.")
        
        # Preview section
        st.subheader("🖼️ Preview")
        
        if st.session_state.colorized_image is not None:
            st.image(
                st.session_state.colorized_image, 
                caption="Colorized Result"
            )
            # Main-pane download
            img_array = (np.clip(st.session_state.colorized_image, 0, 1) * 255).astype(np.uint8)
            if img_array.ndim == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            pil_image = Image.fromarray(img_array)
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            st.download_button(
                label="📥 Download Colorized Image",
                data=img_buffer.getvalue(),
                file_name="colorized_image.png",
                mime="image/png",
                key="download_main"
            )
        else:
            st.info("Colorize the image to see the preview here.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using PyTorch, Streamlit, and OpenCV</p>
        <p>Interactive User-Guided Image Colorization</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()


