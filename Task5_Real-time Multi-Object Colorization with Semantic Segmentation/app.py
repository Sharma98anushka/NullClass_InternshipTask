import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from PIL import Image
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import threading
import queue

# Page configuration
st.set_page_config(
    page_title="Real-time Video Colorization",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .color-picker-container {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Color mapping for different object classes
DEFAULT_COLORS = {
    'person': [255, 0, 0],      # Red
    'bicycle': [255, 165, 0],   # Orange
    'car': [255, 0, 0],         # Red
    'motorcycle': [255, 165, 0], # Orange
    'airplane': [0, 0, 255],    # Blue
    'bus': [255, 0, 0],         # Red
    'train': [128, 0, 128],     # Purple
    'truck': [255, 0, 0],       # Red
    'boat': [0, 0, 255],        # Blue
    'traffic light': [255, 255, 0], # Yellow
    'fire hydrant': [255, 0, 0], # Red
    'stop sign': [255, 0, 0],   # Red
    'parking meter': [128, 128, 128], # Gray
    'bench': [139, 69, 19],     # Brown
    'bird': [0, 255, 0],        # Green
    'cat': [255, 165, 0],       # Orange
    'dog': [255, 165, 0],       # Orange
    'horse': [139, 69, 19],     # Brown
    'sheep': [255, 255, 255],   # White
    'cow': [139, 69, 19],       # Brown
    'elephant': [128, 128, 128], # Gray
    'bear': [139, 69, 19],      # Brown
    'zebra': [255, 255, 255],   # White
    'giraffe': [255, 165, 0],   # Orange
    'backpack': [128, 0, 128],  # Purple
    'umbrella': [128, 0, 128],  # Purple
    'handbag': [128, 0, 128],   # Purple
    'tie': [128, 0, 128],       # Purple
    'suitcase': [128, 0, 128],  # Purple
    'frisbee': [255, 255, 0],   # Yellow
    'skis': [0, 255, 255],      # Cyan
    'snowboard': [0, 255, 255], # Cyan
    'sports ball': [255, 255, 0], # Yellow
    'kite': [255, 255, 0],      # Yellow
    'baseball bat': [139, 69, 19], # Brown
    'baseball glove': [139, 69, 19], # Brown
    'skateboard': [128, 0, 128], # Purple
    'surfboard': [0, 255, 255], # Cyan
    'tennis racket': [255, 255, 0], # Yellow
    'bottle': [0, 255, 0],      # Green
    'wine glass': [0, 255, 0],  # Green
    'cup': [0, 255, 0],         # Green
    'fork': [128, 128, 128],    # Gray
    'knife': [128, 128, 128],   # Gray
    'spoon': [128, 128, 128],   # Gray
    'bowl': [0, 255, 0],        # Green
    'banana': [255, 255, 0],    # Yellow
    'apple': [255, 0, 0],       # Red
    'sandwich': [255, 165, 0],  # Orange
    'orange': [255, 165, 0],    # Orange
    'broccoli': [0, 255, 0],    # Green
    'carrot': [255, 165, 0],    # Orange
    'hot dog': [255, 165, 0],   # Orange
    'pizza': [255, 0, 0],       # Red
    'donut': [255, 255, 0],     # Yellow
    'cake': [255, 255, 0],      # Yellow
    'chair': [139, 69, 19],     # Brown
    'couch': [139, 69, 19],     # Brown
    'potted plant': [0, 255, 0], # Green
    'bed': [139, 69, 19],       # Brown
    'dining table': [139, 69, 19], # Brown
    'toilet': [128, 128, 128],  # Gray
    'tv': [128, 128, 128],      # Gray
    'laptop': [128, 128, 128],  # Gray
    'mouse': [128, 128, 128],   # Gray
    'remote': [128, 128, 128],  # Gray
    'keyboard': [128, 128, 128], # Gray
    'cell phone': [128, 128, 128], # Gray
    'microwave': [128, 128, 128], # Gray
    'oven': [128, 128, 128],    # Gray
    'toaster': [128, 128, 128], # Gray
    'sink': [128, 128, 128],    # Gray
    'refrigerator': [128, 128, 128], # Gray
    'book': [139, 69, 19],      # Brown
    'clock': [128, 128, 128],   # Gray
    'vase': [0, 255, 0],        # Green
    'scissors': [128, 128, 128], # Gray
    'teddy bear': [139, 69, 19], # Brown
    'hair drier': [128, 128, 128], # Gray
    'toothbrush': [128, 128, 128], # Gray
}

class VideoTransformer(VideoProcessorBase):
    def __init__(self):
        self.model = None
        self.colors = DEFAULT_COLORS.copy()
        self.confidence_threshold = 0.5
        self.frame_skip = 2
        self.frame_count = 0
        
    def load_model(self):
        """Load the YOLOv8 segmentation model"""
        if self.model is None:
            try:
                self.model = YOLO('yolov8n-seg.pt')
                return True
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return False
        return True
    
    def set_colors(self, colors):
        """Update color mapping"""
        self.colors = colors
    
    def set_confidence_threshold(self, threshold):
        """Set confidence threshold for detection"""
        self.confidence_threshold = threshold
    
    def set_frame_skip(self, skip):
        """Set frame skip for performance"""
        self.frame_skip = skip
    
    def colorize_frame(self, frame, results):
        """Apply colorization based on segmentation results"""
        if results is None or len(results) == 0:
            return frame
        
        # Create a copy of the frame for colorization
        colorized_frame = frame.copy()
        
        for result in results:
            if result.masks is not None:
                # Get the segmentation mask
                mask = result.masks.data[0].cpu().numpy()
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask = mask > 0.5
                
                # Get the class name
                class_id = int(result.boxes.cls[0].cpu().numpy())
                class_name = result.names[class_id]
                
                # Get color for this class
                color = self.colors.get(class_name, [128, 128, 128])
                
                # Apply color to the masked region
                colorized_frame[mask] = color
        
        return colorized_frame
    
    def transform(self, frame):
        """Transform frame with segmentation and colorization"""
        self.frame_count += 1
        
        # Skip frames for performance
        if self.frame_count % self.frame_skip != 0:
            return frame
        
        # Load model if not loaded
        if not self.load_model():
            return frame
        
        # Convert frame to RGB
        img = frame.to_ndarray(format="bgr24")
        
        try:
            # Run segmentation
            results = self.model(img, conf=self.confidence_threshold, verbose=False)
            
            # Colorize the frame
            colorized_img = self.colorize_frame(img, results)
            
            # Convert back to BGR for display
            return av.VideoFrame.from_ndarray(colorized_img, format="bgr24")
            
        except Exception as e:
            st.error(f"Error processing frame: {e}")
            return frame

# Initialize session state at the very beginning
if 'video_transformer' not in st.session_state:
    st.session_state.video_transformer = VideoTransformer()

# Main header
st.markdown('<h1 class="main-header">🎨 Real-time Video Colorization</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Multi-object colorization based on semantic segmentation</p>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model settings
    st.subheader("Model Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Minimum confidence for object detection"
    )
    
    frame_skip = st.slider(
        "Frame Skip",
        min_value=1,
        max_value=5,
        value=2,
        step=1,
        help="Process every Nth frame for better performance"
    )
    
    # Color customization
    st.subheader("🎨 Color Customization")
    
    # Get unique class names from default colors
    class_names = list(DEFAULT_COLORS.keys())
    
    # Create color pickers for each class
    custom_colors = {}
    for class_name in class_names[:20]:  # Limit to first 20 classes for sidebar
        color = st.color_picker(
            f"{class_name.title()}",
            value=f"#{DEFAULT_COLORS[class_name][0]:02x}{DEFAULT_COLORS[class_name][1]:02x}{DEFAULT_COLORS[class_name][2]:02x}",
            key=f"color_{class_name}"
        )
        
        # Convert hex to RGB
        color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        custom_colors[class_name] = list(color_rgb)
    
    # Add remaining classes with default colors
    for class_name in class_names[20:]:
        custom_colors[class_name] = DEFAULT_COLORS[class_name]
    
    # Reset colors button
    if st.button("Reset to Default Colors"):
        for class_name in class_names:
            custom_colors[class_name] = DEFAULT_COLORS[class_name]
        st.rerun()

# Note: Video processor settings are applied directly in the WebRTC factory function

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📹 Input Options")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Webcam", "Upload Video"],
        help="Select whether to use webcam or upload a video file"
    )
    
    if input_method == "Upload Video":
        uploaded_file = st.file_uploader(
            "Upload a video file (MP4, AVI, MOV)",
            type=['mp4', 'avi', 'mov'],
            help="Upload a video file to process"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            # Display video info
            st.info(f"📁 Uploaded: {uploaded_file.name}")
            
            # Process uploaded video
            if st.button("🎬 Start Processing"):
                st.video(video_path)
                
                # Process video frames
                cap = cv2.VideoCapture(video_path)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Create placeholders for video display
                original_placeholder = st.empty()
                colorized_placeholder = st.empty()
                
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_count}/{total_frames}")
                    
                    # Skip frames for performance
                    if frame_count % frame_skip != 0:
                        continue
                    
                    # Create a temporary processor for video upload processing
                    temp_processor = VideoTransformer()
                    temp_processor.set_colors(custom_colors)
                    temp_processor.set_confidence_threshold(confidence_threshold)
                    temp_processor.set_frame_skip(frame_skip)
                    
                    # Run segmentation
                    results = temp_processor.model(frame, conf=confidence_threshold, verbose=False)
                    
                    # Colorize frame
                    colorized_frame = temp_processor.colorize_frame(frame, results)
                    
                    # Display frames
                    original_placeholder.image(frame, caption="Original", channels="BGR")
                    colorized_placeholder.image(colorized_frame, caption="Colorized", channels="BGR")
                    
                    time.sleep(1/fps)  # Maintain video timing
                
                cap.release()
                progress_bar.empty()
                status_text.text("✅ Processing complete!")
                
                # Clean up temporary file
                os.unlink(video_path)

with col2:
    st.subheader("🎥 Real-time Processing")
    
    if input_method == "Webcam":
        # WebRTC configuration
        rtc_configuration = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })
        
        # Create a new video processor instance for WebRTC
        def create_video_processor():
            processor = VideoTransformer()
            processor.set_colors(custom_colors)
            processor.set_confidence_threshold(confidence_threshold)
            processor.set_frame_skip(frame_skip)
            return processor
        
        # WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="video-colorization",
            video_processor_factory=create_video_processor,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        if webrtc_ctx.state.playing:
            st.success("🎥 Webcam is active! Processing frames in real-time.")
            
            # Display processing info
            st.info("""
            **Processing Information:**
            - Frames are being processed with YOLOv8 segmentation
            - Objects are colorized based on their class
            - Performance optimized with frame skipping
            - Adjust settings in the sidebar for customization
            """)
        else:
            st.info("👆 Click 'Start' to begin webcam processing")

# Information section
st.markdown("---")
st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.markdown("""
### 📋 How it works:
1. **Segmentation**: Uses YOLOv8 to identify objects in each frame
2. **Colorization**: Applies predetermined colors to each object class
3. **Real-time**: Processes frames efficiently for smooth performance
4. **Customization**: Adjust colors and settings in the sidebar

### 🎯 Supported Objects:
- People, vehicles, animals, furniture, electronics, and more
- 80+ different object classes from COCO dataset
- Customizable color scheme for each class

### ⚡ Performance Tips:
- Lower confidence threshold for more detections
- Increase frame skip for better performance
- Use webcam for real-time processing
- Upload videos for batch processing
""")
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    Built with ❤️ using Streamlit, OpenCV, and YOLOv8
</div>
""", unsafe_allow_html=True)
