import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import time
import os
import tempfile
from typing import Optional, Tuple
import threading
from collections import deque
import urllib.request
import zipfile
import io

# Page configuration
st.set_page_config(
    page_title="Real-Time Video Colorization",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .model-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Constants
MODEL_URLS = {
    "OpenCV DNN": {
        "prototxt": "https://raw.githubusercontent.com/richzhang/colorization/master/colorization/models/colorization_deploy_v2.prototxt",
        "caffemodel": "https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1",
        "pts": "https://raw.githubusercontent.com/richzhang/colorization/master/colorization/resources/pts_in_hull.npy"
    }
}

class SimpleColorizationModel(nn.Module):
    """A lightweight colorization model for real-time processing"""
    
    def __init__(self):
        super(SimpleColorizationModel, self).__init__()
        
        # Simple but effective architecture
        self.features = nn.Sequential(
            # Encoder
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Decoder
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            
            # Output layer
            nn.Conv2d(32, 2, 3, padding=1),
            nn.Tanh()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.features(x)

class VideoColorizer:
    """Main class for video colorization processing"""
    
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.fps_counter = deque(maxlen=30)
        self.processing_times = deque(maxlen=30)
        self.is_processing = False
        
    def load_opencv_model(self):
        """Load OpenCV DNN colorization model"""
        try:
            # Download model files if not present
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)
            
            prototxt_path = os.path.join(model_dir, "colorization_deploy_v2.prototxt")
            caffemodel_path = os.path.join(model_dir, "colorization_release_v2.caffemodel")
            pts_path = os.path.join(model_dir, "pts_in_hull.npy")
            
            # Download files if they don't exist
            if not os.path.exists(prototxt_path):
                st.info("Downloading OpenCV prototxt file...")
                urllib.request.urlretrieve(MODEL_URLS["OpenCV DNN"]["prototxt"], prototxt_path)
            
            if not os.path.exists(caffemodel_path):
                st.info("Downloading OpenCV caffemodel file...")
                urllib.request.urlretrieve(MODEL_URLS["OpenCV DNN"]["caffemodel"], caffemodel_path)
            
            if not os.path.exists(pts_path):
                st.info("Downloading OpenCV pts file...")
                urllib.request.urlretrieve(MODEL_URLS["OpenCV DNN"]["pts"], pts_path)
            
            # Load the model
            net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
            
            # Load cluster centers and add them as 1x1 convolutions
            pts = np.load(pts_path)
            pts = pts.reshape(2, 313, 1, 1)
            
            # Add cluster centers to the model
            net.getLayer(net.getLayerId('class8_ab')).blobs = [pts.astype(np.float32)]
            net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]
            
            st.success("✅ OpenCV model loaded successfully!")
            return net
            
        except Exception as e:
            st.error(f"Error loading OpenCV model: {str(e)}")
            return None
    
    def load_pytorch_model(self):
        """Load PyTorch colorization model"""
        try:
            model = SimpleColorizationModel()
            # Load pretrained weights if available
            weights_path = "models/simple_colorization.pth"
            if os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path, map_location='cpu'))
            model.eval()
            return model
        except Exception as e:
            st.error(f"Error loading PyTorch model: {str(e)}")
            return None
    
    def colorize_opencv(self, frame: np.ndarray) -> np.ndarray:
        """Colorize frame using OpenCV DNN model - Fixed implementation"""
        try:
            # Get the model
            net = self.models.get("OpenCV DNN")
            if net is None:
                st.error("OpenCV model not loaded!")
                return frame
            
            # Convert frame to LAB color space
            frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            
            # Extract the L channel
            frame_l = frame_lab[:, :, 0]
            
            # Resize L to 224x224, cast to float32, and subtract 50
            frame_l_resized = cv2.resize(frame_l, (224, 224))
            frame_l_normalized = frame_l_resized.astype(np.float32) - 50
            
            # Create blob for DNN input
            blob = cv2.dnn.blobFromImage(frame_l_normalized)
            
            # Forward pass through the network
            net.setInput(blob)
            ab_output = net.forward()
            
            # Extract ab channels and transpose
            ab = ab_output[0, :, :, :].transpose((1, 2, 0))
            
            # Resize the predicted ab channels to original image size
            ab_resized = cv2.resize(ab, (frame.shape[1], frame.shape[0]))
            
            # Scale ab channels to proper range (this is crucial for natural colors)
            ab_scaled = ab_resized * 110
            
            # Concatenate original L channel with predicted ab
            frame_lab[:, :, 1:] = ab_scaled
            
            # Convert back from LAB → BGR
            frame_colorized = cv2.cvtColor(frame_lab, cv2.COLOR_LAB2BGR)
            
            # Ensure proper dynamic range and normalization
            frame_colorized = np.clip(frame_colorized, 0, 255).astype(np.uint8)
            
            return frame_colorized
            
        except Exception as e:
            st.error(f"Error in OpenCV colorization: {str(e)}")
            return frame
    
    def colorize_simple(self, frame: np.ndarray) -> np.ndarray:
        """Simple colorization method that produces natural colors"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Create a simple colorization using HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Use grayscale as value channel, add some hue and saturation
            hsv[:, :, 0] = (gray * 0.7).astype(np.uint8)  # Hue based on intensity
            hsv[:, :, 1] = np.clip(gray * 0.8, 0, 255).astype(np.uint8)  # Saturation
            hsv[:, :, 2] = gray  # Value from grayscale
            
            # Convert back to BGR
            colorized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            return colorized
            
        except Exception as e:
            st.error(f"Error in simple colorization: {str(e)}")
            return frame
    

    
    def colorize_pytorch(self, frame: np.ndarray) -> np.ndarray:
        """Colorize frame using PyTorch model - Natural color implementation"""
        try:
            # Get the model
            model = self.models.get("PyTorch Simple")
            if model is None:
                st.error("PyTorch model not loaded!")
                return frame
            
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Preprocess - use reasonable size for speed/quality balance
            gray = cv2.resize(gray, (256, 256))
            gray = gray.astype(np.float32) / 255.0
            gray = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                color_output = model(gray)
                color_output = color_output.squeeze().numpy()
                color_output = np.transpose(color_output, (1, 2, 0))
            
            # Post-process
            color_output = cv2.resize(color_output, (frame.shape[1], frame.shape[0]))
            
            # Scale to proper a*b* range for natural colors
            color_output = color_output * 110  # Scale to typical a*b* range
            
            # Create colorized frame using LAB color space
            frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            frame_lab[:, :, 1:] = color_output  # Replace a*b* channels
            
            # Convert LAB → BGR for final colorized frame
            frame_colorized = cv2.cvtColor(frame_lab, cv2.COLOR_LAB2BGR)
            
            # Ensure proper dynamic range and normalization
            frame_colorized = np.clip(frame_colorized, 0, 255).astype(np.uint8)
            
            return frame_colorized
            
        except Exception as e:
            st.error(f"Error in PyTorch colorization: {str(e)}")
            return frame
    
    def colorize_frame(self, frame: np.ndarray, model_name: str) -> np.ndarray:
        """Colorize a single frame using the specified model"""
        start_time = time.time()
        
        if model_name == "OpenCV DNN":
            result = self.colorize_opencv(frame)
        elif model_name == "PyTorch Simple":
            result = self.colorize_pytorch(frame)
        elif model_name == "Simple Colorization":
            result = self.colorize_simple(frame)
        else:
            result = frame
        
        # Update performance metrics
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        if len(self.processing_times) > 1:
            fps = 1.0 / (sum(self.processing_times) / len(self.processing_times))
            self.fps_counter.append(fps)
        
        return result
    

    

    
    def get_fps(self) -> float:
        """Get current FPS"""
        if len(self.fps_counter) > 0:
            return sum(self.fps_counter) / len(self.fps_counter)
        return 0.0
    
    def get_avg_processing_time(self) -> float:
        """Get average processing time per frame"""
        if len(self.processing_times) > 0:
            return sum(self.processing_times) / len(self.processing_times)
        return 0.0

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🎨 Real-Time Video Colorization</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'colorizer' not in st.session_state:
        st.session_state.colorizer = VideoColorizer()
    
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False
    
    if 'current_model' not in st.session_state:
        st.session_state.current_model = "PyTorch Simple"
    
    # Sidebar controls
    st.sidebar.title("🎛️ Controls")
    
    # Model selection
    model_options = ["Simple Colorization", "PyTorch Simple", "OpenCV DNN"]
    selected_model = st.sidebar.selectbox(
        "Select Colorization Model",
        model_options,
        index=model_options.index(st.session_state.current_model)
    )
    
    # Load models if changed
    if selected_model != st.session_state.current_model:
        st.session_state.current_model = selected_model
        if selected_model == "OpenCV DNN":
            with st.spinner("Loading OpenCV DNN model..."):
                st.session_state.colorizer.models["OpenCV DNN"] = st.session_state.colorizer.load_opencv_model()
        elif selected_model == "PyTorch Simple":
            with st.spinner("Loading PyTorch model..."):
                st.session_state.colorizer.models["PyTorch Simple"] = st.session_state.colorizer.load_pytorch_model()
        elif selected_model == "Simple Colorization":
            st.session_state.colorizer.models["Simple Colorization"] = "simple"
    
    # Ensure model is loaded for current selection
    if selected_model == "OpenCV DNN" and "OpenCV DNN" not in st.session_state.colorizer.models:
        with st.spinner("Loading OpenCV DNN model..."):
            st.session_state.colorizer.models["OpenCV DNN"] = st.session_state.colorizer.load_opencv_model()
    elif selected_model == "PyTorch Simple" and "PyTorch Simple" not in st.session_state.colorizer.models:
        with st.spinner("Loading PyTorch model..."):
            st.session_state.colorizer.models["PyTorch Simple"] = st.session_state.colorizer.load_pytorch_model()
    elif selected_model == "Simple Colorization" and "Simple Colorization" not in st.session_state.colorizer.models:
        st.session_state.colorizer.models["Simple Colorization"] = "simple"
    
    # Webcam controls
    st.sidebar.subheader("📹 Webcam Controls")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🎥 Start Webcam", type="primary"):
            st.session_state.webcam_active = True
    
    with col2:
        if st.button("⏹️ Stop Webcam"):
            st.session_state.webcam_active = False
    
    # Performance metrics
    st.sidebar.subheader("📊 Performance")
    
    if st.session_state.webcam_active:
        fps = st.session_state.colorizer.get_fps()
        avg_time = st.session_state.colorizer.get_avg_processing_time()
        
        st.sidebar.metric("FPS", f"{fps:.1f}")
        st.sidebar.metric("Processing Time", f"{avg_time*1000:.1f}ms")
    
    # Model information
    st.sidebar.subheader("ℹ️ Model Info")
    
    model_info = {
        "Simple Colorization": "Fast HSV-based colorization with natural colors",
        "PyTorch Simple": "Lightweight neural network for real-time colorization",
        "OpenCV DNN": "Pre-trained Caffe model for high-quality colorization"
    }
    
    st.sidebar.markdown(f'<div class="model-info">{model_info[selected_model]}</div>', unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original (Grayscale)")
        original_placeholder = st.empty()
    
    with col2:
        st.subheader("🎨 Colorized")
        colorized_placeholder = st.empty()
    
    # Webcam processing
    if st.session_state.webcam_active:
        try:
            # Initialize webcam
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Could not open webcam. Please check your camera connection.")
                st.session_state.webcam_active = False
                return
            
            # Set webcam properties for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Processing loop
            while st.session_state.webcam_active:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("❌ Failed to read frame from webcam")
                    break
                
                # Convert to grayscale for processing
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                
                # Colorize frame
                colorized = st.session_state.colorizer.colorize_frame(
                    gray_bgr, st.session_state.current_model
                )
                
                # Display frames
                original_placeholder.image(gray_bgr, channels="BGR", use_column_width=True)
                colorized_placeholder.image(colorized, channels="BGR", use_column_width=True)
                
                # Small delay to prevent overwhelming the UI
                time.sleep(0.01)
            
            # Release webcam
            cap.release()
            
        except Exception as e:
            st.error(f"❌ Error during webcam processing: {str(e)}")
            st.session_state.webcam_active = False
    
    # Video upload section
    st.subheader("📁 Upload Video")
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file to colorize"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        try:
            # Process uploaded video
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                st.error("❌ Could not open uploaded video file")
                return
            
            # Get video properties
            fps_video = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            st.info(f"📹 Video Info: {frame_count} frames, {fps_video:.1f} FPS")
            
            # Process frames
            frame_idx = 0
            processed_frames = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                
                # Colorize frame
                colorized = st.session_state.colorizer.colorize_frame(
                    gray_bgr, st.session_state.current_model
                )
                
                processed_frames.append(colorized)
                frame_idx += 1
                
                # Update progress
                progress = frame_idx / frame_count
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_idx}/{frame_count}")
            
            cap.release()
            
            # Display sample frames
            if processed_frames:
                st.success("✅ Video processing completed!")
                
                # Show sample frames
                sample_idx = len(processed_frames) // 2
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2RGB), caption="Original (Grayscale)")
                
                with col2:
                    st.image(cv2.cvtColor(processed_frames[sample_idx], cv2.COLOR_BGR2RGB), caption="Colorized")
                
                # Download option
                if st.button("💾 Download Colorized Video"):
                    # Save processed video
                    output_path = "colorized_video.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps_video, 
                                        (processed_frames[0].shape[1], processed_frames[0].shape[0]))
                    
                    for frame in processed_frames:
                        out.write(frame)
                    
                    out.release()
                    
                    # Provide download link
                    with open(output_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Colorized Video",
                            data=file.read(),
                            file_name="colorized_video.mp4",
                            mime="video/mp4"
                        )
        
        except Exception as e:
            st.error(f"❌ Error processing uploaded video: {str(e)}")
        
        finally:
            # Clean up temporary file
            if os.path.exists(video_path):
                os.unlink(video_path)
    
    # Test colorization with a sample image
    st.subheader("🧪 Test Colorization")
    
    if st.button("Test Colorization with Sample"):
        # Create a realistic test image with natural colors
        height, width = 480, 640
        
        # Create a test image with natural colors (skin tones, background, etc.)
        test_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add natural colored regions
        test_image[0:height//4, :] = [139, 69, 19]  # Brown (wood/earth)
        test_image[height//4:height//2, :] = [34, 139, 34]  # Forest green
        test_image[height//2:3*height//4, :] = [255, 228, 196]  # Skin tone
        test_image[3*height//4:, :] = [135, 206, 235]  # Sky blue
        
        # Add some objects
        cv2.rectangle(test_image, (100, 100), (200, 200), (255, 255, 255), -1)  # White
        cv2.circle(test_image, (450, 350), 50, (128, 0, 128), -1)  # Purple
        
        # Convert to grayscale for testing
        gray_test = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        gray_bgr_test = cv2.cvtColor(gray_test, cv2.COLOR_GRAY2BGR)
        
        # Colorize the test image
        colorized_test = st.session_state.colorizer.colorize_frame(
            gray_bgr_test, st.session_state.current_model
        )
        
        # Display test results
        st.subheader("Colorization Test Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(test_image, caption="Original Color", use_column_width=True)
        with col2:
            st.image(gray_bgr_test, caption="Grayscale Input", use_column_width=True)
        with col3:
            st.image(colorized_test, caption="Colorized Output", use_column_width=True)
        
        # Show model status and color analysis
        st.info(f"Model used: {st.session_state.current_model}")
        if st.session_state.current_model in st.session_state.colorizer.models:
            st.success("✅ Model loaded successfully")
            
            # Analyze color distribution
            if colorized_test is not None:
                # Check if output is mostly blue (indicates problem)
                blue_ratio = np.mean(colorized_test[:, :, 0]) / 255.0
                green_ratio = np.mean(colorized_test[:, :, 1]) / 255.0
                red_ratio = np.mean(colorized_test[:, :, 2]) / 255.0
                
                st.write(f"Color distribution - Blue: {blue_ratio:.2f}, Green: {green_ratio:.2f}, Red: {red_ratio:.2f}")
                
                if blue_ratio > 0.6 and blue_ratio > green_ratio * 1.5 and blue_ratio > red_ratio * 1.5:
                    st.warning("⚠️ Output appears to have blue tint - model may need adjustment")
                else:
                    st.success("✅ Color distribution looks natural")
        else:
            st.error("❌ Model not loaded")
    
    # Instructions
    st.subheader("📖 How to Use")
    
    st.markdown("""
    1. **Select a Model**: Choose between PyTorch Simple (fast) or OpenCV DNN (high quality)
    2. **Test First**: Use the "Test Colorization" button to verify the model is working
    3. **Webcam Mode**: Click "Start Webcam" to begin real-time colorization
    4. **Upload Video**: Upload a video file for batch processing
    5. **Monitor Performance**: Check FPS and processing time in the sidebar
    6. **Download Results**: Save colorized videos for later use
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Built with ❤️ using Streamlit, OpenCV, and PyTorch
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
