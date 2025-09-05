#!/usr/bin/env python3
"""
Test script for the Video Colorization App
Verifies all components are working correctly
"""

import sys
import os
import time

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing package imports...")
    
    try:
        import streamlit
        print("   ✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"   ❌ Streamlit import failed: {e}")
        return False
    
    try:
        import cv2
        print("   ✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"   ❌ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("   ✅ NumPy imported successfully")
    except ImportError as e:
        print(f"   ❌ NumPy import failed: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("   ✅ Ultralytics imported successfully")
    except ImportError as e:
        print(f"   ❌ Ultralytics import failed: {e}")
        return False
    
    try:
        import torch
        print("   ✅ PyTorch imported successfully")
    except ImportError as e:
        print(f"   ❌ PyTorch import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("   ✅ Pillow imported successfully")
    except ImportError as e:
        print(f"   ❌ Pillow import failed: {e}")
        return False
    
    try:
        from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
        print("   ✅ Streamlit-WebRTC imported successfully")
    except ImportError as e:
        print(f"   ❌ Streamlit-WebRTC import failed: {e}")
        return False
    
    try:
        import av
        print("   ✅ PyAV imported successfully")
    except ImportError as e:
        print(f"   ❌ PyAV import failed: {e}")
        return False
    
    return True

def test_yolo_model():
    """Test YOLO model loading"""
    print("\n🤖 Testing YOLO model...")
    
    try:
        from ultralytics import YOLO
        print("   Loading YOLOv8 model...")
        model = YOLO('yolov8n-seg.pt')
        print("   ✅ YOLOv8 model loaded successfully")
        return True
    except Exception as e:
        print(f"   ❌ YOLO model loading failed: {e}")
        print("   ℹ️  This is normal on first run - model will download automatically")
        return False

def test_opencv():
    """Test OpenCV functionality"""
    print("\n📹 Testing OpenCV...")
    
    try:
        import cv2
        import numpy as np
        
        # Test creating a simple image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:] = [255, 0, 0]  # Red color
        
        # Test video capture (just check if it can be initialized)
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("   ✅ Webcam access available")
            cap.release()
        else:
            print("   ⚠️  Webcam not available (this is normal if no camera is connected)")
        
        print("   ✅ OpenCV functionality test passed")
        return True
        
    except Exception as e:
        print(f"   ❌ OpenCV test failed: {e}")
        return False

def test_streamlit():
    """Test Streamlit functionality"""
    print("\n🌐 Testing Streamlit...")
    
    try:
        import streamlit as st
        
        # Test basic Streamlit functions
        st.set_page_config(page_title="Test", page_icon="🎨")
        print("   ✅ Streamlit configuration test passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Streamlit test failed: {e}")
        return False

def test_gpu():
    """Test GPU availability"""
    print("\n🖥️  Testing GPU support...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"   ✅ CUDA GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   ✅ CUDA version: {torch.version.cuda}")
            return True
        else:
            print("   ℹ️  No CUDA GPU detected - will use CPU")
            return True
            
    except Exception as e:
        print(f"   ❌ GPU test failed: {e}")
        return False

def test_file_structure():
    """Test if all required files exist"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        'app.py',
        'requirements.txt',
        'README.md',
        'demo.py',
        'config.py',
        'install.py'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ {file} exists")
        else:
            print(f"   ❌ {file} missing")
            missing_files.append(file)
    
    if missing_files:
        print(f"   ⚠️  Missing files: {', '.join(missing_files)}")
        return False
    
    return True

def run_demo_test():
    """Run a quick demo test"""
    print("\n🎬 Running demo test...")
    
    try:
        # Import demo functions
        from demo import colorize_frame, DEFAULT_COLORS
        
        # Create a test image
        import numpy as np
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test colorization function
        result = colorize_frame(test_image, None, DEFAULT_COLORS)
        
        if result is not None:
            print("   ✅ Demo colorization function works")
            return True
        else:
            print("   ❌ Demo colorization function failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Demo test failed: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 Video Colorization App - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("File Structure", test_file_structure),
        ("OpenCV", test_opencv),
        ("Streamlit", test_streamlit),
        ("GPU Support", test_gpu),
        ("YOLO Model", test_yolo_model),
        ("Demo Functions", run_demo_test),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"   ⚠️  {test_name} test failed")
        except Exception as e:
            print(f"   ❌ {test_name} test error: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The app is ready to run.")
        print("\n📋 Next steps:")
        print("1. Run: streamlit run app.py")
        print("2. Open browser to: http://localhost:8501")
        print("3. Enjoy the video colorization app!")
    elif passed >= total - 1:
        print("✅ Most tests passed! The app should work.")
        print("⚠️  Some optional features may not be available.")
        print("\n📋 You can still try running the app:")
        print("1. Run: streamlit run app.py")
        print("2. Open browser to: http://localhost:8501")
    else:
        print("❌ Several tests failed. Please check the installation.")
        print("💡 Try running: python install.py")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


