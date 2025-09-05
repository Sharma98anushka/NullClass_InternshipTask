#!/usr/bin/env python3
"""
Installation script for the Video Colorization App
Automatically sets up the environment and dependencies
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def print_banner():
    """Print installation banner"""
    print("=" * 60)
    print("🎨 Video Colorization App - Installation Script")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_pip_packages():
    """Install required pip packages"""
    print("\n📦 Installing required packages...")
    
    packages = [
        'streamlit==1.28.1',
        'opencv-python==4.8.1.78',
        'ultralytics==8.0.196',
        'torch==2.1.0',
        'torchvision==0.16.0',
        'numpy==1.24.3',
        'Pillow==10.0.1',
        'streamlit-webrtc==0.47.1',
        'av==10.0.0'
    ]
    
    for package in packages:
        print(f"   Installing {package}...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"   ✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {package}: {e}")
            return False
    
    return True

def check_gpu_support():
    """Check for GPU support"""
    print("\n🖥️  Checking GPU support...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("ℹ️  No CUDA GPU detected, will use CPU")
            return False
    except ImportError:
        print("ℹ️  PyTorch not installed yet, will check after installation")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = ['logs', 'outputs', 'temp']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✅ Created directory: {directory}")

def download_model():
    """Download YOLOv8 model"""
    print("\n🤖 Downloading YOLOv8 model...")
    try:
        from ultralytics import YOLO
        print("   Downloading yolov8n-seg.pt...")
        model = YOLO('yolov8n-seg.pt')
        print("   ✅ Model downloaded successfully")
        return True
    except Exception as e:
        print(f"   ❌ Failed to download model: {e}")
        print("   ℹ️  Model will be downloaded automatically when running the app")
        return False

def test_installation():
    """Test the installation"""
    print("\n🧪 Testing installation...")
    
    try:
        # Test imports
        import streamlit
        import cv2
        import ultralytics
        import torch
        import numpy
        from PIL import Image
        print("   ✅ All packages imported successfully")
        
        # Test OpenCV
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("   ✅ Webcam access test passed")
            cap.release()
        else:
            print("   ⚠️  Webcam not available (this is normal if no camera is connected)")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def create_run_script():
    """Create a run script for easy execution"""
    print("\n📝 Creating run script...")
    
    if platform.system() == "Windows":
        script_content = """@echo off
echo Starting Video Colorization App...
streamlit run app.py
pause
"""
        script_name = "run_app.bat"
    else:
        script_content = """#!/bin/bash
echo "Starting Video Colorization App..."
streamlit run app.py
"""
        script_name = "run_app.sh"
    
    with open(script_name, 'w') as f:
        f.write(script_content)
    
    if platform.system() != "Windows":
        os.chmod(script_name, 0o755)
    
    print(f"   ✅ Created {script_name}")

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 60)
    print("🎉 Installation completed successfully!")
    print("=" * 60)
    print("\n📋 Next steps:")
    print("1. Run the application:")
    if platform.system() == "Windows":
        print("   Double-click 'run_app.bat' or run: streamlit run app.py")
    else:
        print("   ./run_app.sh or run: streamlit run app.py")
    
    print("2. Open your browser and go to: http://localhost:8501")
    print("3. Choose your input method (webcam or video upload)")
    print("4. Adjust settings in the sidebar as needed")
    print("\n📚 For more information, see README.md")
    print("\n🔧 Troubleshooting:")
    print("- If webcam doesn't work, check camera permissions")
    print("- If performance is slow, increase frame skip in settings")
    print("- For GPU acceleration, ensure CUDA is properly installed")

def main():
    """Main installation function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check GPU support
    gpu_available = check_gpu_support()
    
    # Install packages
    if not install_pip_packages():
        print("\n❌ Installation failed. Please check the error messages above.")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Download model (optional)
    download_model()
    
    # Test installation
    if not test_installation():
        print("\n❌ Installation test failed. Please check the error messages above.")
        sys.exit(1)
    
    # Create run script
    create_run_script()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()


