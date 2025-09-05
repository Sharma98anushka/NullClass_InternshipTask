#!/usr/bin/env python3
"""
Interactive User-Guided Image Colorization
Main entry point for the application.

This script launches the Streamlit web interface for the AI colorization tool.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import torch
        import streamlit
        import cv2
        import numpy as np
        print("✓ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def main():
    """Main entry point for the application."""
    print("🎨 Interactive User-Guided Image Colorization")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Get the path to the Streamlit app
    current_dir = Path(__file__).parent
    app_path = current_dir / "gui" / "app.py"
    
    if not app_path.exists():
        print(f"✗ Streamlit app not found at {app_path}")
        sys.exit(1)
    
    print("🚀 Launching Streamlit application...")
    print("📱 Open your browser and navigate to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("-" * 50)
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path), "--server.port", "8501"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error launching application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()




