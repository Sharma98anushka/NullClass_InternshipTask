# 🎨 Real-Time Video Colorization System

A comprehensive Streamlit application for real-time video colorization using deep learning models. This system can colorize grayscale video streams, live webcam feeds, and uploaded video files with multiple model options.

## ✨ Features

- **Real-time Processing**: Live webcam colorization with minimal latency
- **Multiple Models**: Switch between PyTorch Simple and OpenCV DNN models
- **Video Upload**: Process uploaded video files with progress tracking
- **Performance Monitoring**: Real-time FPS and processing time display
- **Download Results**: Save colorized videos for later use
- **User-friendly GUI**: Clean, intuitive Streamlit interface
- **Error Handling**: Robust error handling for various scenarios

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam (for live processing)
- Sufficient RAM (4GB+ recommended)

### Installation

1. **Clone or download the project files**
   ```bash
   # Ensure you have app.py, requirements.txt, and README.md
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in the terminal

## 🎛️ How to Use

### 1. Model Selection
- Choose between two colorization models:
  - **PyTorch Simple**: Fast, lightweight model for real-time processing
  - **OpenCV DNN**: High-quality pre-trained model for better results

### 2. Live Webcam Mode
- Click "🎥 Start Webcam" to begin real-time colorization
- View original grayscale and colorized video side by side
- Monitor FPS and processing time in the sidebar
- Click "⏹️ Stop Webcam" to stop the feed

### 3. Video Upload Mode
- Upload a video file (MP4, AVI, MOV, MKV)
- Watch the processing progress in real-time
- View sample frames from the processed video
- Download the colorized video using the download button

### 4. Performance Monitoring
- **FPS**: Frames per second (higher is better)
- **Processing Time**: Time per frame in milliseconds
- **Model Info**: Description of the selected model

## 🔧 Technical Details

### Models

#### PyTorch Simple Model
- **Architecture**: Lightweight encoder-decoder network
- **Input**: Grayscale images (1 channel)
- **Output**: Color channels (a*b* in LAB color space)
- **Advantages**: Fast inference, suitable for real-time processing
- **Use Case**: Live webcam feeds, quick processing

#### OpenCV DNN Model
- **Architecture**: Pre-trained Caffe model
- **Input**: Grayscale images (L channel in LAB)
- **Output**: Full color images
- **Advantages**: High-quality results, proven performance
- **Use Case**: High-quality video processing, batch operations

### Performance Optimization

- **Frame Resizing**: Automatic resizing for optimal performance
- **Batch Processing**: Efficient handling of multiple frames
- **Memory Management**: Proper cleanup of temporary files
- **Error Recovery**: Graceful handling of model loading failures

## 📁 File Structure

```
project/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── models/            # Model files (created automatically)
    ├── colorization_deploy_v2.prototxt
    ├── pts_in_hull.npy
    └── simple_colorization.pth (optional)
```

## 🛠️ Troubleshooting

### Common Issues

1. **Webcam not working**
   - Ensure your webcam is connected and not in use by other applications
   - Check browser permissions for camera access
   - Try refreshing the page

2. **Model loading errors**
   - Check your internet connection (models are downloaded automatically)
   - Ensure sufficient disk space for model files
   - Try restarting the application

3. **Low FPS**
   - Use the PyTorch Simple model for better performance
   - Close other applications to free up system resources
   - Consider using a GPU if available

4. **Memory errors**
   - Reduce video resolution or frame rate
   - Process shorter video clips
   - Restart the application

### Performance Tips

- **For Real-time Processing**: Use PyTorch Simple model
- **For High Quality**: Use OpenCV DNN model
- **For Large Videos**: Process in smaller chunks
- **For Better FPS**: Close unnecessary applications

## 🔮 Future Enhancements

- [ ] GPU acceleration support
- [ ] Additional colorization models
- [ ] Batch processing for multiple videos
- [ ] Custom color palette selection
- [ ] Video quality settings
- [ ] Export to different formats

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Review the error messages in the application
3. Open an issue with detailed information about your problem

---

**Built with ❤️ using Streamlit, OpenCV, and PyTorch**

