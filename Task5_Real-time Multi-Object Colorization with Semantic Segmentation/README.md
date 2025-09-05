# 🎨 Real-time Video Colorization App

A powerful Streamlit web application that performs real-time multi-object colorization based on semantic segmentation using YOLOv8.

## ✨ Features

- **Real-time Processing**: Live webcam feed with instant colorization
- **Video Upload**: Process uploaded video files (MP4, AVI, MOV)
- **Semantic Segmentation**: Uses YOLOv8 to identify 80+ object classes
- **Custom Colorization**: Predefined color scheme for each object class
- **Interactive UI**: Clean, responsive Streamlit interface
- **Performance Optimization**: Configurable frame skipping and confidence thresholds
- **Color Customization**: Adjust colors for each object class in real-time

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam (for real-time processing)
- Modern web browser

### Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📖 Usage Guide

### Webcam Processing

1. Select "Webcam" as the input method
2. Click "Start" to begin webcam processing
3. The app will automatically:
   - Load the YOLOv8 segmentation model
   - Process frames in real-time
   - Apply colorization based on detected objects
   - Display the colorized output

### Video Upload Processing

1. Select "Upload Video" as the input method
2. Upload a video file (MP4, AVI, or MOV format)
3. Click "Start Processing" to begin
4. Watch the progress as frames are processed
5. View both original and colorized frames side-by-side

### Configuration Options

#### Model Settings (Sidebar)
- **Confidence Threshold**: Adjust detection sensitivity (0.1 - 1.0)
- **Frame Skip**: Process every Nth frame for performance (1-5)

#### Color Customization (Sidebar)
- **Individual Color Pickers**: Customize colors for each object class
- **Reset Button**: Return to default color scheme

## 🎯 Supported Object Classes

The app recognizes 80+ object classes from the COCO dataset, including:

### People & Animals
- Person, Bird, Cat, Dog, Horse, Sheep, Cow, Elephant, Bear, Zebra, Giraffe

### Vehicles
- Car, Bicycle, Motorcycle, Airplane, Bus, Train, Truck, Boat

### Furniture & Objects
- Chair, Couch, Bed, Table, TV, Laptop, Phone, Book, Clock

### Food & Kitchen Items
- Apple, Orange, Pizza, Cake, Bottle, Cup, Fork, Knife, Spoon

### Outdoor Objects
- Traffic Light, Stop Sign, Fire Hydrant, Bench, Tree, Building

## ⚡ Performance Optimization

### For Real-time Processing
- **Increase Frame Skip**: Process every 2nd or 3rd frame
- **Lower Confidence**: Reduce threshold for faster detection
- **Use Webcam**: Direct processing without file I/O overhead

### For Video Processing
- **Higher Frame Skip**: Process every 3rd-5th frame for large videos
- **Adjust Confidence**: Balance between accuracy and speed
- **Monitor Progress**: Use the progress bar to track processing

## 🔧 Technical Details

### Architecture
- **Frontend**: Streamlit with custom CSS styling
- **Video Processing**: OpenCV for frame handling
- **Segmentation**: YOLOv8-nano for real-time performance
- **WebRTC**: Streamlit-webrtc for live webcam streaming
- **Colorization**: Custom mask-based color application

### Model Information
- **Model**: YOLOv8-nano segmentation (`yolov8n-seg.pt`)
- **Dataset**: COCO (Common Objects in Context)
- **Classes**: 80 object categories
- **Input**: RGB images/video frames
- **Output**: Segmentation masks with bounding boxes

### Color Scheme
Each object class has a predefined color:
- **Red**: People, cars, fire hydrants, stop signs
- **Green**: Plants, bottles, bowls, vegetables
- **Blue**: Airplanes, boats, water-related objects
- **Orange**: Animals, food items, vehicles
- **Purple**: Bags, accessories, sports equipment
- **Yellow**: Traffic lights, fruits, sports items
- **Gray**: Electronics, furniture, appliances
- **Brown**: Wooden objects, furniture, animals

## 🛠️ Customization

### Adding New Colors
Edit the `DEFAULT_COLORS` dictionary in `app.py`:
```python
DEFAULT_COLORS = {
    'person': [255, 0, 0],      # Red
    'car': [0, 255, 0],         # Green
    # Add more custom colors...
}
```

### Changing Model
Replace the model in the `load_model()` method:
```python
self.model = YOLO('yolov8s-seg.pt')  # Larger model
# or
self.model = YOLO('yolov8m-seg.pt')  # Medium model
```

## 📱 Browser Compatibility

- **Chrome**: Full support (recommended)
- **Firefox**: Full support
- **Safari**: Full support
- **Edge**: Full support

## 🔍 Troubleshooting

### Common Issues

1. **Webcam not working**
   - Ensure camera permissions are granted
   - Try refreshing the page
   - Check if another app is using the camera

2. **Slow performance**
   - Increase frame skip value
   - Lower confidence threshold
   - Close other applications
   - Use a smaller video file

3. **Model loading error**
   - Check internet connection (model downloads automatically)
   - Ensure sufficient disk space
   - Restart the application

4. **Video upload issues**
   - Check file format (MP4, AVI, MOV)
   - Ensure file size is reasonable (< 100MB recommended)
   - Try a different video file

### Performance Tips

- **GPU Acceleration**: Install CUDA for faster processing
- **Memory Management**: Close other applications
- **Network**: Stable internet for model download
- **Browser**: Use Chrome for best performance

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section
2. Review the console output for error messages
3. Ensure all dependencies are properly installed

---

**Built with ❤️ using Streamlit, OpenCV, and YOLOv8**

