# Interactive User-Guided Image Colorization

An AI-powered application that allows users to interactively colorize grayscale images by selecting regions and applying custom colors. The system uses a deep learning model to generate realistic colorization based on user input while maintaining natural appearance.

## Features

- **Image Upload**: Upload grayscale images in various formats (JPG, PNG, etc.)
- **Interactive Region Selection**: Select specific areas using brush or bounding box tools
- **Custom Color Picker**: Choose exact colors for selected regions
- **Real-time Preview**: See colorization results update dynamically
- **AI-Powered Colorization**: Deep learning model ensures realistic color application

## Project Structure

```
Project_Null4/
├── model/              # Model implementation
│   ├── __init__.py
│   ├── colorization_model.py
│   └── model_utils.py
├── gui/                # Streamlit interface
│   ├── __init__.py
│   ├── app.py
│   └── components.py
├── utils/              # Helper functions
│   ├── __init__.py
│   ├── image_processing.py
│   └── color_utils.py
├── tests/              # Unit tests
│   ├── __init__.py
│   ├── test_image_processing.py
│   └── test_color_utils.py
├── main.py             # Application entry point
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Project_Null1
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Launch the application**:
   ```bash
   python main.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Upload an image** using the file uploader

4. **Select regions** using the interactive tools:
   - Brush tool: Freehand selection
   - Bounding box: Rectangular selection

5. **Choose colors** using the color picker

6. **View results** in the real-time preview window

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Project Components

- **Model**: PyTorch-based colorization model with user guidance
- **GUI**: Streamlit interface with interactive tools
- **Utils**: Image processing and color manipulation functions
- **Tests**: Unit tests for core functionality

## Technical Details

- **AI Framework**: PyTorch for deep learning
- **Image Processing**: OpenCV and NumPy
- **Web Interface**: Streamlit
- **Interactive Tools**: Custom canvas components

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the deep learning framework
- Streamlit for the web interface
- OpenCV for image processing capabilities




