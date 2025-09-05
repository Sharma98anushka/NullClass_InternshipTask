"""
Configuration file for the Video Colorization App
Centralize all settings and parameters here for easy customization
"""

# Model Configuration
MODEL_CONFIG = {
    'model_name': 'yolov8n-seg.pt',  # Model to use (nano, small, medium, large)
    'confidence_threshold': 0.5,      # Default confidence threshold
    'frame_skip': 2,                  # Default frame skip for performance
    'device': 'auto',                 # Device to use ('cpu', 'cuda', 'auto')
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'max_fps': 30,                    # Maximum FPS for processing
    'buffer_size': 10,                # Buffer size for frame processing
    'enable_gpu': True,               # Enable GPU acceleration if available
    'optimize_memory': True,          # Enable memory optimization
}

# UI Configuration
UI_CONFIG = {
    'page_title': 'Real-time Video Colorization',
    'page_icon': '🎨',
    'layout': 'wide',
    'sidebar_state': 'expanded',
    'theme': {
        'primary_color': '#1f77b4',
        'background_color': '#ffffff',
        'text_color': '#333333',
    }
}

# Color Scheme Configuration
# Each object class has a predefined color in RGB format
DEFAULT_COLORS = {
    # People
    'person': [255, 0, 0],      # Red
    
    # Vehicles
    'bicycle': [255, 165, 0],   # Orange
    'car': [255, 0, 0],         # Red
    'motorcycle': [255, 165, 0], # Orange
    'airplane': [0, 0, 255],    # Blue
    'bus': [255, 0, 0],         # Red
    'train': [128, 0, 128],     # Purple
    'truck': [255, 0, 0],       # Red
    'boat': [0, 0, 255],        # Blue
    
    # Traffic Objects
    'traffic light': [255, 255, 0], # Yellow
    'fire hydrant': [255, 0, 0], # Red
    'stop sign': [255, 0, 0],   # Red
    'parking meter': [128, 128, 128], # Gray
    
    # Outdoor Objects
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
    
    # Bags and Accessories
    'backpack': [128, 0, 128],  # Purple
    'umbrella': [128, 0, 128],  # Purple
    'handbag': [128, 0, 128],   # Purple
    'tie': [128, 0, 128],       # Purple
    'suitcase': [128, 0, 128],  # Purple
    
    # Sports Equipment
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
    
    # Kitchen Items
    'bottle': [0, 255, 0],      # Green
    'wine glass': [0, 255, 0],  # Green
    'cup': [0, 255, 0],         # Green
    'fork': [128, 128, 128],    # Gray
    'knife': [128, 128, 128],   # Gray
    'spoon': [128, 128, 128],   # Gray
    'bowl': [0, 255, 0],        # Green
    
    # Food Items
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
    
    # Furniture
    'chair': [139, 69, 19],     # Brown
    'couch': [139, 69, 19],     # Brown
    'potted plant': [0, 255, 0], # Green
    'bed': [139, 69, 19],       # Brown
    'dining table': [139, 69, 19], # Brown
    'toilet': [128, 128, 128],  # Gray
    
    # Electronics
    'tv': [128, 128, 128],      # Gray
    'laptop': [128, 128, 128],  # Gray
    'mouse': [128, 128, 128],   # Gray
    'remote': [128, 128, 128],  # Gray
    'keyboard': [128, 128, 128], # Gray
    'cell phone': [128, 128, 128], # Gray
    
    # Appliances
    'microwave': [128, 128, 128], # Gray
    'oven': [128, 128, 128],    # Gray
    'toaster': [128, 128, 128], # Gray
    'sink': [128, 128, 128],    # Gray
    'refrigerator': [128, 128, 128], # Gray
    
    # Other Objects
    'book': [139, 69, 19],      # Brown
    'clock': [128, 128, 128],   # Gray
    'vase': [0, 255, 0],        # Green
    'scissors': [128, 128, 128], # Gray
    'teddy bear': [139, 69, 19], # Brown
    'hair drier': [128, 128, 128], # Gray
    'toothbrush': [128, 128, 128], # Gray
}

# Color Categories for UI Organization
COLOR_CATEGORIES = {
    'People & Animals': ['person', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
    'Vehicles': ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat'],
    'Traffic Objects': ['traffic light', 'fire hydrant', 'stop sign', 'parking meter'],
    'Furniture': ['chair', 'couch', 'bed', 'dining table', 'bench'],
    'Electronics': ['tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone'],
    'Kitchen Items': ['bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl'],
    'Food': ['banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake'],
    'Sports': ['frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket'],
    'Accessories': ['backpack', 'umbrella', 'handbag', 'tie', 'suitcase'],
    'Appliances': ['microwave', 'oven', 'toaster', 'sink', 'refrigerator'],
    'Other': ['book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'toilet', 'potted plant']
}

# WebRTC Configuration
WEBRTC_CONFIG = {
    'ice_servers': [{"urls": ["stun:stun.l.google.com:19302"]}],
    'media_constraints': {"video": True, "audio": False},
    'async_processing': True,
}

# File Upload Configuration
UPLOAD_CONFIG = {
    'allowed_types': ['mp4', 'avi', 'mov', 'mkv'],
    'max_file_size': 100 * 1024 * 1024,  # 100MB
    'temp_dir': None,  # Use system temp directory
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'app.log',
}

# Feature Flags
FEATURES = {
    'enable_webcam': True,
    'enable_video_upload': True,
    'enable_color_customization': True,
    'enable_performance_settings': True,
    'enable_screenshot': True,
    'enable_progress_tracking': True,
}

# Performance Presets
PERFORMANCE_PRESETS = {
    'real_time': {
        'frame_skip': 1,
        'confidence_threshold': 0.3,
        'max_fps': 30,
    },
    'balanced': {
        'frame_skip': 2,
        'confidence_threshold': 0.5,
        'max_fps': 20,
    },
    'performance': {
        'frame_skip': 3,
        'confidence_threshold': 0.7,
        'max_fps': 15,
    },
    'quality': {
        'frame_skip': 1,
        'confidence_threshold': 0.8,
        'max_fps': 10,
    }
}


