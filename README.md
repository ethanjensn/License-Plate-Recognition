# License Plate Recognition (LPR) System

This directory contains a real-time license plate recognition system with separate server and client components.

## Server Component

### Files
- `lambda_server.py` - FastAPI server for license plate detection and OCR
- `lambda_server_requirements.txt` - Server dependencies

### Description
The server component provides a REST API endpoint for license plate detection and text recognition. It uses:

- **YOLOv8** for license plate detection
- **EasyOCR** for text recognition
- **FastAPI** for the web API
- **Real-time processing** with plate tracking

### Features
- GPU acceleration support (CUDA)
- Plate tracking across frames
- Confidence-based result filtering
- Real-time processing optimization
- PIL ANTIALIAS deprecation fix

### API Endpoint
- `POST /detect_plates/` - Accepts image files and returns detected plates

### Response Format
```json
{
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "text": "ABC123",
      "confidence": 0.85
    }
  ],
  "server_processing_time": 0.045
}
```

### Installation (Linux/Lambda Environment)
```bash
# Install server dependencies in Lambda environment
pip install -r lambda_server_requirements.txt
```

### Usage
```bash
python lambda_server.py
```

## Client Component

### Files
- `client_app.py` - Real-time camera client
- `client_app_requirements.txt` - Client dependencies

### Description
The client component captures video from a camera and sends frames to the server for processing. It features:

- **Real-time video capture** using OpenCV
- **HTTP communication** with the server
- **Frame rate optimization** (processes every 3rd frame)
- **Connection pooling** and retry logic
- **Visual display** of detections

### Features
- Multi-threaded architecture
- Network error handling and recovery
- Frame queue management
- Real-time visualization
- Configurable camera settings

### Configuration
- Server URL: `http://<replace with your IP>:5000/detect_plates/`
- Resolution: 1280x720 (720p) (configurable)
- JPEG Quality: 100% (configurable)
- Camera ID: 1 (configurable)

**Configuration Code Snippet:**
```python
# --- Configuration ---
SERVER_URL = "http://<replace with your IP>:5000/detect_plates/"
RESIZE_DIM = (1280, 720) # 720p resolution
JPEG_QUALITY = 100 # 100% quality for best OCR

# Camera settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

### Installation (Client Side)
```bash
# Install client dependencies
pip install -r client_app_requirements.txt
```

### Usage
```bash
python client_app.py
```

### Controls
- Press 'q' to quit the application

## System Architecture

```
Camera → Client → HTTP → Server → YOLO + EasyOCR → Results → Display
```

1. **Client** captures frames from camera
2. **Client** resizes and encodes frames as JPEG
3. **Client** sends frames to server via HTTP POST
4. **Server** detects license plates using YOLOv8
5. **Server** performs OCR using EasyOCR
6. **Server** returns detection results
7. **Client** displays results with bounding boxes

## Performance Optimizations

### Server
- GPU acceleration for YOLO and EasyOCR
- Plate tracking to maintain best results
- Optimized OCR parameters for speed
- Connection pooling and session management

### Client
- Frame skipping (every 3rd frame)
- Non-blocking queues
- Connection retry logic
- Minimal buffer sizes for real-time performance

## Requirements

### Server Dependencies
- FastAPI
- OpenCV
- NumPy
- Ultralytics (YOLOv8)
- EasyOCR
- PyTorch
- Pillow
- Python-multipart

### Client Dependencies
- OpenCV
- Requests
- urllib3

## Notes
- Ensure the server is running before starting the client
- Update the SERVER_URL in the client for your specific server
- The system is optimized for real-time performance
- GPU support provides significant speed improvements 
