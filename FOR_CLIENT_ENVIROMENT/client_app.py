import cv2
import requests
import json
import time
import threading
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Configuration ---
# IMPORTANT: Replace with your Lambda Labs server's public IP
SERVER_URL = "http://127.0.0.1:5000/detect_plates/"
RESIZE_DIM = (1280, 720) # 720p resolution
JPEG_QUALITY = 100 # 100% quality for best OCR

# Shared frame/result state (no queueing)
frame_condition = threading.Condition()
frame_version = 0
shared_frame = None

result_lock = threading.Lock()
latest_results = {"detections": []}

def network_worker():
    """Background thread for network processing"""
    global latest_results
    # Create session with connection pooling and retries
    session = requests.Session()
    retry_strategy = Retry(
        total=5,  # More retries
        backoff_factor=0.1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST", "GET"]  # Allow POST retries
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy, 
        pool_connections=20,  # More connections
        pool_maxsize=20,      # Larger pool
        pool_block=False      # Don't block when pool is full
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    processed_version = 0
    
    while True:
        try:
            # Wait for a new frame (always process the most recent one)
            with frame_condition:
                frame_condition.wait_for(lambda: frame_version > processed_version)
                frame = shared_frame.copy() if shared_frame is not None else None
                processed_version = frame_version

            if frame is None:
                continue

            # 1. Resize the frame
            resized_frame = cv2.resize(frame, RESIZE_DIM)

            # 2. Encode with JPEG quality
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            _, buffer = cv2.imencode('.jpg', resized_frame, encode_param)

            # 3. Send HTTP POST request with better error handling
            files = {'file': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}
            try:
                response = session.post(SERVER_URL, files=files, timeout=3.0)  # Longer timeout
            except requests.exceptions.ConnectionError:
                # Recreate session on connection error
                session = requests.Session()
                adapter = HTTPAdapter(
                    max_retries=retry_strategy, 
                    pool_connections=20,
                    pool_maxsize=20,
                    pool_block=False
                )
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                response = session.post(SERVER_URL, files=files, timeout=3.0)
            
            if response.status_code == 200:
                results = response.json()
                print(f"Received {len(results.get('detections', []))} detections")
                with result_lock:
                    latest_results = results
            else:
                print(f"Server error: {response.status_code}")
                    
        except Exception as e:
            print(f"Network error: {e}")
            # Continue processing
            time.sleep(0.1)  # Brief pause on error

# --- Main Application Logic ---
if __name__ == "__main__":
    # --- OpenCV Video Capture and Display Logic ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()
    
    # Set camera properties for consistent performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set to 30 FPS
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for real-time
    
    # Wait a moment for camera to initialize
    time.sleep(1)
    
    print(f"Camera initialized: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} @ {cap.get(cv2.CAP_PROP_FPS)} FPS")

    # Start network worker thread
    network_thread = threading.Thread(target=network_worker, daemon=True)
    network_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Publish frame to the network worker (always overwrite with latest frame)
        with frame_condition:
            shared_frame = frame.copy()
            frame_version += 1
            frame_condition.notify()

        # Snapshot latest detections without blocking rendering
        with result_lock:
            detections_snapshot = [
                {
                    "bbox": det["bbox"][:],
                    "text": det["text"],
                    "confidence": det["confidence"],
                }
                for det in latest_results.get("detections", [])
            ]

        # Draw detections from latest results
        if detections_snapshot:
            # Get current frame dimensions for scaling
            frame_h, frame_w = frame.shape[:2]
            
            for detection in detections_snapshot:
                x1, y1, x2, y2 = detection['bbox']
                text = detection['text']
                confidence = detection['confidence']
                
                # Scale coordinates from resized image to current frame size
                scale_x = frame_w / RESIZE_DIM[0]
                scale_y = frame_h / RESIZE_DIM[1]
                
                x1_scaled = int(x1 * scale_x)
                y1_scaled = int(y1 * scale_y)
                x2_scaled = int(x2 * scale_x)
                y2_scaled = int(y2 * scale_y)
                
                # Draw the bounding box
                cv2.rectangle(frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (0, 255, 0), 2)
                # Draw the label with confidence
                label = f"{text} ({confidence:.2f})"
                cv2.putText(frame, label, (x1_scaled, y1_scaled - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)



        # Show the final frame
        cv2.imshow('Real-time License Plate Detection', frame)

        # Exit on 'q' key (faster key check)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()

    cv2.destroyAllWindows() 
