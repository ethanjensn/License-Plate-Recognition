# client_app2_http.py
# python LAMBDA\GEMINI\client_app2_http.py
# HTTP version that works with lambda_server3.py
# .\lpr_env_laptop\Scripts\Activate.ps1

import cv2
import requests
import json
import time
import threading
import queue
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Configuration ---
# IMPORTANT: Replace with your Lambda Labs server's public IP
SERVER_URL = "http://141.148.35.122:5000/detect_plates/"
RESIZE_DIM = (1280, 720) # 720p resolution
JPEG_QUALITY = 100 # 100% quality for best OCR

# Thread-safe queues
frame_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue(maxsize=2)

def network_worker():
    """Background thread for network processing"""
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
    
    frame_count = 0
    
    while True:
        try:
            # Get frame from queue
            frame = frame_queue.get()
            frame_count += 1
            
            # Skip frames to maintain performance
            if frame_count % 3 == 0:  # Process every 3rd frame
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
                    
                    # Put result in queue, replace if full
                    try:
                        result_queue.put_nowait(results)
                    except queue.Full:
                        try:
                            result_queue.get_nowait()  # Remove old result
                        except queue.Empty:
                            pass
                        result_queue.put_nowait(results)
                else:
                    print(f"Server error: {response.status_code}")
                    
        except Exception as e:
            print(f"Network error: {e}")
            # Continue processing
            time.sleep(0.1)  # Brief pause on error

# --- Main Application Logic ---
if __name__ == "__main__":
    # --- OpenCV Video Capture and Display Logic ---
    cap = cv2.VideoCapture(1)
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

    latest_results = {"detections": []}
    
    # Start network worker thread
    network_thread = threading.Thread(target=network_worker, daemon=True)
    network_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Send frame to network thread (non-blocking)
        try:
            frame_queue.put_nowait(frame.copy())
        except queue.Full:
            # Skip frame if queue is full
            pass

        # Get latest results from network thread (non-blocking)
        try:
            latest_results = result_queue.get_nowait()
        except queue.Empty:
            # Keep using previous results if no new ones
            pass

        # Draw detections from latest results
        if "detections" in latest_results:
            # Get current frame dimensions for scaling
            frame_h, frame_w = frame.shape[:2]
            
            for detection in latest_results["detections"]:
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