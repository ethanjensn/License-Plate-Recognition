import argparse
import cv2
import json
import time
import threading
from websocket import create_connection, WebSocketTimeoutException

# --- Configuration ---
SERVER_URL = "ws://127.0.0.1:5000/ws/detect"
RESIZE_DIM = (960, 540) # 720p resolution
JPEG_QUALITY = 100 # 100% quality for best OCR


def draw_detection_overlay(frame, detections):
    frame_h, frame_w = frame.shape[:2]
    scale_x = frame_w / RESIZE_DIM[0]
    scale_y = frame_h / RESIZE_DIM[1]

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        text = detection["text"]
        confidence = detection["confidence"]

        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        cv2.rectangle(frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (0, 255, 0), 2)

        label = f"{text} ({confidence:.2f})"
        label_width = max(140, (len(label) * 10) + 10)
        label_top = max(0, y1_scaled - 20)
        cv2.rectangle(frame, (x1_scaled, label_top), (x1_scaled + label_width, y1_scaled), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1_scaled + 5, y1_scaled - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Shared frame/result state (no queueing)
frame_condition = threading.Condition()
frame_version = 0
shared_frame = None

result_lock = threading.Lock()
latest_results = {"detections": []}

processed_condition = threading.Condition()
processed_version = 0
processed_detections = []


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time license plate detection client")
    parser.add_argument(
        "--video",
        type=str,
        help="Path to a video file to run detection on instead of the live camera feed",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index to open when no video is supplied (default: 0)",
    )
    return parser.parse_args()

def network_worker():
    """Background thread for WebSocket metadata streaming"""
    global latest_results, processed_detections, processed_version
    last_processed_frame_version = 0
    ws = None
    
    while True:
        try:
            if ws is None:
                ws = create_connection(SERVER_URL, timeout=3)

            # Wait for a new frame (always process the most recent one)
            with frame_condition:
                frame_condition.wait_for(lambda: frame_version > last_processed_frame_version)
                frame = shared_frame.copy() if shared_frame is not None else None
                current_frame_version = frame_version

            if frame is None:
                continue

            last_processed_frame_version = current_frame_version

            # 1. Resize the frame
            resized_frame = cv2.resize(frame, RESIZE_DIM)

            # 2. Encode with JPEG quality
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            _, buffer = cv2.imencode('.jpg', resized_frame, encode_param)

            ws.send_binary(buffer.tobytes())
            try:
                message = ws.recv()
            except WebSocketTimeoutException:
                continue

            results = json.loads(message)
            detections = results.get('detections', [])
            print(f"Received {len(detections)} detections")
            with result_lock:
                latest_results = results
            with processed_condition:
                processed_detections = [
                    {
                        "bbox": det["bbox"][:],
                        "text": det["text"],
                        "confidence": det["confidence"],
                    }
                    for det in detections
                ]
                processed_version += 1
                processed_condition.notify()
                    
        except Exception as e:
            print(f"Network error: {e}")
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass
                ws = None
            # Continue processing
            time.sleep(0.1)  # Brief pause on error

# --- Main Application Logic ---
if __name__ == "__main__":
    args = parse_args()

    # --- OpenCV Video Capture and Display Logic ---
    frame_interval = None
    last_frame_time = None

    if args.video:
        print(f"Using video file: {args.video}")
        cap = cv2.VideoCapture(args.video)
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if not native_fps or native_fps <= 0:
            native_fps = 30.0
        frame_interval = 1.0 / native_fps
        last_frame_time = time.time()
        print(f"Detected video FPS: {native_fps:.2f}")
    else:
        print(f"Using camera index {args.camera_index}")
        cap = cv2.VideoCapture(args.camera_index)

    if not cap.isOpened():
        source = f"video '{args.video}'" if args.video else f"camera index {args.camera_index}"
        print(f"Error: Could not open {source}.")
        exit()

    if not args.video:
        # Set camera properties for consistent performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)  # Set to 30 FPS
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for real-time

        # Wait a moment for camera to initialize
        time.sleep(1)

        print(
            "Camera initialized: "
            f"{cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}"
            f" @ {cap.get(cv2.CAP_PROP_FPS)} FPS"
        )

    stop_event = threading.Event()

    def capture_worker():
        global shared_frame, frame_version
        local_last_frame_time = time.time()
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                if args.video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            with frame_condition:
                shared_frame = frame.copy()
                frame_version += 1
                frame_condition.notify()

            if args.video and frame_interval:
                elapsed = time.time() - local_last_frame_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                local_last_frame_time = time.time()

        stop_event.set()

    # Start worker threads
    network_thread = threading.Thread(target=network_worker, daemon=True)
    network_thread.start()

    capture_thread = threading.Thread(target=capture_worker, daemon=True)
    capture_thread.start()

    displayed_frame_version = 0
    latest_displayed_detection_version = 0
    latest_display_detections = []

    while not stop_event.is_set():
        with frame_condition:
            frame_condition.wait_for(lambda: frame_version > displayed_frame_version or stop_event.is_set())
            if stop_event.is_set() and shared_frame is None:
                break
            frame_to_show = None if shared_frame is None else shared_frame.copy()
            displayed_frame_version = frame_version

        if frame_to_show is None:
            continue

        with processed_condition:
            if processed_version > latest_displayed_detection_version:
                latest_display_detections = [
                    {
                        "bbox": det["bbox"][:],
                        "text": det["text"],
                        "confidence": det["confidence"],
                    }
                    for det in processed_detections
                ]
                latest_displayed_detection_version = processed_version

        if latest_display_detections:
            draw_detection_overlay(frame_to_show, latest_display_detections)

        cv2.imshow('Real-time License Plate Detection', frame_to_show)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    stop_event.set()
    capture_thread.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()
