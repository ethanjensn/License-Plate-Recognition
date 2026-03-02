import os
import sys

# ============================================================
# OCR WORKER MODE - runs when spawned as subprocess
# This block must be at the very top before any torch imports
# ============================================================
if os.environ.get("RUN_AS_OCR_WORKER") == "1":
    import pickle
    import numpy as np
    import cv2
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(lang='en', use_angle_cls=True, use_gpu=False, show_log=False)

    sys.stdout.buffer.write(b"READY\n")
    sys.stdout.buffer.flush()

    while True:
        try:
            length_bytes = sys.stdin.buffer.read(4)
            if not length_bytes or len(length_bytes) < 4:
                break
            length = int.from_bytes(length_bytes, 'little')
            data = sys.stdin.buffer.read(length)
            if not data:
                break

            img_bytes = pickle.loads(data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            result = ocr.ocr(img, cls=False) if img is not None and img.size > 0 else None

            response = pickle.dumps(result)
            sys.stdout.buffer.write(len(response).to_bytes(4, 'little'))
            sys.stdout.buffer.write(response)
            sys.stdout.buffer.flush()

        except Exception:
            response = pickle.dumps(None)
            sys.stdout.buffer.write(len(response).to_bytes(4, 'little'))
            sys.stdout.buffer.write(response)
            sys.stdout.buffer.flush()

    sys.exit(0)

# ============================================================
# SERVER MODE - normal startup from here down
# ============================================================
import subprocess
import pickle
import threading

import torch
from ultralytics import YOLO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import time
import logging

logging.getLogger("ppocr").setLevel(logging.WARNING)

# ==== CONFIGURATION ====
YOLO_MODEL_PATH = 'best.pt'
USE_GPU = True
CONFIDENCE_THRESHOLD = 0.3

app = FastAPI()

# ==== GLOBALS ====
detection_model = None
ocr_process = None
plate_tracker = {}
tracker_counter = 0


def send_ocr_request(plate_crop: np.ndarray):
    """Send image to OCR worker process and get result."""
    global ocr_process
    try:
        _, buf = cv2.imencode('.jpg', plate_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        payload = pickle.dumps(buf.tobytes())

        ocr_process.stdin.write(len(payload).to_bytes(4, 'little'))
        ocr_process.stdin.write(payload)
        ocr_process.stdin.flush()

        length_bytes = ocr_process.stdout.read(4)
        if not length_bytes:
            return None
        length = int.from_bytes(length_bytes, 'little')
        response_data = ocr_process.stdout.read(length)
        return pickle.loads(response_data)

    except Exception as e:
        print(f"[OCR Worker Error] {e}")
        return None


def start_ocr_worker():
    """Spawn this same script as the OCR worker subprocess."""
    global ocr_process
    ocr_env = os.environ.copy()
    ocr_env["RUN_AS_OCR_WORKER"] = "1"

    ocr_process = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=ocr_env
    )

    ready_line = [b""]
    ready_event = threading.Event()

    def read_ready():
        ready_line[0] = ocr_process.stdout.readline()
        ready_event.set()

    t = threading.Thread(target=read_ready, daemon=True)
    t.start()
    ready_event.wait(timeout=120)

    if ready_line[0].strip() == b"READY":
        print("[INFO] OCR worker process ready.")
    else:
        stderr_out = ocr_process.stderr.read(4096).decode(errors='replace')
        raise RuntimeError(f"OCR worker failed to start.\n{stderr_out}")


@app.on_event("startup")
async def startup_event():
    global detection_model
    print("[INFO] Loading models on startup...")
    try:
        start_ocr_worker()

        detection_model = YOLO(YOLO_MODEL_PATH)
        if USE_GPU and torch.cuda.is_available():
            detection_model.to('cuda')
            print("[INFO] YOLO running on GPU.")
        else:
            detection_model.to('cpu')
            print("[INFO] YOLO running on CPU.")

        print("[INFO] All models loaded successfully!")

    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        raise RuntimeError(f"Failed to load models: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    global ocr_process
    if ocr_process:
        ocr_process.terminate()


def track_plate(x1, y1, x2, y2, text, confidence):
    global plate_tracker, tracker_counter
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    matched_id = None
    for plate_id, plate_data in plate_tracker.items():
        existing_center_x = (plate_data['bbox'][0] + plate_data['bbox'][2]) / 2
        existing_center_y = (plate_data['bbox'][1] + plate_data['bbox'][3]) / 2

        if ((center_x - existing_center_x) ** 2 + (center_y - existing_center_y) ** 2) ** 0.5 < 150:
            matched_id = plate_id
            break

    if matched_id is not None:
        if confidence > plate_tracker[matched_id]['confidence']:
            plate_tracker[matched_id] = {'bbox': [int(x1), int(y1), int(x2), int(y2)], 'text': str(text), 'confidence': float(confidence), 'last_seen': time.time()}
        else:
            plate_tracker[matched_id]['bbox'] = [int(x1), int(y1), int(x2), int(y2)]
            plate_tracker[matched_id]['last_seen'] = time.time()
    else:
        tracker_counter += 1
        matched_id = tracker_counter
        plate_tracker[matched_id] = {'bbox': [int(x1), int(y1), int(x2), int(y2)], 'text': str(text), 'confidence': float(confidence), 'last_seen': time.time()}

    return plate_tracker[matched_id]


def cleanup_old_plates():
    global plate_tracker
    current_time = time.time()
    plates_to_remove = [pid for pid, data in plate_tracker.items() if current_time - data['last_seen'] > 1.0]
    for pid in plates_to_remove:
        del plate_tracker[pid]


@app.post("/detect_plates/")
async def detect_plates(file: UploadFile = File(...)):
    if detection_model is None or ocr_process is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        start_time = time.time()
        cleanup_old_plates()

        results = detection_model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD, iou=0.5)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int) if results[0].boxes is not None else []
        confidences = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else []

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            h, w, _ = frame.shape
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            if (x2 - x1) < 20 or (y2 - y1) < 5:
                continue

            plate_crop = frame[y1:y2, x1:x2]
            ocr_text = "DETECTED"
            ocr_confidence = float(confidences[i]) if i < len(confidences) else 0.5

            if plate_crop.size > 0 and plate_crop.shape[0] > 8 and plate_crop.shape[1] > 20:
                ocr_result = send_ocr_request(plate_crop)

                valid_results = []
                if ocr_result and ocr_result[0]:
                    for line in ocr_result[0]:
                        bbox = line[0]  # 4 corner points [[x,y],[x,y],[x,y],[x,y]]
                        text = line[1][0].strip().upper()
                        conf = line[1][1]

                        # Strip non-alphanumeric characters (star symbols, dots, etc.)
                        import re
                        text = re.sub(r'[^A-Z0-9]', '', text)

                        # Calculate character height from bounding box
                        char_height = (abs(bbox[2][1] - bbox[0][1]) + abs(bbox[3][1] - bbox[1][1])) / 2

                        # Must be at least 25% of plate height (filters state name, subtitles)
                        plate_h = plate_crop.shape[0]
                        if char_height < plate_h * 0.25:
                            continue
                        if not (4 <= len(text) <= 9):  # 9 allows TX2000 + extra
                            continue
                        # Skip common header/footer words that appear on plates
                        skip_words = {"OHIO", "FLORIDA", "TEXAS", "CALIFORNIA", "MICHIGAN",
                                      "INDIANA", "ILLINOIS", "GEORGIA", "VIRGINIA", "DEALER",
                                      "STATE", "TRUCK", "APPORT", "TRANSIT", "VANITY", "CITY",
                                      "THELON", "LONESTA", "THELON", "STARTE"}
                        if text in skip_words:
                            continue

                        valid_results.append((text, float(conf), char_height))

                if valid_results:
                    # Biggest text wins - plate numbers are always the largest characters
                    valid_results.sort(key=lambda x: x[2], reverse=True)

                    # Try to merge fragments on the same horizontal line
                    # (happens when plate is large and OCR splits TX / 2000 separately)
                    if len(valid_results) > 1:
                        tallest_height = valid_results[0][2]
                        # Collect all results within 30% height of the tallest
                        same_line = [r for r in valid_results if r[2] >= tallest_height * 0.7]
                        if len(same_line) > 1:
                            # Sort by x-position of their bbox to merge left-to-right
                            # We don't have bbox here so just concatenate by confidence order
                            merged = ''.join(r[0] for r in same_line)
                            if 4 <= len(merged) <= 10:
                                ocr_text = merged
                                ocr_confidence = min(r[1] for r in same_line)
                            else:
                                ocr_text = valid_results[0][0]
                                ocr_confidence = valid_results[0][1]
                        else:
                            ocr_text = valid_results[0][0]
                            ocr_confidence = valid_results[0][1]
                    else:
                        ocr_text = valid_results[0][0]
                        ocr_confidence = valid_results[0][1]

            track_plate(x1, y1, x2, y2, ocr_text, ocr_confidence)

        detected_plates_data = [
            {"bbox": [int(x) for x in data['bbox']], "text": str(data['text']), "confidence": float(data['confidence'])}
            for pid, data in plate_tracker.items()
        ]

        return JSONResponse(content={
            "detections": detected_plates_data,
            "server_processing_time": time.time() - start_time
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)