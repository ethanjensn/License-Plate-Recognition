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
import re
import asyncio
import json

import torch
from ultralytics import YOLO

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi_mcp import FastApiMCP
import httpx
import cv2
import numpy as np
import time
import logging

logging.getLogger("ppocr").setLevel(logging.WARNING)

# ==== CONFIGURATION ====
YOLO_MODEL_PATH = 'FOR_SERVER_ENVIROMENT/best.pt'
USE_GPU = True
CONFIDENCE_THRESHOLD = 0.3
OCR_MAX_PER_FRAME = 1  # Limit expensive OCR calls per frame
OCR_REFRESH_INTERVAL = 0.35
OCR_CACHE_TTL = 2.0
TRACK_TTL = 1.0
TRACK_MATCH_DISTANCE = 120.0

# Enterprise REST API Configuration
API_ENDPOINT = os.environ.get("ALPR_API_ENDPOINT", "https://api.enterprise.com/v1/vehicle-logs")
SOURCE_ID = os.environ.get("ALPR_SOURCE_ID", "DETECTION_VALIDATION")
VIDEO_DIR = os.environ.get("ALPR_VIDEO_DIR", "detection-validation")
API_TIMEOUT = 5.0
MIN_CONFIDENCE_FOR_SYNC = 0.7

app = FastAPI(title="ALPR-Detection-Server")

# ==== GLOBALS ====
detection_model = None
ocr_process = None
ocr_cache = {}
ocr_cache_lock = threading.Lock()
ocr_process_lock = threading.Lock()

# HTTP client for REST API calls (initialized on startup)
api_client: httpx.AsyncClient | None = None


def _normalize_text(text: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', text.strip().upper())


def _make_plate_cache_key(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    center_x = x1 + width / 2
    center_y = y1 + height / 2
    return (
        int(center_x / 32),
        int(center_y / 24),
        int(width / 32),
        int(height / 16),
    )


def _get_cached_ocr(cache_key, now: float):
    with ocr_cache_lock:
        cached = ocr_cache.get(cache_key)
        if not cached:
            return None
        if now - cached["updated_at"] > OCR_CACHE_TTL:
            del ocr_cache[cache_key]
            return None
        return dict(cached)


def _set_cached_ocr(cache_key, text: str, confidence: float, now: float):
    with ocr_cache_lock:
        ocr_cache[cache_key] = {
            "text": text,
            "confidence": float(confidence),
            "updated_at": now,
        }


def _prune_ocr_cache(now: float):
    with ocr_cache_lock:
        expired_keys = [
            key for key, value in ocr_cache.items()
            if now - value["updated_at"] > OCR_CACHE_TTL
        ]
        for key in expired_keys:
            del ocr_cache[key]


def _extract_ocr_text(ocr_result, plate_crop: np.ndarray):
    ocr_text = None
    ocr_confidence = None

    valid_results = []
    if ocr_result and ocr_result[0]:
        for line in ocr_result[0]:
            bbox = line[0]
            text = _normalize_text(line[1][0])
            conf = line[1][1]

            char_height = (abs(bbox[2][1] - bbox[0][1]) + abs(bbox[3][1] - bbox[1][1])) / 2

            plate_h = plate_crop.shape[0]
            if char_height < plate_h * 0.25:
                continue
            if not (4 <= len(text) <= 9):
                continue

            skip_words = {"OHIO", "FLORIDA", "TEXAS", "CALIFORNIA", "MICHIGAN",
                          "INDIANA", "ILLINOIS", "GEORGIA", "VIRGINIA", "DEALER",
                          "STATE", "TRUCK", "APPORT", "TRANSIT", "VANITY", "CITY",
                          "THELON", "LONESTA", "THELON", "STARTE"}
            if text in skip_words:
                continue

            valid_results.append((text, float(conf), char_height))

    if valid_results:
        valid_results.sort(key=lambda x: x[2], reverse=True)
        if len(valid_results) > 1:
            tallest_height = valid_results[0][2]
            same_line = [r for r in valid_results if r[2] >= tallest_height * 0.7]
            if len(same_line) > 1:
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

    return ocr_text, ocr_confidence


async def _submit_to_rest_api(plate_text: str, confidence: float) -> bool:
    """Fire-and-forget async submission to enterprise REST API.
    
    Returns True if submission succeeded, False otherwise.
    Defensive error handling: logs failures but never raises.
    """
    if api_client is None:
        print("[REST API] Client not initialized, skipping submission")
        return False
    
    if confidence < MIN_CONFIDENCE_FOR_SYNC:
        print(f"[REST API] Confidence {confidence:.2f} below threshold {MIN_CONFIDENCE_FOR_SYNC}, skipping")
        return False
    
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_id": SOURCE_ID,
        "detection_type": "ALPR",
        "payload": {
            "license_plate": plate_text,
            "confidence_score": round(confidence, 2)
        },
        "network_status": "pending"
    }
    
    try:
        response = await api_client.post(
            API_ENDPOINT,
            json=payload,
            timeout=API_TIMEOUT
        )
        
        if response.status_code == 201:
            print(f"[REST API] Successfully synced plate '{plate_text}' (confidence: {confidence:.2f})")
            return True
        else:
            print(f"[REST API] Failed to sync plate '{plate_text}': HTTP {response.status_code}")
            return False
            
    except httpx.TimeoutException:
        print(f"[REST API] Timeout syncing plate '{plate_text}' after {API_TIMEOUT}s")
        return False
    except httpx.ConnectError as e:
        print(f"[REST API] Connection error syncing plate '{plate_text}': {e}")
        return False
    except Exception as e:
        print(f"[REST API] Unexpected error syncing plate '{plate_text}': {e}")
        return False


def send_ocr_request(plate_crop: np.ndarray):
    """Send image to OCR worker process and get result."""
    global ocr_process
    try:
        _, buf = cv2.imencode('.jpg', plate_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        payload = pickle.dumps(buf.tobytes())

        with ocr_process_lock:
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


def _prune_tracks(track_state: dict, now: float):
    expired_ids = [
        track_id for track_id, track in track_state.items()
        if now - track["last_seen"] > TRACK_TTL
    ]
    for track_id in expired_ids:
        del track_state[track_id]


def _match_track(track_state: dict, bbox, now: float):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    best_track_id = None
    best_distance = None

    for track_id, track in track_state.items():
        prev_x1, prev_y1, prev_x2, prev_y2 = track["bbox"]
        prev_center_x = (prev_x1 + prev_x2) / 2
        prev_center_y = (prev_y1 + prev_y2) / 2
        distance = ((center_x - prev_center_x) ** 2 + (center_y - prev_center_y) ** 2) ** 0.5
        if distance > TRACK_MATCH_DISTANCE:
            continue
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_track_id = track_id

    if best_track_id is not None:
        track_state[best_track_id]["bbox"] = [int(x1), int(y1), int(x2), int(y2)]
        track_state[best_track_id]["last_seen"] = now
        return best_track_id

    next_track_id = max(track_state.keys(), default=0) + 1
    track_state[next_track_id] = {
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "last_seen": now,
        "text": "DETECTED",
        "confidence": 0.0,
        "ocr_updated_at": 0.0,
    }
    return next_track_id


def _run_detection_pipeline(frame: np.ndarray, now: float, track_state: dict):
    _prune_ocr_cache(now)
    _prune_tracks(track_state, now)

    # Force the model and inference to be consistent. 
    # Ultralytics handles this automatically if you pass half=True
    results = detection_model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD, iou=0.5, half=True)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int) if results[0].boxes is not None else []
    confidences = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else []

    detections = []
    ocr_budget = OCR_MAX_PER_FRAME
    sorted_indices = np.argsort(confidences)[::-1] if len(confidences) > 0 else range(len(boxes))
    frame_h, frame_w = frame.shape[:2]

    for idx in sorted_indices:
        x1, y1, x2, y2 = boxes[idx]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_w, x2), min(frame_h, y2)
        if (x2 - x1) < 20 or (y2 - y1) < 5:
            continue

        bbox = [int(x1), int(y1), int(x2), int(y2)]
        track_id = _match_track(track_state, bbox, now)
        track = track_state[track_id]
        plate_crop = frame[y1:y2, x1:x2]

        text = track.get("text", "DETECTED")
        confidence = track.get("confidence", float(confidences[idx]) if idx < len(confidences) else 0.5)
        if not track.get("confidence"):
            track["confidence"] = float(confidence)
        should_refresh_ocr = (
            ocr_budget > 0
            and plate_crop.size > 0
            and plate_crop.shape[0] > 8
            and plate_crop.shape[1] > 20
            and (now - track.get("ocr_updated_at", 0.0)) >= OCR_REFRESH_INTERVAL
        )

        if should_refresh_ocr:
            ocr_result = send_ocr_request(plate_crop)
            ocr_budget -= 1
            refreshed_text, refreshed_confidence = _extract_ocr_text(ocr_result, plate_crop)
            if refreshed_text:
                text = refreshed_text
                confidence = refreshed_confidence
                track["text"] = text
                track["confidence"] = float(confidence)
                track["ocr_updated_at"] = time.time()

        track["text"] = text
        track["confidence"] = float(confidence)

        detections.append({
            "track_id": int(track_id),
            "bbox": bbox,
            "text": str(text),
            "confidence": float(confidence),
            "timestamp": now,
        })

    return detections


@app.on_event("startup")
async def startup_event():
    global detection_model, api_client
    print("[INFO] Loading models on startup...")
    try:
        start_ocr_worker()

        detection_model = YOLO(YOLO_MODEL_PATH)
        if USE_GPU and torch.cuda.is_available():
            detection_model.to('cuda')
            try:
                detection_model.model.half()
                print("[INFO] YOLO running on GPU (FP16).")
            except AttributeError:
                print("[INFO] YOLO running on GPU (FP32 fallback).")
        else:
            detection_model.to('cpu')
            print("[INFO] YOLO running on CPU.")

        # Initialize HTTP client for REST API calls
        api_client = httpx.AsyncClient()
        print(f"[INFO] REST API client initialized (endpoint: {API_ENDPOINT})")
        print(f"[INFO] Video source directory: {VIDEO_DIR}")

        print("[INFO] All models loaded successfully!")

    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        raise RuntimeError(f"Failed to load models: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    global ocr_process, api_client
    if ocr_process:
        ocr_process.terminate()
    if api_client:
        await api_client.aclose()


class ScanVideoRequest(BaseModel):
    video_path: str

@app.post("/scan_video")
async def scan_video(request: ScanVideoRequest):
    video_path = request.video_path
    """MCP-callable endpoint: Process a video file and detect license plates.
    
    - Runs YOLO detection + PaddleOCR on each frame
    - Aggregates unique license plates found
    - Syncs high-confidence detections to enterprise REST API
    - Returns structured results for AI assistant consumption
    
    Video path can be absolute or relative to VIDEO_DIR (detection-validation/).
    Example: "11939614_2160_3840_60fps.mp4" or "C:/full/path/to/video.mp4"
    """
    if detection_model is None or ocr_process is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")
    
    # Resolve video path - check absolute first, then relative to VIDEO_DIR
    resolved_path = video_path
    if not os.path.exists(resolved_path):
        resolved_path = os.path.join(VIDEO_DIR, video_path)
    
    if not os.path.exists(resolved_path):
        raise HTTPException(
            status_code=400, 
            detail=f"Video file not found: '{video_path}' (also checked in '{VIDEO_DIR}/')"
        )
    
    cap = cv2.VideoCapture(resolved_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail=f"Could not open video: {video_path}")
    
    start_time = time.time()
    frame_count = 0
    track_state = {}
    unique_plates = {}  # plate_text -> {"confidence": float, "api_synced": bool}
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            now = time.time()
            
            # Process frame through detection pipeline
            detections = _run_detection_pipeline(frame, now, track_state)
            
            # Collect unique plates and trigger REST API sync
            for det in detections:
                plate_text = det["text"]
                confidence = det["confidence"]
                
                # Track best confidence for each plate
                if plate_text not in unique_plates or unique_plates[plate_text]["confidence"] < confidence:
                    unique_plates[plate_text] = {"confidence": confidence, "api_synced": False}
                    
                    # Fire-and-forget REST API submission for high-confidence plates
                    if confidence >= MIN_CONFIDENCE_FOR_SYNC:
                        asyncio.create_task(_submit_to_rest_api(plate_text, confidence))
                        unique_plates[plate_text]["api_synced"] = True
    
    finally:
        cap.release()
    
    processing_time = time.time() - start_time
    plates_list = list(unique_plates.keys())
    api_synced_count = sum(1 for p in unique_plates.values() if p["api_synced"])
    
    return JSONResponse(content={
        "status": "success",
        "video_path": resolved_path,
        "plates_detected": plates_list,
        "total_frames_processed": frame_count,
        "api_synced": api_synced_count > 0,
        "api_synced_count": api_synced_count,
        "processing_time_seconds": round(processing_time, 2)
    })


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
        detected_plates_data = _run_detection_pipeline(frame, start_time, {})

        return JSONResponse(content={
            "detections": detected_plates_data,
            "server_processing_time": time.time() - start_time
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.websocket("/ws/detect")
async def detect_stream(websocket: WebSocket):
    if detection_model is None or ocr_process is None:
        await websocket.close(code=1013)
        return

    await websocket.accept()
    track_state = {}

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            frame_bytes = message.get("bytes")
            if not frame_bytes:
                text_payload = message.get("text")
                if text_payload == "ping":
                    await websocket.send_text("pong")

                    continue
                continue

            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            started_at = time.time()
            detections = _run_detection_pipeline(frame, started_at, track_state)
            payload = {
                "type": "detections",
                "timestamp": started_at,
                "server_processing_time": time.time() - started_at,
                "detections": detections,
            }
            await websocket.send_text(json.dumps(payload))
    except WebSocketDisconnect:
        return
    except Exception:
        await websocket.close(code=1011)


# Mount MCP layer after all routes are defined so fastapi-mcp picks them all up
mcp = FastApiMCP(app)
mcp.mount_http()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)