from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import time
import base64
import io
import torch
from PIL import Image
import threading
import queue

# Fix for PIL ANTIALIAS deprecation
try:
    from PIL import Image
    # Try to import Resampling (newer Pillow versions)
    from PIL import Image as PILImage
    if hasattr(PILImage, 'Resampling'):
        # Use the new Resampling enum
        ANTIALIAS = PILImage.Resampling.LANCZOS
        # Monkey patch to fix EasyOCR compatibility
        PILImage.ANTIALIAS = PILImage.Resampling.LANCZOS
    else:
        # Fallback for older versions
        ANTIALIAS = PILImage.ANTIALIAS
except ImportError:
    # If PIL is not available, use a default value
    ANTIALIAS = None

# ==== CONFIGURATION (Server-side) ====
# IMPORTANT: Adjust this path on your Lambda Labs instance
YOLO_MODEL_PATH = 'best.pt' # e.g., /app/models/best.pt or /mnt/data/models/best.pt

USE_GPU = True  # Lambda Labs instances will have GPUs
CONFIDENCE_THRESHOLD = 0.3  # Lower threshold to trust the model
OCR_CROP_WIDTH = 150  # Smaller crop for speed

# Performance settings
# FRAME_SKIP will be handled by the client sending fewer frames

app = FastAPI()

# ==== LOAD MODELS (on server startup) ====
detection_model = None
ocr_reader = None

# Plate tracking system
plate_tracker = {}  # Track plates by position and maintain best results
tracker_counter = 0  # Unique ID counter for plates

@app.on_event("startup")
async def startup_event():
    global detection_model, ocr_reader
    print("[INFO] Loading models on startup...")
    try:
        # Load YOLOv8 detection model
        detection_model = YOLO(YOLO_MODEL_PATH)
        if USE_GPU and torch.cuda.is_available():
            detection_model.to('cuda')
            print("[INFO] YOLO model moved to CUDA.")
        else:
            detection_model.to('cpu')
            print("[INFO] YOLO model running on CPU (CUDA not available or not used).")


        # Load EasyOCR reader
        # EasyOCR needs to know if it should use GPU at initialization
        ocr_reader = easyocr.Reader(['en'], gpu=USE_GPU and torch.cuda.is_available())
        print("[INFO] EasyOCR reader initialized.")

        print("[INFO] Models loaded successfully!")
        print("[INFO] Running in PURE REAL-TIME mode with plate tracking!")
        
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        raise RuntimeError(f"Failed to load models: {e}")

def track_plate(x1, y1, x2, y2, text, confidence):
    """Track plates and maintain highest confidence results"""
    global plate_tracker, tracker_counter
    
    # Calculate center point of detection
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Check if this detection matches any existing tracked plate
    matched_id = None
    for plate_id, plate_data in plate_tracker.items():
        # Calculate distance between centers
        existing_center_x = (plate_data['bbox'][0] + plate_data['bbox'][2]) / 2
        existing_center_y = (plate_data['bbox'][1] + plate_data['bbox'][3]) / 2
        
        distance = ((center_x - existing_center_x) ** 2 + (center_y - existing_center_y) ** 2) ** 0.5
        
        # If centers are close (within 150 pixels), consider it the same plate
        if distance < 150:
            matched_id = plate_id
            break
    
    if matched_id is not None:
        # Update existing plate with better result if confidence is higher
        existing_confidence = plate_tracker[matched_id]['confidence']
        if confidence > existing_confidence:
            plate_tracker[matched_id] = {
                'bbox': [int(x1), int(y1), int(x2), int(y2)],  # Convert to native Python int
                'text': str(text),  # Convert to string
                'confidence': float(confidence),  # Convert to native Python float
                'last_seen': time.time()
            }
            print(f"[TRACKING] Updated plate {matched_id}: {text} (conf: {confidence:.3f})")
        else:
            # Keep existing result but update position
            plate_tracker[matched_id]['bbox'] = [int(x1), int(y1), int(x2), int(y2)]
            plate_tracker[matched_id]['last_seen'] = time.time()
            print(f"[TRACKING] Kept existing plate {matched_id}: {plate_tracker[matched_id]['text']} (conf: {plate_tracker[matched_id]['confidence']:.3f})")
    else:
        # New plate detected
        tracker_counter += 1
        plate_tracker[tracker_counter] = {
            'bbox': [int(x1), int(y1), int(x2), int(y2)],  # Convert to native Python int
            'text': str(text),  # Convert to string
            'confidence': float(confidence),  # Convert to native Python float
            'last_seen': time.time()
        }
        matched_id = tracker_counter
        print(f"[TRACKING] New plate {matched_id}: {text} (conf: {confidence:.3f})")
    
    return plate_tracker[matched_id]

def cleanup_old_plates():
    """Remove plates that haven't been seen for 3 seconds"""
    global plate_tracker
    current_time = time.time()
    plates_to_remove = []
    
    for plate_id, plate_data in plate_tracker.items():
        if current_time - plate_data['last_seen'] > 1.0:
            plates_to_remove.append(plate_id)
    
    for plate_id in plates_to_remove:
        del plate_tracker[plate_id]

@app.post("/detect_plates/")
async def detect_plates(file: UploadFile = File(...)):
    if detection_model is None or ocr_reader is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Server starting up.")

    try:
        # Read image data
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        start_time = time.time()
        
        # Clean up old plates first
        cleanup_old_plates()
        
        # --- 1. License Plate Detection (ULTRA-FAST MODE) ---
        results = detection_model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD, iou=0.5)  # Lower IOU to allow more detections
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int) if results[0].boxes is not None else []
        confidences = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else []

        # --- 2. PURE REAL-TIME DETECTION + OCR with TRACKING ---
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            # Validate coordinates
            h, w, _ = frame.shape
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # Trust the model - only basic validation
            box_width = x2 - x1
            box_height = y2 - y1
            
            # Only skip if box is extremely small (likely noise)
            if box_width < 20 or box_height < 5:
                continue

            # Crop the license plate
            plate_crop = frame[y1:y2, x1:x2]

            # PURE REAL-TIME OCR (no delays, no queues)
            ocr_text = "DETECTED"
            ocr_confidence = float(confidences[i]) if i < len(confidences) else 0.5
            ocr_confidence = float(ocr_confidence)  # Ensure it's native Python float
            
            if plate_crop.size > 0 and plate_crop.shape[0] > 8 and plate_crop.shape[1] > 20:
                try:
                    # Fast OCR preprocessing
                    crop_resized = cv2.resize(plate_crop, (100, 30))  # Smaller for speed
                    gray_crop = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
                    gray_crop = cv2.convertScaleAbs(gray_crop, alpha=1.0, beta=0)  # Minimal processing
                    
                    # Single OCR attempt with balanced parameters
                    ocr_result = ocr_reader.readtext(
                        gray_crop,
                        detail=1,
                        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                        paragraph=False,
                        height_ths=0.1,  # Very lenient for speed
                        width_ths=0.1,   # Very lenient for speed
                        text_threshold=0.1,  # Very lenient for speed
                        link_threshold=0.05,  # Very lenient for speed
                        add_margin=0.005  # Minimal margin for speed
                    )
                    
                    # Collect top 3 OCR results
                    ocr_results = []
                    for (_, text, confidence) in ocr_result:
                        if len(text.strip()) >= 2:
                            ocr_results.append((text.strip().upper(), float(confidence)))
                    
                    # Sort by confidence and take top 3
                    ocr_results.sort(key=lambda x: x[1], reverse=True)
                    top_results = ocr_results[:3]
                    
                    if top_results:
                        # Combine top 3 results with commas
                        ocr_text = ", ".join([text for text, _ in top_results])
                        ocr_confidence = top_results[0][1]  # Use highest confidence
                    else:
                        ocr_text = "DETECTED"
                        ocr_confidence = 0.5
                            
                except Exception as e:
                    ocr_text = "ERROR"
            
            # Track the plate and get best result
            tracked_plate = track_plate(x1, y1, x2, y2, ocr_text, ocr_confidence)
        
        # Return all currently tracked plates
        detected_plates_data = []
        for plate_id, plate_data in plate_tracker.items():
            detected_plates_data.append({
                "bbox": [int(x) for x in plate_data['bbox']],  # Convert to native Python int
                "text": str(plate_data['text']),  # Convert to string
                "confidence": float(plate_data['confidence'])  # Convert to native Python float
            })

        elapsed_time = time.time() - start_time
        print(f"[SERVER INFO] Processed frame in {elapsed_time:.4f} seconds. Detections: {len(detected_plates_data)}")

        return JSONResponse(content={"detections": detected_plates_data, "server_processing_time": elapsed_time})

    except Exception as e:
        print(f"[SERVER ERROR] An error occurred during frame processing: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
