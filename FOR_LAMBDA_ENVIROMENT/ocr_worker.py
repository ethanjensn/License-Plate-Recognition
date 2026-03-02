"""
Isolated PaddleOCR worker process.
Reads pickled image bytes from stdin, writes pickled OCR results to stdout.
Runs in its own process so paddle never conflicts with torch.
"""
import sys
import pickle
import numpy as np
import cv2
from paddleocr import PaddleOCR

# Initialize once at startup
ocr = PaddleOCR(lang='en', use_angle_cls=True, use_gpu=False, show_log=False)

# Signal ready
sys.stdout.buffer.write(b"READY\n")
sys.stdout.buffer.flush()

while True:
    try:
        # Read length-prefixed message: 4-byte little-endian length + data
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

        if img is None or img.size == 0:
            result = None
        else:
            result = ocr.ocr(img, cls=False)

        response = pickle.dumps(result)
        sys.stdout.buffer.write(len(response).to_bytes(4, 'little'))
        sys.stdout.buffer.write(response)
        sys.stdout.buffer.flush()

    except Exception as e:
        # Send None on error so server doesn't hang
        response = pickle.dumps(None)
        sys.stdout.buffer.write(len(response).to_bytes(4, 'little'))
        sys.stdout.buffer.write(response)
        sys.stdout.buffer.flush()
