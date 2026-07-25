"""ALPR Performance Evaluator

Measures the four metric categories the user asked for:
1. Speed / Throughput  (latency per frame, FPS)
2. Detection Accuracy   (precision / recall / F1 / AP vs YOLO labels)
3. Tracking Quality   (internal tracker statistics from sequential frames)
4. Resource Utilization (CPU / RAM / GPU samples via psutil / pynvml)

The script deliberately replicates the logic in
FOR_SERVER_ENVIROMENT/detection_server.py so the numbers reflect the real
pipeline (YOLO + custom distance-based tracker).  OCR is optional because the
provided labels only contain plate bounding boxes, not text.

Example:
    python evaluate_alpr.py --mode all \
        --dataset-root "detection-validation/license plate recognition.v2/license plate recognition.v2" \
        --splits valid test \
        --frames-dir "detection-validation/extracted_frames" \
        --output results.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import threading
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

# Optional monitoring libraries ------------------------------------------------
try:
    import psutil
except Exception:
    psutil = None

try:
    import pynvml
except Exception:
    pynvml = None

# Optional OCR library ---------------------------------------------------------
try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

# Same constants used by detection_server.py ------------------------------------
CONFIDENCE_THRESHOLD = 0.3
OCR_MAX_PER_FRAME = 1
OCR_REFRESH_INTERVAL = 0.35
OCR_CACHE_TTL = 2.0
TRACK_TTL = 1.0
TRACK_MATCH_DISTANCE = 120.0


def _normalize_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.strip().upper())


def _extract_ocr_text(ocr_result: Any, plate_crop: np.ndarray) -> tuple[str | None, float | None]:
    """Identical filtering logic to detection_server.py."""
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
            skip_words = {
                "OHIO", "FLORIDA", "TEXAS", "CALIFORNIA", "MICHIGAN",
                "INDIANA", "ILLINOIS", "GEORGIA", "VIRGINIA", "DEALER",
                "STATE", "TRUCK", "APPORT", "TRANSIT", "VANITY", "CITY",
                "THELON", "LONESTA", "STARTE",
            }
            if text in skip_words:
                continue
            valid_results.append((text, float(conf), char_height))

    if valid_results:
        valid_results.sort(key=lambda x: x[2], reverse=True)
        if len(valid_results) > 1:
            tallest_height = valid_results[0][2]
            same_line = [r for r in valid_results if r[2] >= tallest_height * 0.7]
            if len(same_line) > 1:
                merged = "".join(r[0] for r in same_line)
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


# Geometry helpers -------------------------------------------------------------
def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Intersection-over-Union for two xyxy boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> list[list[float]]:
    """Load YOLO format labels and return absolute xyxy boxes."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, cx, cy, w, h = parts
            cx, cy, w, h = map(float, (cx, cy, w, h))
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([x1, y1, x2, y2])
    return boxes


def compute_ap(all_preds: list[tuple[float, list[float]]], all_gts: list[list[float]], iou_thresh: float) -> float:
    """Compute Average Precision for a single IoU threshold."""
    if not all_preds or not all_gts:
        return 0.0

    sorted_preds = sorted(all_preds, key=lambda x: x[0], reverse=True)
    matched = set()
    tp = []
    n_gt = len(all_gts)

    for _, pred_box in sorted_preds:
        best_iou = 0.0
        best_idx = -1
        for gi, gt_box in enumerate(all_gts):
            if gi in matched:
                continue
            iou = bbox_iou(np.array(pred_box), np.array(gt_box))
            if iou > best_iou:
                best_iou = iou
                best_idx = gi
        if best_iou >= iou_thresh and best_idx not in matched:
            matched.add(best_idx)
            tp.append(1)
        else:
            tp.append(0)

    tp = np.array(tp, dtype=np.float64)
    fp = 1 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / (tp_cum + fp_cum + 1e-9)

    # Append sentinels and compute AP via Pascal VOC 11-point interpolation.
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx])
    return float(ap)


# Tracker (exact copy of detection_server.py logic) ------------------------------
def prune_tracks(track_state: dict, now: float, ttl: float = TRACK_TTL) -> None:
    expired = [tid for tid, t in track_state.items() if now - t["last_seen"] > ttl]
    for tid in expired:
        del track_state[tid]


def match_track(track_state: dict, bbox: list[int], now: float, max_distance: float = TRACK_MATCH_DISTANCE) -> int:
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    best_tid = None
    best_dist = None

    for tid, track in track_state.items():
        px1, py1, px2, py2 = track["bbox"]
        pcx = (px1 + px2) / 2
        pcy = (py1 + py2) / 2
        dist = math.hypot(center_x - pcx, center_y - pcy)
        if dist > max_distance:
            continue
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_tid = tid

    if best_tid is not None:
        track_state[best_tid]["bbox"] = [int(x1), int(y1), int(x2), int(y2)]
        track_state[best_tid]["last_seen"] = now
        return best_tid

    next_tid = max(track_state.keys(), default=0) + 1
    track_state[next_tid] = {
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "last_seen": now,
        "text": "DETECTED",
        "confidence": 0.0,
        "ocr_updated_at": 0.0,
    }
    return next_tid


def run_detection_pipeline(
    model: YOLO,
    frame: np.ndarray,
    now: float,
    track_state: dict,
    conf: float = CONFIDENCE_THRESHOLD,
    iou: float = 0.5,
    half: bool = True,
    device: str | None = None,
    ocr_engine: Any | None = None,
) -> list[dict]:
    """Run one frame through YOLO + tracker (+ optional OCR)."""
    prune_tracks(track_state, now)

    results = model.predict(
        frame,
        verbose=False,
        conf=conf,
        iou=iou,
        half=half,
        device=device,
    )
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int) if results[0].boxes is not None else np.empty((0, 4))
    confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else np.empty(0)

    detections = []
    frame_h, frame_w = frame.shape[:2]
    sorted_indices = np.argsort(confs)[::-1] if len(confs) > 0 else range(len(boxes))
    ocr_budget = OCR_MAX_PER_FRAME if ocr_engine is not None else 0

    for idx in sorted_indices:
        x1, y1, x2, y2 = boxes[idx]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_w, x2), min(frame_h, y2)
        if (x2 - x1) < 20 or (y2 - y1) < 5:
            continue

        bbox = [int(x1), int(y1), int(x2), int(y2)]
        track_id = match_track(track_state, bbox, now)
        det_conf = float(confs[idx]) if idx < len(confs) else 0.5
        detection = {
            "track_id": int(track_id),
            "bbox": bbox,
            "confidence": det_conf,
            "text": track_state[track_id].get("text", "DETECTED"),
        }

        plate_crop = frame[y1:y2, x1:x2]
        should_refresh = (
            ocr_budget > 0
            and plate_crop.size > 0
            and plate_crop.shape[0] > 8
            and plate_crop.shape[1] > 20
            and (now - track_state[track_id].get("ocr_updated_at", 0.0)) >= OCR_REFRESH_INTERVAL
        )
        if should_refresh:
            ocr_result = ocr_engine.ocr(plate_crop, cls=False)
            ocr_budget -= 1
            refreshed_text, refreshed_conf = _extract_ocr_text(ocr_result, plate_crop)
            if refreshed_text:
                detection["text"] = refreshed_text
                detection["ocr_confidence"] = refreshed_conf
                track_state[track_id]["text"] = refreshed_text
                track_state[track_id]["confidence"] = float(refreshed_conf)
                track_state[track_id]["ocr_updated_at"] = now

        detections.append(detection)
        track_state[track_id]["confidence"] = float(detection.get("confidence", det_conf))

    return detections


# Resource monitor -------------------------------------------------------------
class ResourceMonitor:
    """Samples CPU/RAM/GPU in a background thread while the main loop runs."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._stop = threading.Event()
        self.samples = []
        self._gpu_handle = None
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._gpu_handle = None
        if psutil is not None:
            psutil.cpu_percent(interval=None)  # warm-up

    def start(self):
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _sample_loop(self):
        while not self._stop.wait(self.interval):
            sample = {"timestamp": time.time()}
            if psutil is not None:
                sample["cpu_percent"] = psutil.cpu_percent(interval=None)
                sample["ram_percent"] = psutil.virtual_memory().percent
            if self._gpu_handle is not None:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                    sample["gpu_util_percent"] = util.gpu
                    sample["gpu_mem_used_mb"] = mem.used / (1024 * 1024)
                    sample["gpu_mem_total_mb"] = mem.total / (1024 * 1024)
                except Exception:
                    pass
            self.samples.append(sample)

    def summarize(self) -> dict:
        if not self.samples:
            return {}
        summary = {
            "sample_count": len(self.samples),
            "duration_seconds": round(self.samples[-1]["timestamp"] - self.samples[0]["timestamp"], 2),
        }
        for key in ("cpu_percent", "ram_percent", "gpu_util_percent", "gpu_mem_used_mb"):
            vals = [s[key] for s in self.samples if key in s]
            if vals:
                summary[f"{key}_mean"] = round(sum(vals) / len(vals), 2)
                summary[f"{key}_max"] = round(max(vals), 2)
        if pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        return summary


# Evaluation drivers -----------------------------------------------------------
def evaluate_accuracy(args: argparse.Namespace) -> dict:
    """Evaluate detection accuracy against YOLO labels."""
    model = YOLO(args.model_path)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and torch.cuda.is_available():
        model.to("cuda")

    image_label_pairs = []
    for split in args.splits:
        img_dir = Path(args.dataset_root) / split / "images"
        lbl_dir = Path(args.dataset_root) / split / "labels"
        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                image_label_pairs.append((img_path, lbl_path))

    if not image_label_pairs:
        return {"error": "no labeled images found"}

    iou_thresholds = [0.5, 0.75]
    metrics = {t: {"tp": 0, "fp": 0, "fn": 0} for t in iou_thresholds}
    all_preds_by_thresh = {t: [] for t in iou_thresholds}
    all_gts = []
    per_image_results = []

    for img_path, lbl_path in tqdm(image_label_pairs, desc="Accuracy"):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        img_h, img_w = frame.shape[:2]
        gt_boxes = load_yolo_labels(lbl_path, img_w, img_h)
        all_gts.extend(gt_boxes)

        detections = run_detection_pipeline(
            model, frame, time.time(), {},
            conf=args.confidence_threshold,
            iou=args.iou_threshold,
            half=args.half and device == "cuda",
            device=device,
        )
        pred_boxes = [d["bbox"] for d in detections]
        pred_confs = [d["confidence"] for d in detections]

        for box, conf in zip(pred_boxes, pred_confs):
            for t in iou_thresholds:
                all_preds_by_thresh[t].append((conf, box))

        for t in iou_thresholds:
            tp, fp, fn = match_detections_to_labels(pred_boxes, gt_boxes, t)
            metrics[t]["tp"] += tp
            metrics[t]["fp"] += fp
            metrics[t]["fn"] += fn

        per_image_results.append({
            "image": str(img_path),
            "gt_count": len(gt_boxes),
            "pred_count": len(pred_boxes),
        })

    results = {
        "images_evaluated": len(image_label_pairs),
        "ground_truth_boxes": len(all_gts),
    }
    for t in iou_thresholds:
        tp = metrics[t]["tp"]
        fp = metrics[t]["fp"]
        fn = metrics[t]["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        ap = compute_ap(all_preds_by_thresh[t], all_gts, t)
        results[f"iou_{t}"] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "ap": round(ap, 4),
        }

    return results


def match_detections_to_labels(pred_boxes: list[list[int]], gt_boxes: list[list[float]], iou_thresh: float) -> tuple[int, int, int]:
    """Greedy matching of predictions to ground-truth boxes."""
    matched_gt = set()
    tp = fp = 0
    for pb in pred_boxes:
        best_iou = 0.0
        best_idx = -1
        for gi, gb in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = bbox_iou(np.array(pb), np.array(gb))
            if iou > best_iou:
                best_iou = iou
                best_idx = gi
        if best_iou >= iou_thresh and best_idx not in matched_gt:
            matched_gt.add(best_idx)
            tp += 1
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def get_frame_source(args: argparse.Namespace) -> tuple[list[Path] | cv2.VideoCapture, int]:
    """Return an iterable frame source (list of paths or cv2 VideoCapture) and the source FPS."""
    if args.frames_dir:
        frame_paths = sorted(Path(args.frames_dir).glob("*.jpg"))
        if args.max_frames:
            frame_paths = frame_paths[: args.max_frames]
        return frame_paths, args.fps

    if args.video_path:
        cap = cv2.VideoCapture(str(args.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
        return cap, int(fps) if fps else args.fps

    return [], args.fps


def evaluate_speed_and_tracking(args: argparse.Namespace) -> dict:
    """Run sequential frames and collect latency, throughput and tracker stats."""
    model = YOLO(args.model_path)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and torch.cuda.is_available():
        model.to("cuda")

    ocr_engine = None
    if args.enable_ocr:
        if PaddleOCR is None:
            raise RuntimeError("PaddleOCR is not installed; install it or remove --enable-ocr")
        ocr_engine = PaddleOCR(
            lang="en",
            use_angle_cls=True,
            use_gpu=(device == "cuda"),
            show_log=False,
        )

    source, fps = get_frame_source(args)
    if not source:
        return {"error": "no frames or video provided"}

    # Warm-up
    dummy = np.zeros((540, 960, 3), dtype=np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_detection_pipeline(
            model, dummy, time.time(), {},
            conf=args.confidence_threshold,
            iou=args.iou_threshold,
            half=args.half and device == "cuda",
            device=device,
        )

    monitor = ResourceMonitor(interval=0.5)
    monitor.start()

    track_state = {}
    latencies = []
    detection_counts = []
    frame_times = []
    track_id_seen_frames: dict[int, list[int]] = defaultdict(list)
    prev_dets: list[dict] = []
    id_switches = 0
    total_frames = 0

    cap = None
    if isinstance(source, list):
        total_available = len(source)

        def frame_iter():
            for idx, path in enumerate(source):
                img = cv2.imread(str(path))
                yield idx, img
        iterator = frame_iter()
    else:
        cap = source
        fps = cap.get(cv2.CAP_PROP_FPS) or fps
        total_available = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None

        def frame_iter():
            idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                yield idx, frame
                idx += 1
        iterator = frame_iter()

    pbar_total = args.max_frames
    if pbar_total is None and total_available is not None:
        pbar_total = total_available
    pbar = tqdm(total=pbar_total, desc="Speed/Tracking")

    try:
        for frame_idx, frame in iterator:
            if frame is None:
                break
            if args.max_frames and total_frames >= args.max_frames:
                break

            now = frame_idx / max(fps, 1)
            t0 = time.perf_counter()
            detections = run_detection_pipeline(
                model, frame, now, track_state,
                conf=args.confidence_threshold,
                iou=args.iou_threshold,
                half=args.half and device == "cuda",
                device=device,
                ocr_engine=ocr_engine,
            )
            t1 = time.perf_counter()
            latency = t1 - t0

            latencies.append(latency)
            detection_counts.append(len(detections))
            frame_times.append(now)
            total_frames += 1

            for d in detections:
                track_id_seen_frames[d["track_id"]].append(frame_idx)

            # Count ID switches between consecutive frames by overlap.
            current_dets = detections
            if prev_dets and current_dets:
                matched_prev = set()
                for cd in current_dets:
                    best_iou = 0.0
                    best_prev = None
                    for pd in prev_dets:
                        if id(pd) in matched_prev:
                            continue
                        iou = bbox_iou(np.array(cd["bbox"]), np.array(pd["bbox"]))
                        if iou > best_iou:
                            best_iou = iou
                            best_prev = pd
                    if best_iou >= 0.3 and best_prev is not None:
                        matched_prev.add(id(best_prev))
                        if cd["track_id"] != best_prev["track_id"]:
                            id_switches += 1
            prev_dets = current_dets

            pbar.update(1)
    finally:
        if not isinstance(source, list):
            source.release()
        pbar.close()
        monitor.stop()

    if not latencies:
        return {"error": "no frames were processed"}

    total_time = sum(latencies)
    speed = {
        "frames_processed": total_frames,
        "wall_time_seconds": round(total_time, 2),
        "throughput_fps": round(total_frames / total_time, 2) if total_time else 0.0,
        "latency_seconds": {
            "mean": round(float(np.mean(latencies)), 4),
            "median": round(float(np.median(latencies)), 4),
            "p95": round(float(np.percentile(latencies, 95)), 4),
            "p99": round(float(np.percentile(latencies, 99)), 4),
            "min": round(float(np.min(latencies)), 4),
            "max": round(float(np.max(latencies)), 4),
        },
        "detections_per_frame": {
            "mean": round(float(np.mean(detection_counts)), 2),
            "max": int(max(detection_counts)) if detection_counts else 0,
        },
    }

    track_lengths = [len(frames) for frames in track_id_seen_frames.values()]
    tracking = {
        "total_unique_track_ids": len(track_id_seen_frames),
        "track_length_frames": {
            "mean": round(float(np.mean(track_lengths)), 2) if track_lengths else 0.0,
            "max": int(max(track_lengths)) if track_lengths else 0,
            "min": int(min(track_lengths)) if track_lengths else 0,
        },
        "internal_id_switches_between_frames": id_switches,
        "note": "Ground-truth track IDs are not provided, so IDF1/MOTA cannot be computed; "
                "use these internal statistics as a proxy for tracker consistency.",
    }

    return {
        "speed": speed,
        "tracking": tracking,
        "resources": monitor.summarize(),
    }


# CLI --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parent.parent
    default_model = repo_root / "FOR_SERVER_ENVIROMENT" / "best.pt"
    default_dataset = repo_root / "detection-validation" / "license plate recognition.v2" / "license plate recognition.v2"
    default_frames = repo_root / "detection-validation" / "extracted_frames"
    default_video = repo_root / "detection-validation" / "Vehicles Driving Through Flooding A93 Road Perthshire Scotland.mp4"

    parser = argparse.ArgumentParser(description="ALPR performance evaluator")
    parser.add_argument(
        "--mode",
        choices=["accuracy", "speed", "tracking", "all"],
        default="all",
        help="Metric category to evaluate. 'tracking' is included in 'speed'.",
    )
    parser.add_argument("--model-path", type=Path, default=default_model, help="Path to YOLO .pt file")
    parser.add_argument("--dataset-root", type=Path, default=default_dataset, help="Root of YOLO dataset")
    parser.add_argument("--splits", nargs="+", default=["valid", "test"], help="Dataset splits for accuracy")
    parser.add_argument("--frames-dir", type=Path, default=default_frames, help="Directory of extracted frames")
    parser.add_argument("--video-path", type=Path, default=default_video, help="Alternative video source")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames for speed test")
    parser.add_argument("--fps", type=int, default=30, help="Frame-rate assumed for tracking TTL logic")
    parser.add_argument("--confidence-threshold", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference (GPU only)")
    parser.add_argument("--enable-ocr", action="store_true", help="Run PaddleOCR inside the pipeline")
    parser.add_argument("--output", type=Path, default=Path("alpr_evaluation_results.json"))
    return parser


def make_json_safe(obj):
    """Recursively convert Path objects to strings so json.dump works."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    return obj


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.model_path.exists():
        print(f"[ERROR] Model not found: {args.model_path}")
        return

    report = {"arguments": make_json_safe(vars(args))}

    if args.mode in ("accuracy", "all"):
        print("\n=== Detection Accuracy ===")
        report["detection_accuracy"] = evaluate_accuracy(args)
        print(json.dumps(report["detection_accuracy"], indent=2))

    if args.mode in ("speed", "tracking", "all"):
        print("\n=== Speed / Throughput / Tracking ===")
        speed_tracking = evaluate_speed_and_tracking(args)
        report["speed_and_tracking"] = speed_tracking
        print(json.dumps(speed_tracking, indent=2))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[INFO] Results written to {args.output}")


if __name__ == "__main__":
    main()
