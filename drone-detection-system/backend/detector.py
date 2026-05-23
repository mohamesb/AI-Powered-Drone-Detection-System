"""
YOLO11 inference wrapper for video frames.

This module encapsulates two responsibilities:
  1. Loading the fine-tuned drone-vs-bird model (or falling back to the
     pretrained generic model if you haven't trained yet)
  2. Running detection on a single frame and returning a tidy result dict

The backend WebSocket layer streams frames through `detect_frame` and pushes
the results back to the frontend, which draws the boxes.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "drone_detector" / "weights" / "best.pt"


class DroneDetector:
    """Lightweight wrapper around an Ultralytics YOLO model."""

    def __init__(self, weights_path: Optional[Path] = None, conf: float = 0.35):
        self.conf = conf
        weights = weights_path or DEFAULT_WEIGHTS

        if weights.exists():
            print(f"[detector] Loading fine-tuned weights: {weights}")
            self.model = YOLO(str(weights))
            self.is_finetuned = True
        else:
            # Fallback so the system still runs end-to-end before you train
            print("[detector] WARNING: fine-tuned weights not found.")
            print(f"  Expected: {weights}")
            print("  Falling back to pretrained yolo11n.pt (general objects).")
            print("  Run scripts/train.py to get drone-specific detection.")
            self.model = YOLO("yolo11n.pt")
            self.is_finetuned = False

        # Warm up the model so the first real inference isn't slow
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, conf=self.conf, verbose=False)

    def detect_frame(self, frame_bgr: np.ndarray) -> dict:
        """Run inference on a single BGR frame and return a result dict."""
        t0 = time.perf_counter()
        results = self.model.predict(frame_bgr, conf=self.conf, verbose=False)
        inference_ms = (time.perf_counter() - t0) * 1000

        detections = []
        if results and len(results) > 0:
            r = results[0]
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = r.names[cls_id]
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "class": cls_name,
                    "confidence": round(conf, 3),
                    "bbox": [round(x1, 1), round(y1, 1),
                             round(x2, 1), round(y2, 1)],
                    # Normalised for resolution-independent frontend drawing
                    "bbox_norm": [
                        round(x1 / frame_bgr.shape[1], 4),
                        round(y1 / frame_bgr.shape[0], 4),
                        round(x2 / frame_bgr.shape[1], 4),
                        round(y2 / frame_bgr.shape[0], 4),
                    ],
                })

        return {
            "inference_ms": round(inference_ms, 1),
            "frame_w": frame_bgr.shape[1],
            "frame_h": frame_bgr.shape[0],
            "detections": detections,
        }
