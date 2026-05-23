"""
Detection statistics tracker.

Aggregates per-frame results from the detector into a session summary the
frontend displays at the end of a video run. This is the "operator-grade
performance summary" — what an operator would actually want to see.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List


class StatsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_ts = time.time()
        self.frames_processed = 0
        self.frames_with_detection = 0
        self.total_detections = 0
        self.class_counts: Dict[str, int] = defaultdict(int)
        self.confidence_sum = 0.0
        self.confidence_n = 0
        self.inference_times_ms: List[float] = []
        self.peak_concurrent = 0  # max drones seen in a single frame

    def update(self, frame_result: dict):
        self.frames_processed += 1
        dets = frame_result.get("detections", [])
        drone_dets = [d for d in dets if "drone" in d["class"].lower()]
        if drone_dets:
            self.frames_with_detection += 1
        self.total_detections += len(drone_dets)
        self.peak_concurrent = max(self.peak_concurrent, len(drone_dets))
        for d in dets:
            self.class_counts[d["class"]] += 1
            self.confidence_sum += d["confidence"]
            self.confidence_n += 1
        self.inference_times_ms.append(frame_result["inference_ms"])

    def summary(self) -> dict:
        elapsed = time.time() - self.start_ts
        avg_conf = (self.confidence_sum / self.confidence_n) if self.confidence_n else 0
        avg_infer = (sum(self.inference_times_ms) / len(self.inference_times_ms)
                     if self.inference_times_ms else 0)
        p95_infer = (sorted(self.inference_times_ms)[int(0.95 * len(self.inference_times_ms))]
                     if len(self.inference_times_ms) > 10 else avg_infer)
        detection_rate = (self.frames_with_detection / self.frames_processed
                          if self.frames_processed else 0)

        return {
            "session_seconds": round(elapsed, 1),
            "frames_processed": self.frames_processed,
            "frames_with_drone": self.frames_with_detection,
            "detection_rate": round(detection_rate, 3),
            "total_detections": self.total_detections,
            "peak_concurrent_drones": self.peak_concurrent,
            "avg_confidence": round(avg_conf, 3),
            "avg_inference_ms": round(avg_infer, 1),
            "p95_inference_ms": round(p95_infer, 1),
            "effective_fps": round(1000 / avg_infer if avg_infer else 0, 1),
            "class_breakdown": dict(self.class_counts),
        }
