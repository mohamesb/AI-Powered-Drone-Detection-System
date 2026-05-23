"""
FastAPI backend for the drone detection & swarm visualization system.

Two channels:
  • REST /api/* for one-shot queries (videos available, stats)
  • WebSocket /ws/swarm    — streams swarm positions ~5 Hz
  • WebSocket /ws/camera   — streams the selected drone's camera feed with
                              YOLO detections drawn on each frame

Frame source for the "camera feed" is a YouTube video downloaded into
data/videos/. Different drones could in principle have different feeds; for
this demo, all drones share the same video, treated as what each one would
be "seeing." The currently-selected drone's feed is what the frontend shows.
"""
from __future__ import annotations

import asyncio
import base64
import random
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.detector import DroneDetector
from backend.stats import StatsTracker
from backend.swarm import Swarm

PROJECT_ROOT = Path(__file__).parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
VIDEOS_DIR = PROJECT_ROOT / "data" / "videos"


# ---------- App state ----------
class AppState:
    swarm: Swarm
    detector: DroneDetector
    stats: StatsTracker
    selected_drone_id: Optional[str] = None
    current_video: Optional[Path] = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start swarm + detector once at boot, tear them down on shutdown."""
    print("[startup] Initialising swarm…")
    state.swarm = Swarm(n_drones=6)

    print("[startup] Loading detector…")
    state.detector = DroneDetector()

    print("[startup] Initialising stats…")
    state.stats = StatsTracker()

    # Pick the first available video (if any) as the initial feed
    videos = sorted(VIDEOS_DIR.glob("*.mp4")) if VIDEOS_DIR.exists() else []
    state.current_video = videos[0] if videos else None
    if state.current_video:
        print(f"[startup] Initial video: {state.current_video.name}")
    else:
        print("[startup] No videos in data/videos/. "
              "Run scripts/download_youtube.py to add some.")

    # Pick a random drone to focus the camera view on initially
    state.selected_drone_id = random.choice(state.swarm.drones).id
    print(f"[startup] Selected drone: {state.selected_drone_id}")

    # Background task: tick the swarm continuously
    sim_task = asyncio.create_task(_swarm_loop())

    yield  # serve

    sim_task.cancel()
    print("[shutdown] Bye.")


async def _swarm_loop():
    """Continuously advance the swarm so its state is always fresh."""
    try:
        while True:
            state.swarm.step()
            await asyncio.sleep(0.1)   # 10 Hz tick
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan, title="Stendr-style Drone Detection Demo")


# ---------- REST API ----------
@app.get("/api/videos")
async def list_videos():
    """List available YouTube-downloaded videos."""
    if not VIDEOS_DIR.exists():
        return {"videos": []}
    return {
        "videos": [v.name for v in sorted(VIDEOS_DIR.glob("*.mp4"))],
        "current": state.current_video.name if state.current_video else None,
    }


@app.post("/api/videos/{name}")
async def select_video(name: str):
    """Switch which video the camera feed plays."""
    target = VIDEOS_DIR / name
    if not target.exists():
        return {"ok": False, "error": "Video not found"}
    state.current_video = target
    state.stats.reset()
    return {"ok": True, "current": target.name}


@app.post("/api/drone/{drone_id}/select")
async def select_drone(drone_id: str):
    """Switch which drone's camera the frontend should focus on."""
    if not any(d.id == drone_id for d in state.swarm.drones):
        return {"ok": False, "error": "Unknown drone"}
    state.selected_drone_id = drone_id
    return {"ok": True, "selected": drone_id}


@app.get("/api/stats")
async def get_stats():
    """Return the current session's detection statistics."""
    return state.stats.summary()


@app.post("/api/stats/reset")
async def reset_stats():
    state.stats.reset()
    return {"ok": True}


# ---------- WebSocket: swarm positions ----------
@app.websocket("/ws/swarm")
async def swarm_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            snap = state.swarm.snapshot()
            snap["selected_drone_id"] = state.selected_drone_id
            await websocket.send_json(snap)
            await asyncio.sleep(0.2)    # 5 Hz update
    except WebSocketDisconnect:
        pass


# ---------- WebSocket: camera feed + detections ----------
@app.websocket("/ws/camera")
async def camera_ws(websocket: WebSocket):
    """Stream the currently selected drone's video feed with YOLO overlays."""
    await websocket.accept()

    if not state.current_video:
        await websocket.send_json({"type": "error",
                                   "message": "No video loaded. "
                                              "Add one in data/videos/"})
        await websocket.close()
        return

    cap = cv2.VideoCapture(str(state.current_video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = 1.0 / max(src_fps, 5)

    state.stats.reset()
    last_video_path = state.current_video

    try:
        while True:
            # If the user switches videos mid-stream, restart with the new one
            if state.current_video != last_video_path and state.current_video:
                cap.release()
                cap = cv2.VideoCapture(str(state.current_video))
                last_video_path = state.current_video
                state.stats.reset()

            ok, frame = cap.read()
            if not ok:
                # Loop the video when it ends and emit the session summary
                await websocket.send_json({
                    "type": "summary",
                    "stats": state.stats.summary(),
                })
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                state.stats.reset()
                continue

            # Downscale to a sensible size for the wire
            h, w = frame.shape[:2]
            target_w = 640
            if w > target_w:
                scale = target_w / w
                frame = cv2.resize(frame, (target_w, int(h * scale)))

            # Run YOLO
            result = state.detector.detect_frame(frame)
            state.stats.update(result)

            # JPEG-encode for the WebSocket (small + browsers render natively)
            ok, jpg = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                continue
            b64 = base64.b64encode(jpg.tobytes()).decode("ascii")

            await websocket.send_json({
                "type": "frame",
                "jpeg_b64": b64,
                "result": result,
                "selected_drone_id": state.selected_drone_id,
                "live_stats": state.stats.summary(),
            })

            await asyncio.sleep(frame_delay)
    except WebSocketDisconnect:
        pass
    finally:
        cap.release()


# ---------- Frontend ----------
@app.get("/")
async def root():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=5000,
        reload=False,
        log_level="info",
    )
