# Drone Detection & Swarm Visualization System

A full-stack project demonstrating AI-powered drone detection inspired by counter-UAS defense systems.

## What this is

A complete pipeline that:
1. **Downloads** pre-trained drone-detector weights from Hugging Face
2. **Visualizes** a simulated drone swarm in a dark-mode tactical interface — half map (live drones flying over Oslo) + half camera view (what the selected drone "sees")
3. **Detects** drones in YouTube footage using YOLO, drawing bounding boxes in real time
4. **Reports** detection statistics after each session

Optionally, you can fine-tune the model on your own dataset using the included `train.py` script.

## Quick start (90 seconds, no training needed)

```bash
# 1. Set up environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Download pre-trained drone-detector weights (one HTTP call, no auth)
python scripts/download_dataset.py

# 3. Download some YouTube drone footage for evaluation
python scripts/download_youtube.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"

# 4. Run the application
python -m backend.main

# 5. Open the browser
# → http://localhost:8000
```

That's it. The system is now running with a community-trained drone-detector that already works well out of the box.

## Optional: train your own model

Want to improve accuracy on your specific footage? You can fine-tune the model:

```bash
# Download the Kaggle dataset (~9 000 labelled drone images).
# This needs a free Kaggle API token — one-time 30-second setup.
python scripts/download_dataset.py --with-dataset

# Fine-tune (auto-detects GPU / Apple MPS / CPU)
python scripts/train.py

# Run the app with your fine-tuned weights
python -m backend.main
```

### Kaggle API token setup
1. https://www.kaggle.com → log in (or sign up — free)
2. Top-right avatar → Settings → API → "Create New Token"
3. Save the downloaded `kaggle.json` to:
   - Linux/macOS: `~/.kaggle/kaggle.json`
   - Windows: `%USERPROFILE%\.kaggle\kaggle.json`

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Frontend (HTML/JS/CSS — no build step needed)           │
│  ┌────────────────────┬────────────────────────────────┐ │
│  │  Tactical Map      │   Camera Feed (selected drone) │ │
│  │  (Leaflet + dark)  │   (YOLO detections overlaid)   │ │
│  │  Swarm positions   │   Live bounding boxes          │ │
│  └────────────────────┴────────────────────────────────┘ │
└────────────────────────────┬─────────────────────────────┘
                             │ WebSocket + REST
┌────────────────────────────┴─────────────────────────────┐
│  Backend (FastAPI)                                       │
│  • Swarm simulator (collision avoidance via shared pos)  │
│  • YOLO inference on video frames                        │
│  • Track aggregation + detection statistics              │
└──────────────────────────────────────────────────────────┘
```

## Project structure

```
drone-detection-system/
├── backend/
│   ├── main.py              # FastAPI server (REST + WebSocket)
│   ├── detector.py          # YOLO inference wrapper
│   ├── swarm.py             # Drone swarm simulator with collision avoidance
│   └── stats.py             # Detection statistics tracker
├── frontend/
│   ├── index.html           # Dark-mode tactical UI
│   ├── styles.css           # Minimal monospace aesthetic
│   └── app.js               # Map, video, and WebSocket client
├── scripts/
│   ├── download_dataset.py  # Pre-trained weights + optional Kaggle dataset
│   ├── download_youtube.py  # YouTube footage via yt-dlp
│   └── train.py             # Fine-tune YOLO11 on the dataset
├── data/                    # Dataset goes here (only if you train)
├── models/                  # Weights (.pt files) — auto-populated
├── requirements.txt
└── README.md
```

## Why this matters for defence tech

This project demonstrates the core capability behind modern counter-drone systems: turning raw camera feeds into trustable operator-grade situational awareness. It mirrors the sensor-fusion and tactical-display logic that companies build at industrial scale.
