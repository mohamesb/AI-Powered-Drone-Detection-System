"""
Fine-tune YOLO11 on the drone-detection dataset.

This is OPTIONAL — the system runs fine with the pre-trained weights from
download_dataset.py. Run this only if you want to improve detection on
your own footage.

Requires that scripts/download_dataset.py --with-dataset has been run first.

Supports CUDA (Nvidia), MPS (Apple Silicon), and CPU.
Defaults target the free Google Colab T4 (~50 min for 50 epochs).

Run from the project root:
    python scripts/train.py
    python scripts/train.py --epochs 100
    python scripts/train.py --model yolo11s   # bigger, more accurate
    python scripts/train.py --batch 8         # if you hit OOM
"""
import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def find_data_yaml() -> Path:
    """Locate data.yaml, with a clear error if missing."""
    primary = DATA_DIR / "data.yaml"
    if primary.exists():
        return primary

    print(f"ERROR: no data.yaml found under {DATA_DIR}")
    print()
    print("Training is OPTIONAL. To set up the dataset, run:")
    print("  python scripts/download_dataset.py --with-dataset")
    print()
    print("…which needs a free Kaggle API token (~30 sec setup).")
    print()
    print("If you just want the system to RUN, you don't need training at all:")
    print("  python scripts/download_dataset.py     # pre-trained weights")
    print("  python -m backend.main                 # launch the app")
    sys.exit(1)


def detect_device() -> str:
    import torch
    if torch.cuda.is_available():
        print(f"✓ CUDA GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("✓ Apple Silicon MPS (Metal)")
        return "mps"
    print("⚠  No GPU found — training on CPU (slow). "
          "Consider Google Colab for a free T4 GPU.")
    return "cpu"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo11n",
                   choices=["yolo11n", "yolo11s", "yolo11m"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--imgsz",  type=int, default=640)
    p.add_argument("--batch",  type=int, default=16)
    p.add_argument("--patience", type=int, default=15)
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    yaml   = find_data_yaml()
    device = detect_device()

    print(f"\nDataset : {yaml}")
    print(f"Model   : {args.model}.pt")
    print(f"Epochs  : {args.epochs}  (early-stop patience {args.patience})")
    print(f"Imgsz   : {args.imgsz}")
    print(f"Batch   : {args.batch}")
    print(f"Device  : {device}\n")

    from ultralytics import YOLO
    model = YOLO(f"{args.model}.pt")

    results = model.train(
        data=str(yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=device,
        project=str(MODELS_DIR),
        name="drone_detector",
        exist_ok=True,
        plots=True,
        save=True,
    )

    best   = Path(results.save_dir) / "weights" / "best.pt"
    stable = MODELS_DIR / "drone_detector_best.pt"
    if best.exists():
        shutil.copy(best, stable)
        print(f"\n✓ Best weights → {stable}")
        print("  Backend will load these automatically on next start.")

    print("\nFinal validation metrics:")
    rd = results.results_dict
    for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)",
                "metrics/precision(B)", "metrics/recall(B)"):
        val = rd.get(key, "n/a")
        label = key.split("/")[1].split("(")[0]
        if isinstance(val, float):
            print(f"  {label:<14} {val:.3f}")
        else:
            print(f"  {label:<14} {val}")


if __name__ == "__main__":
    main()
