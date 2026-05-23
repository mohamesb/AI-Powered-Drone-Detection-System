"""
Get a working drone detector running in under a minute.

There are TWO paths, and 99% of users should take Path A:

  Path A (DEFAULT) — Download pre-trained drone-detector weights from
                     Hugging Face. No API keys, no auth, no SDK quirks.
                     Just an HTTP download. Run the system immediately.

  Path B (--with-dataset) — Download the Kaggle drone-vs-bird dataset
                            and prepare it for fine-tuning. Requires a
                            free Kaggle account + API token.

Usage:
    python scripts/download_dataset.py                # → Path A (recommended)
    python scripts/download_dataset.py --with-dataset # → Path B (training)
"""
import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

WEIGHTS_DEST = MODELS_DIR / "drone_detector_best.pt"


# ── PATH A: Pre-trained weights via Hugging Face ─────────────────────────────

def download_pretrained_weights() -> bool:
    """
    Download a community-trained drone detector from Hugging Face Hub.

    Why this works reliably:
      • Hugging Face Hub serves public model files over a CDN
      • No auth required for public repos
      • huggingface_hub library handles resume and retries automatically
      • Files end up in a known location we control
    """
    if WEIGHTS_DEST.exists():
        size_mb = WEIGHTS_DEST.stat().st_size / 1_000_000
        print(f"✓ Weights already present: {WEIGHTS_DEST}  ({size_mb:.1f} MB)")
        return True

    # Ensure huggingface_hub is installed
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Installing huggingface_hub …")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "--quiet", "huggingface_hub"])
        from huggingface_hub import hf_hub_download

    print("Downloading pre-trained drone-detector weights from Hugging Face …")
    print("  Source:  https://huggingface.co/doguilmak/Drone-Detection-YOLOv8x")
    print("  License: MIT")
    print()

    # Try the known weight filenames in order — repos vary
    candidates = [
        ("doguilmak/Drone-Detection-YOLOv8x",   "weight/best.pt"),
        ("doguilmak/Drone-Detection-YOLOv8x",   "best.pt"),
        ("doguilmak/Drone-Detection-YOLOv11x",  "best.pt"),
        ("doguilmak/Drone-Detection-YOLOv11x",  "weights/best.pt"),
    ]

    last_err = None
    for repo_id, filename in candidates:
        try:
            print(f"  Trying {repo_id} :: {filename}")
            path = hf_hub_download(repo_id=repo_id, filename=filename)
            shutil.copy(path, WEIGHTS_DEST)
            size_mb = WEIGHTS_DEST.stat().st_size / 1_000_000
            print(f"\n✓ Weights downloaded: {WEIGHTS_DEST}  ({size_mb:.1f} MB)")
            print("  The backend will pick these up automatically.")
            return True
        except Exception as e:
            last_err = e
            print(f"  → not at this path, trying next …")

    print(f"\nAll candidate paths failed. Last error: {last_err}")
    print("\nManual fallback — pre-trained weights are still downloadable directly:")
    print("  1. Visit https://huggingface.co/doguilmak/Drone-Detection-YOLOv8x")
    print("  2. Click 'Files and versions' tab")
    print("  3. Download 'best.pt' (or whichever .pt file you find)")
    print(f"  4. Move it to: {WEIGHTS_DEST}")
    return False


# ── PATH B: Full dataset via Kaggle  ─────────────────────────────────────────

def download_kaggle_dataset() -> bool:
    """
    Download the YOLO Drone Detection Dataset from Kaggle.

    Requires a free Kaggle API token (one-time setup, ~30 sec):
      1. https://www.kaggle.com → Settings → API → Create New Token
      2. Save kaggle.json to ~/.kaggle/kaggle.json
         (Linux/macOS) or %USERPROFILE%\\.kaggle\\kaggle.json (Windows)
    """
    if (DATA_DIR / "data.yaml").exists():
        print(f"✓ Dataset already present at {DATA_DIR}/data.yaml")
        return True

    # Ensure kagglehub is installed
    try:
        import kagglehub
    except ImportError:
        print("Installing kagglehub …")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "--quiet", "kagglehub"])
        import kagglehub

    # Check Kaggle credentials exist before attempting download
    kaggle_json = Path.home() / ".kaggle" / "access_token"
    if not kaggle_json.exists():
        print("ERROR: Kaggle API credentials not found.")
        print(f"  Expected file: {kaggle_json}")
        print()
        print("One-time setup (free, ~30 seconds):")
        print("  1. Go to https://www.kaggle.com → log in (or sign up)")
        print("  2. Top-right avatar → Settings → API → 'Create New Token'")
        print("  3. Save the downloaded kaggle.json to:")
        print(f"       {kaggle_json}")
        print("  4. Run this script again")
        return False

    print("Downloading 'YOLO Drone Detection Dataset' from Kaggle …")
    try:
        # This is the dataset doguilmak's model was trained on (~9 000 images,
        # YOLO format, drone class)
        path = kagglehub.dataset_download("muki2003/yolo-drone-detection-dataset")
        print(f"  Downloaded to: {path}")
    except Exception as e:
        print(f"\nKaggle download failed: {e}")
        return False

    # Reorganise into canonical layout
    return organise_kaggle_dataset(Path(path))


def organise_kaggle_dataset(src: Path) -> bool:
    """Lay out the Kaggle download into data/train and data/valid."""
    print("Organising files into data/train + data/valid …")

    # Find all images and labels — be tolerant of layout variations
    images = list(src.rglob("*.jpg")) + list(src.rglob("*.png"))
    labels = list(src.rglob("*.txt"))
    label_by_stem = {lbl.stem: lbl for lbl in labels}

    if len(images) < 10:
        print(f"WARNING: only {len(images)} images found in {src} — unexpected layout.")
        print(f"  Contents: {[p.name for p in src.iterdir()][:10]}")
        return False

    print(f"  Found {len(images)} images, {len(labels)} label files")

    train_img = DATA_DIR / "train" / "images"; train_img.mkdir(parents=True, exist_ok=True)
    train_lbl = DATA_DIR / "train" / "labels"; train_lbl.mkdir(parents=True, exist_ok=True)
    valid_img = DATA_DIR / "valid" / "images"; valid_img.mkdir(parents=True, exist_ok=True)
    valid_lbl = DATA_DIR / "valid" / "labels"; valid_lbl.mkdir(parents=True, exist_ok=True)

    images.sort()
    split_idx = int(len(images) * 0.9)   # 90% train, 10% val

    for i, img in enumerate(images):
        is_train = i < split_idx
        img_dest = (train_img if is_train else valid_img) / img.name
        lbl_dest = (train_lbl if is_train else valid_lbl) / (img.stem + ".txt")
        shutil.copy(img, img_dest)
        if img.stem in label_by_stem:
            shutil.copy(label_by_stem[img.stem], lbl_dest)
        else:
            # Image has no label = no drones in frame; empty label is valid YOLO
            lbl_dest.write_text("")

    # Write a clean self-contained data.yaml with relative paths
    yaml_text = (
        "# Auto-generated by scripts/download_dataset.py\n"
        f"path: {DATA_DIR}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "nc: 1\n"
        "names: ['drone']\n"
    )
    (DATA_DIR / "data.yaml").write_text(yaml_text)
    print(f"  Wrote {DATA_DIR / 'data.yaml'}")
    print(f"\n✓ Dataset ready. {split_idx} train, {len(images) - split_idx} val images.")
    return True


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--with-dataset", action="store_true",
                   help="Also download the full dataset for fine-tuning. "
                        "Requires a free Kaggle account + API token.")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("PATH A — Pre-trained drone-detector weights")
    print("=" * 60)
    ok_weights = download_pretrained_weights()

    if args.with_dataset:
        print()
        print("=" * 60)
        print("PATH B — Kaggle dataset for fine-tuning")
        print("=" * 60)
        ok_dataset = download_kaggle_dataset()
    else:
        ok_dataset = True

    print()
    print("=" * 60)
    if ok_weights:
        print("✓ READY TO RUN")
        print()
        print("Next steps:")
        print("  python scripts/download_youtube.py \"<youtube_url>\"")
        print("  python -m backend.main")
        print("  open http://localhost:8000")
        if args.with_dataset and ok_dataset:
            print()
            print("Optional: fine-tune on the Kaggle dataset for better accuracy")
            print("  python scripts/train.py")
        sys.exit(0)
    else:
        print("⚠  Setup incomplete — read the messages above")
        sys.exit(1)


if __name__ == "__main__":
    main()
