"""
Download YouTube drone footage for evaluation using yt-dlp.

Usage:
    python scripts/download_youtube.py <youtube_url> [<youtube_url> …]

Or with no args, downloads a small list of suitable drone footage clips
that are typically suggested for drone-detection demos.

yt-dlp version 2026.03.17+ required (handles current YouTube extraction).
"""
import subprocess
import sys
from pathlib import Path

VIDEOS_DIR = Path(__file__).parent.parent / "data" / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)


# Replace these with any short drone clips you find on YouTube. These are
# examples of the *type* of search query that yields useful test footage —
# they may or may not still be live. Pick your own to make it robust.
DEFAULT_QUERIES = [
    "ytsearch1:DJI Mavic flight test short",
    "ytsearch1:drone close up sky",
    "ytsearch1:FPV drone footage short",
]


def download(target: str):
    """Download a single video or search query as mp4."""
    print(f"\n→ Fetching: {target}")
    cmd = [
        "yt-dlp",
        "-f", "bv*+ba/best",
        "--merge-output-format", "mp4",
        "--max-filesize", "100M",   # keep things light
        "-o", str(VIDEOS_DIR / "%(title).80s.%(ext)s"),
        target,
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("ERROR: yt-dlp not found. Install with: pip install yt-dlp")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"WARNING: download failed for {target}: {e}")


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_QUERIES
    for t in targets:
        download(t)

    print("\nDone. Videos saved in:", VIDEOS_DIR)
    print("Available files:")
    for f in sorted(VIDEOS_DIR.glob("*.mp4")):
        size_mb = f.stat().st_size / 1_000_000
        print(f"  {f.name}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
