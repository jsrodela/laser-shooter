import os
from pathlib import Path

import cv2


PROJECT_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = PROJECT_DIR / "videos"


def prompt_int(name: str, default: int) -> int:
    value = input(f"{name} (default: {default}): ").strip()
    if not value:
        return default

    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, received: {value!r}") from exc


def select_video_source() -> int | str:
    print("Select Video input\n\t(0) Webcam (1) Videos\n")
    input_type = prompt_int("Video input", 0)

    if input_type == 0:
        return prompt_int("Webcam input id", 0)

    if input_type != 1:
        raise RuntimeError("Video input must be 0 (webcam) or 1 (video file).")

    if not VIDEOS_DIR.is_dir():
        raise RuntimeError(
            f"Video directory does not exist: {VIDEOS_DIR}. "
            "Create it and add a video file first."
        )

    videos = sorted(path for path in VIDEOS_DIR.iterdir() if path.is_file())
    if not videos:
        raise RuntimeError(f"No video files were found in: {VIDEOS_DIR}")

    for index, path in enumerate(videos):
        print(f"\t({index}) {path.name}")
    print()

    video_id = prompt_int("Video id", 0)
    if not 0 <= video_id < len(videos):
        raise RuntimeError(
            f"Video id must be between 0 and {len(videos) - 1}, received: {video_id}"
        )

    return str(videos[video_id])


def open_video_capture(source: int | str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(source)
    if capture.isOpened():
        return capture
    capture.release()

    # Media Foundation is normally selected on Windows. DirectShow is a useful
    # fallback for cameras that are present but fail to open through that backend.
    if os.name == "nt" and isinstance(source, int):
        capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if capture.isOpened():
            print("Using the DirectShow camera backend.")
            return capture
        capture.release()

    source_type = "webcam" if isinstance(source, int) else "video file"
    raise RuntimeError(
        f"Could not open {source_type}: {source}. "
        "Check the camera id, Windows camera permissions, and whether another app is using it."
    )
