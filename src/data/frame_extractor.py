from pathlib import Path
import cv2


FPS = 25
TARGET_SIZE = (336, 336)


def extract_frames(video_path: str, output_dir: str):

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Failed to open video file.")

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, TARGET_SIZE)

        frame_filename = output_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(frame_filename), frame_resized)

        frame_idx += 1

    cap.release()

    print(f"Extracted {frame_idx} frames from {video_path.name}")



#for testing

# if __name__ == "__main__":
#     extract_frames(
#         video_path="U0101/kinect_frontal/S0100.avi",
#         output_dir="data/processed/U0101/S0100"
#     )