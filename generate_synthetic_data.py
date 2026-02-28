# Generate synthetic warehouse videos with frame-level annotations

import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


# Configuration
NUM_CLIPS = 100
FPS = 25
CLIP_DURATION = 5  # seconds
FRAME_COUNT = FPS * CLIP_DURATION  # 125 frames
VIDEO_SIZE = (480, 640)  # Kinect resolution
OUTPUT_DIR = Path("data/synthetic")

# Operation sequence (realistic warehouse workflow)
OPERATION_SEQUENCE = [
    ("Box Setup", 0, 20),
    ("Inner Packing", 20, 60),
    ("Tape", 60, 100),
    ("Put Items", 100, 125),
]

# Operation colors for visual distinction
OP_COLORS = {
    "Box Setup": (255, 0, 0),           # Blue
    "Inner Packing": (0, 255, 0),       # Green
    "Tape": (0, 0, 255),                # Red
    "Put Items": (255, 255, 0),         # Cyan
    "Pack": (255, 0, 255),              # Magenta
    "Wrap": (0, 255, 255),              # Yellow
    "Label": (128, 128, 0),             # Dark Cyan
    "Final Check": (128, 0, 128),       # Purple
    "Idle": (128, 128, 128),            # Gray
    "Unknown": (64, 64, 64),            # Dark Gray
}


def create_synthetic_frame(frame_idx, current_op, next_op):
    frame = np.ones((VIDEO_SIZE[0], VIDEO_SIZE[1], 3), dtype=np.uint8) * 220
    
    # warehouse background
    square_size = 40
    for i in range(0, VIDEO_SIZE[0], square_size):
        for j in range(0, VIDEO_SIZE[1], square_size):
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                cv2.rectangle(frame, (j, i), (j + square_size, i + square_size),
                            (200, 200, 200), -1)
    
    # Draw current operation as colored rectangle
    color = OP_COLORS.get(current_op, (100, 100, 100))
    cv2.rectangle(frame, (50, 50), (430, 380), color, -1)
    cv2.rectangle(frame, (50, 50), (430, 380), (0, 0, 0), 3)
    
    # Add operation name text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, current_op, (80, 150), font, 1.5, (255, 255, 255), 2)
    
    # Add next operation hint
    cv2.putText(frame, f"Next: {next_op}", (80, 250), font, 1, (100, 100, 100), 1)
    
    # Progress bar
    cv2.rectangle(frame, (50, 400), (430, 420), (0, 0, 0), 2)
    progress = int((frame_idx / FRAME_COUNT) * 380)
    cv2.rectangle(frame, (50, 400), (50 + progress, 420), (0, 200, 0), -1)
    
    # Frame counter
    cv2.putText(frame, f"Frame: {frame_idx}/{FRAME_COUNT}", (50, 450),
                font, 0.8, (0, 0, 0), 1)
    
    return frame


def create_synthetic_video(clip_id, output_path):
    """Create a single synthetic video clip."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, FPS, (VIDEO_SIZE[1], VIDEO_SIZE[0]))
    
    annotations = []
    session_start = datetime.now()
    
    for frame_idx in range(FRAME_COUNT):
        # Determine current and next operation
        current_op = None
        next_op = None
        current_seg = None
        
        for op_name, start, end in OPERATION_SEQUENCE:
            if start <= frame_idx < end:
                current_op = op_name
                current_seg = (start, end)
                break
        
        # Find next operation
        for i, (op_name, start, end) in enumerate(OPERATION_SEQUENCE):
            if start <= frame_idx < end:
                if i + 1 < len(OPERATION_SEQUENCE):
                    next_op = OPERATION_SEQUENCE[i + 1][0]
                break
        
        if current_op is None:
            current_op = "Box Setup"
            current_seg = (0, 20)
        if next_op is None:
            next_op = "Label"
        
        # Create frame
        frame = create_synthetic_frame(frame_idx, current_op, next_op)
        out.write(frame)
        
        # Record annotation at segment boundaries
        if frame_idx == 0 or frame_idx == FRAME_COUNT - 1:
            annotations.append({
                "clip_id": clip_id,
                "frame_idx": frame_idx,
                "operation": current_op,
            })
    
    out.release()
    return annotations


def generate_clip_annotations(clip_idx):
    base_time = datetime.now()
    
    annotations = []
    cumulative_offset = 0
    
    for op_name, start_frame, end_frame in OPERATION_SEQUENCE:
        start_time = base_time + timedelta(seconds=start_frame / FPS)
        end_time = base_time + timedelta(seconds=end_frame / FPS)
        
        annotations.append({
            "operation": op_name,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
        })
    
    return annotations


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    videos_dir = OUTPUT_DIR / "videos"
    videos_dir.mkdir(exist_ok=True)
    
    all_clips_metadata = []
    
    print("[SYNTHETIC DATA] Generating synthetic warehouse videos...")
    print(f"[OUTPUT] Output directory: {OUTPUT_DIR}")
    print(f"[CREATED] {NUM_CLIPS} clips ({CLIP_DURATION}s @ {FPS}fps each)")
    print()
    
    for clip_idx in range(NUM_CLIPS):
        subject_id = f"U_SYNTH_{clip_idx // 25:02d}"  # Group into subjects (25 clips each)
        session_id = f"S_{clip_idx % 25:04d}"
        clip_id = f"{subject_id}_{session_id}_t0000"
        
        video_path = videos_dir / f"{clip_id}.mp4"
        
        # Create video
        create_synthetic_video(clip_id, video_path)
        
        # Generate annotations
        clip_annotations = generate_clip_annotations(clip_idx)
        
        # Build clip metadata (for API responses)
        clip_metadata = {
            "clip_id": clip_id,
            "video_path": str(video_path),
            "dominant_operation": clip_annotations[1]["operation"],  # Inner Packing
            "temporal_segment": {
                "start_frame": clip_annotations[1]["start_frame"],
                "end_frame": clip_annotations[1]["end_frame"],
            },
            "anticipated_next_operation": clip_annotations[2]["operation"],  # Tape
            "sample_frames": list(range(FRAME_COUNT)),  # All frames available
            "annotations": clip_annotations,
        }
        
        all_clips_metadata.append(clip_metadata)
        
        if (clip_idx + 1) % 10 == 0:
            print(f"[OK] Created {clip_idx + 1}/{NUM_CLIPS} clips")
    
    # Save master annotations file
    annotations_file = OUTPUT_DIR / "annotations.json"
    with open(annotations_file, "w") as f:
        json.dump(all_clips_metadata, f, indent=2)
    
    print()
    print("=" * 60)
    print("[SUCCESS] Synthetic data generation complete!")
    print(f"[STATS] Total clips: {NUM_CLIPS}")
    print(f"[DIR] Videos: {videos_dir}")
    print(f"[FILE] Annotations: {annotations_file}")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Run: python -m src.test_pipeline  (to verify pipeline)")
    print("2. Upload to Kaggle notebook")
    print("3. Run fine-tuning notebook")


if __name__ == "__main__":
    main()
