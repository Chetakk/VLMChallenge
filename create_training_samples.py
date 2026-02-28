"""
Create 20 Training Sample Pairs

Extracts 20 clips from synthetic data for training verification.
Used to validate model behavior before full fine-tuning on Kaggle.
"""

import json
import shutil
from pathlib import Path

def create_training_samples():
    """Extract 20 clips for training verification."""
    
    # Paths
    synthetic_dir = Path("data/synthetic")
    videos_dir = synthetic_dir / "videos"
    annotations_file = synthetic_dir / "annotations.json"
    output_dir = Path("training_data_samples")
    
    # Verify synthetic data exists
    if not annotations_file.exists():
        print(f"[ERROR] Synthetic annotations not found: {annotations_file}")
        return False
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Load annotations
    with open(annotations_file) as f:
        all_clips = json.load(f)
    
    print(f"\n[PROCESSING] Loading {len(all_clips)} synthetic clips...")
    
    # Select 20 clips (every 5th clip for diversity)
    selected_clips = all_clips[::5][:20]
    
    training_samples = []
    videos_copied = 0
    
    for i, clip_metadata in enumerate(selected_clips, 1):
        clip_id = clip_metadata["clip_id"]
        src_video = Path(clip_metadata["video_path"])
        dst_video = output_dir / f"{clip_id}.mp4"
        
        # Copy video file
        if src_video.exists():
            shutil.copy2(src_video, dst_video)
            videos_copied += 1
        else:
            print(f"[WARN] Video not found: {src_video}")
            continue
        
        # Create training sample metadata
        sample = {
            "clip_id": clip_id,
            "video_file": str(dst_video),
            "dominant_operation": clip_metadata["dominant_operation"],
            "temporal_segment": clip_metadata["temporal_segment"],
            "anticipated_next_operation": clip_metadata["anticipated_next_operation"],
            "confidence": 0.95,
        }
        
        training_samples.append(sample)
        
        if i % 5 == 0:
            print(f"[OK] Processed {i}/{len(selected_clips)} samples")
    
    # Save training samples index
    index_file = output_dir / "index.json"
    with open(index_file, "w") as f:
        json.dump({
            "total_samples": len(training_samples),
            "samples": training_samples,
        }, f, indent=2)
    
    print("\n" + "="*60)
    print(f"[SUCCESS] Created {len(training_samples)} training samples")
    print(f"[DIR] Output: {output_dir}")
    print(f"[FILE] Index: {index_file}")
    print(f"[VIDEOS] {videos_copied} videos copied")
    print("="*60 + "\n")
    
    return True


if __name__ == "__main__":
    success = create_training_samples()
    exit(0 if success else 1)
