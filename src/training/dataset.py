# PyTorch dataset for VLM training

import json
import torch
import webdataset as wds
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import io


class VLMDataset(Dataset):

    def __init__(self, annotations_file, frames_dir=None, config=None):
        self.annotations_file = Path(annotations_file)
        self.frames_dir = Path(frames_dir) if frames_dir else None
        self.config = config or {}
        
        # Load annotations
        with open(self.annotations_file) as f:
            self.samples = json.load(f)
        
        print(f\"[OK] Loaded {len(self.samples)} samples from {self.annotations_file.name}\")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        clip_id = sample["clip_id"]
        
        # Load frames (try from video path or frames directory)
        frames_tensor = self._load_frames(sample, idx)
        
        # Create training labels
        labels = {
            "dominant_operation": sample["dominant_operation"],
            "anticipated_next_operation": sample["anticipated_next_operation"],
            "temporal_segment": sample["temporal_segment"],
        }
        
        return {
            "clip_id": clip_id,
            "frames": frames_tensor,
            "dominant_operation": sample["dominant_operation"],
            "anticipated_next_operation": sample["anticipated_next_operation"],
            "temporal_segment": sample["temporal_segment"],
            "labels": labels,
        }

    def _load_frames(self, sample, idx):
        try:
            # Try loading from video path
            video_path = Path(sample.get("video_path", ""))
            if video_path.exists():
                return self._load_video_frames(video_path)
        except Exception as e:
            print(f\"[WARN] Error loading video {sample.get('video_path')}: {e}\")
        
        # Fallback: create dummy tensor (for testing)
        return torch.randn(8, 3, 336, 336, dtype=torch.float32)

    def _load_video_frames(self, video_path):
        """Load frames from video file."""
        import cv2
        
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize to 336x336
            frame = cv2.resize(frame, (336, 336))
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            frame = torch.from_numpy(frame).float() / 255.0
            
            # Permute to [C, H, W]
            frame = frame.permute(2, 0, 1)
            
            frames.append(frame)
        
        cap.release()
        
        # Sample 8 frames uniformly (for inference efficiency)
        if len(frames) > 8:
            indices = torch.linspace(0, len(frames) - 1, 8).long()
            frames = [frames[i] for i in indices]
        
        # Stack frames
        frames_tensor = torch.stack(frames)
        return frames_tensor


class WebDatasetLoader:
    
    @staticmethod
    def create_loader(shard_dir, batch_size=4, num_workers=2, shuffle=True):
        dataset = wds.WebDataset(f"{shard_dir}/shard_*.tar")
        
        dataset = dataset.decode("pil").to_dict()
        
        if shuffle:
            dataset = dataset.shuffle(1000)
        
        dataset = dataset.batched(batch_size)
        
        # Convert to DataLoader
        return dataset


def create_vlm_dataloader(annotations_file, frames_dir=None, batch_size=4, 
                          num_workers=0, shuffle=True):
    dataset = VLMDataset(
        annotations_file=annotations_file,
        frames_dir=frames_dir,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
    )
    
    return dataloader, dataset
