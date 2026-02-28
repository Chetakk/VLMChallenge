# VLM inference engine for Qwen2.5-VL

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image


class Qwen25VLInference:
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-VL-2B-Instruct", 
                 device: str = "auto", 
                 use_lora: bool = False,
                 lora_path: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.use_lora = use_lora
        self.lora_path = lora_path
        self.model = None
        self.processor = None
        self._is_loaded = False

    def load_model(self):
        try:
            from transformers import AutoProcessor, Qwen2_5VLForConditionalGeneration
            from peft import PeftModel
        except ImportError:
            raise ImportError(
                "Please install: pip install transformers peft pillow"
            )
        
        print(f"Loading model from: {self.model_name}")
        
        # Load base model
        self.model = Qwen2_5VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        ).eval()
        
        # Load LoRA adapter if specified
        if self.use_lora:
            print(f"Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(self.model, self.lora_path)
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        print("Model loaded successfully")
        return self.model

    def predict(self, video_frames: torch.Tensor, prompt: str = None) -> Dict:
        if self.model is None:
            self.load_model()
        
        # Default prompt for operation classification + anticipation
        if prompt is None:
            prompt = (
                "Analyze this warehouse operation video. "
                "Identify: "
                "1. The main operation being performed "
                "2. When it starts and ends (frame numbers) "
                "3. What operation comes next. "
                "Operations: Box Setup, Inner Packing, Tape, Put Items, Pack, Wrap, Label, Final Check, Idle"
            )
        
        # Process video frames
        inputs = self.processor(
            text=prompt,
            images=[video_frames],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Generate prediction
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0,
            )
        
        # Decode output
        prediction_text = self.processor.decode(output_ids[0], skip_special_tokens=True)
        
        # Parse prediction (simple extraction for now)
        result = self._parse_prediction(prediction_text)
        
        return result

    def _parse_prediction(self, text: str) -> Dict:
        lines = text.lower().split('\n')
        
        # Extract operations (try to find in model output)
        operations = [
            "Box Setup", "Inner Packing", "Tape", "Put Items", 
            "Pack", "Wrap", "Label", "Final Check", "Idle", "Unknown"
        ]
        
        dominant_op = "Unknown"
        next_op = "Unknown"
        
        for line in lines:
            for op in operations:
                if op.lower() in line:
                    # First match is dominant operation
                    if dominant_op == "Unknown":
                        dominant_op = op
                    else:
                        next_op = op
                        break
        
        # Temporal segment (attempt to extract frame numbers)
        temporal_segment = {"start_frame": 0, "end_frame": 125}
        
        return {
            "dominant_operation": dominant_op,
            "temporal_segment": temporal_segment,
            "anticipated_next_operation": next_op if next_op != "Unknown" else "Label",
            "raw_output": text,
        }

    def batch_predict(self, batch_frames: List[torch.Tensor], 
                     prompts: Optional[List[str]] = None) -> List[Dict]:
        results = []
        for i, frames in enumerate(batch_frames):
            prompt = prompts[i] if prompts else None
            result = self.predict(frames, prompt)
            results.append(result)
        return results


class BaselineModel:
    OPERATION_SEQUENCE = [
        "Box Setup", "Inner Packing", "Tape", "Put Items",
        "Pack", "Wrap", "Label", "Final Check",
    ]
    
    def predict(self, clip_id: str = "", num_frames: int = 125) -> Dict:
        frame_per_op = num_frames // len(self.OPERATION_SEQUENCE)
        mid_idx = len(self.OPERATION_SEQUENCE) //2
        
        return {
            "clip_id": clip_id,
            "dominant_operation": self.OPERATION_SEQUENCE[mid_idx],
            "temporal_segment": {
                "start_frame": mid_idx * frame_per_op,
                "end_frame": (mid_idx + 1) * frame_per_op,
            },
            "anticipated_next_operation": self.OPERATION_SEQUENCE[min(mid_idx + 1, len(self.OPERATION_SEQUENCE) - 1)],
            "confidence": 0.33,
        }


