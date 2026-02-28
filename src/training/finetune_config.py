# Fine-tuning configuration for QLoRA training

from dataclasses import dataclass, field
from typing import Optional, List
import json
from pathlib import Path


@dataclass
class LoRAConfig:
    r: int = 8  # LoRA rank
    lora_alpha: int = 16  # LoRA scaling
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"


@dataclass
class FinetuneConfig:
    
    # Model settings
    model_name: str = "Qwen/Qwen2.5-VL-2B-Instruct"
    model_precision: str = "4bit"  # 4bit quantization for QLoRA
    
    # Training settings
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 2  # Per GPU batch size for T4
    gradient_accumulation_steps: int = 16  # Simulate larger batches
    warmup_steps: int = 500
    
    # Data settings
    train_data_path: str = "data/synthetic/annotations.json"
    val_data_path: Optional[str] = None
    max_seq_length: int = 2048
    
    # Optimization
    optimizer: str = "paged_adamw_32bit"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    
    # LoRA configuration
    use_lora: bool = True
    lora_config: LoRAConfig = field(default_factory=LoRAConfig)
    
    # Output settings
    output_dir: str = "outputs/qwen2.5-vl-lora"
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # Hardware/inference settings
    device_map: str = "auto"
    gpu_count: int = 1
    num_workers: int = 2
    
    # Early stopping
    early_stopping_patience: int = 3
    
    def to_dict(self):
        return {
            "model_name": self.model_name,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.batch_size * self.gradient_accumulation_steps,
        }
    
    def save(self, path: str):
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


# Preset configurations for different hardware
KAGGLE_T4_CONFIG = FinetuneConfig(
    batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_epochs=3,
)

GCP_A100_CONFIG = FinetuneConfig(
    batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=5e-4,
    num_epochs=5,
)

LOCAL_DEV_CONFIG = FinetuneConfig(
    batch_size=1,
    gradient_accumulation_steps=32,
    learning_rate=2e-4,
    num_epochs=1,
    save_steps=10,
    eval_steps=10,
)
