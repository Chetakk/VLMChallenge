# VRAM calculation utilities for QLoRA training

import math


# Model and hardware specifications
MODEL_PARAMS = {
    "qwen2.5-vl-2b": {
        "params": 2.4e9,  # 2.4 billion parameters
        "vocab_size": 151643,
        "hidden_dim": 1920,
    },
    "llava-next-video-7b": {
        "params": 7.5e9,  # 7.5 billion parameters
        "vocab_size": 32000,
        "hidden_dim": 4096,
    },
}

GPU_SPECS = {
    "kaggle_t4": {
        "name": "NVIDIA T4",
        "vram_gb": 16,
        "compute_gb_per_second": 130,  # T4 compute throughput
    },
    "gcp_a100": {
        "name": "NVIDIA A100",
        "vram_gb": 40,
        "compute_gb_per_second": 1560,
    },
}


def bytes_to_gb(bytes_val):
    return bytes_val / (1024 ** 3)


def calculate_model_weights_size(num_params, bits=4):
    bytes_per_param = bits / 8
    return num_params * bytes_per_param


def calculate_activations_size(batch_size, seq_length, hidden_dim, num_layers=24):
    activation_bytes = batch_size * seq_length * hidden_dim * num_layers * 4
    return activation_bytes


def calculate_optimizer_state_size(num_params, optimizer="adam"):
    # LoRA adapter size (rough estimate) - typically uses rank=8 or rank=16
    lora_rank = 8
    lora_params = 0.002 * num_params  # ~0.2% of model params
    
    if optimizer == "adam":
        # Adam stores: param, m (first moment), v (second moment)
        multiplier = 3
    else:
        multiplier = 1
    
    return lora_params * multiplier * 4  # 4 bytes per param


def calculate_total_training_memory(
    batch_size,
    model_name="qwen2.5-vl-2b",
    seq_length=512,
    use_gradient_checkpointing=True,
    mixed_precision=True,
):
    model_spec = MODEL_PARAMS[model_name]
    num_params = model_spec["params"]
    hidden_dim = model_spec["hidden_dim"]
    
    # Model weights (4-bit quantized with LoRA)
    model_weights_gb = bytes_to_gb(calculate_model_weights_size(num_params, bits=4))
    
    # LoRA weights (float32)
    lora_params = 0.002 * num_params
    lora_weights_gb = bytes_to_gb(lora_params * 4)
    
    # Activations (reduced if gradient checkpointing is used)
    activation_factor = 0.1 if use_gradient_checkpointing else 1.0
    activations_gb = bytes_to_gb(
        calculate_activations_size(batch_size, seq_length, hidden_dim) * activation_factor
    )
    
    # Optimizer states (Adam: m, v for LoRA params)
    optimizer_gb = bytes_to_gb(calculate_optimizer_state_size(num_params, "adam"))
    
    # Overhead (CUDA kernels, temporary buffers, etc.)
    overhead_gb = 2.0
    
    total_gb = model_weights_gb + lora_weights_gb + activations_gb + optimizer_gb + overhead_gb
    
    return {
        "model_weights_gb": round(model_weights_gb, 2),
        "lora_weights_gb": round(lora_weights_gb, 2),
        "activations_gb": round(activations_gb, 2),
        "optimizer_states_gb": round(optimizer_gb, 2),
        "overhead_gb": round(overhead_gb, 2),
        "total_gb": round(total_gb, 2),
    }


def calculate_optimal_batch_size(
    gpu_vram_gb,
    model_name="qwen2.5-vl-2b",
    seq_length=512,
    use_gradient_checkpointing=True,
    target_utilization=0.85,
):
    available_vram = gpu_vram_gb * target_utilization
    
    # Binary search for maximum batch size
    min_batch = 1
    max_batch = 128
    optimal_batch = 1
    
    while min_batch <= max_batch:
        mid_batch = (min_batch + max_batch) // 2
        memory = calculate_total_training_memory(
            batch_size=mid_batch,
            model_name=model_name,
            seq_length=seq_length,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        
        if memory["total_gb"] <= available_vram:
            optimal_batch = mid_batch
            min_batch = mid_batch + 1
        else:
            max_batch = mid_batch - 1
    
    # Estimate training time
    memory = calculate_total_training_memory(
        batch_size=optimal_batch,
        model_name=model_name,
        seq_length=seq_length,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    
    return {
        "optimal_batch_size": optimal_batch,
        "estimated_vram_gb": memory["total_gb"],
        "estimated_utilization": f"{(memory['total_gb'] / gpu_vram_gb * 100):.1f}%",
        "gradient_accumulation_steps": max(1, 32 // optimal_batch),  # Simulate larger batch
    }


def optimize_config_for_vram(vram_gb=16, model_name="qwen2.5-vl-2b"):
    batch_size_result = calculate_optimal_batch_size(
        gpu_vram_gb=vram_gb,
        model_name=model_name,
        use_gradient_checkpointing=True,
    )
    
    memory = calculate_total_training_memory(
        batch_size=batch_size_result["optimal_batch_size"],
        model_name=model_name,
        use_gradient_checkpointing=True,
    )
    
    # Determine if multi-GPU training is needed
    multi_gpu = vram_gb < 20  # If less than 20GB total, might use multiple T4s
    
    return {
        "model_name": model_name,
        "batch_size": batch_size_result["optimal_batch_size"],
        "gradient_accumulation_steps": batch_size_result["gradient_accumulation_steps"],
        "effective_batch_size": (
            batch_size_result["optimal_batch_size"] * 
            batch_size_result["gradient_accumulation_steps"]
        ),
        "learning_rate": 2e-4,
        "num_epochs": 3,
        "warmup_steps": 500,
        "use_gradient_checkpointing": True,
        "use_mixed_precision": True,
        "use_lora": True,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "memory_breakdown": memory,
        "vram_utilization": batch_size_result["estimated_utilization"],
        "gpu_vram_available_gb": vram_gb,
    }


def print_optimization_report(vram_gb=16, model_name="qwen2.5-vl-2b"):
    config = optimize_config_for_vram(vram_gb, model_name)
    
    print("\n" + "=" * 70)
    print("[VRAM OPTIMIZATION REPORT]".center(70))
    print("=" * 70)
    print(f"\n[INFO] Hardware: {vram_gb} GB VRAM")
    print(f"[INFO] Model: {model_name}")
    print(f"\n[RECOMMENDED] Configuration:")
    print(f"   • Batch Size: {config['batch_size']}")
    print(f"   • Gradient Accumulation: {config['gradient_accumulation_steps']}x")
    print(f"   • Effective Batch Size: {config['effective_batch_size']}")
    print(f"   • Learning Rate: {config['learning_rate']}")
    print(f"   • LoRA Rank: {config['lora_rank']}")
    print(f"\n[MEMORY] Breakdown:")
    mem = config["memory_breakdown"]
    print(f"   • Model Weights (4-bit): {mem['model_weights_gb']} GB")
    print(f"   • LoRA Weights: {mem['lora_weights_gb']} GB")
    print(f"   • Activations: {mem['activations_gb']} GB")
    print(f"   • Optimizer States: {mem['optimizer_states_gb']} GB")
    print(f"   • Overhead: {mem['overhead_gb']} GB")
    print(f"   • Total: {mem['total_gb']} GB ({config['vram_utilization']})")
    print("\n" + "=" * 70 + "\n")
    
    return config


if __name__ == "__main__":
    # Example usage
    print("\n🔍 Kaggle T4 (16GB):")
    config_t4 = print_optimization_report(vram_gb=16, model_name="qwen2.5-vl-2b")
    
    print("\n🔍 GCP A100 (40GB):")
    config_a100 = print_optimization_report(vram_gb=40, model_name="qwen2.5-vl-2b")
