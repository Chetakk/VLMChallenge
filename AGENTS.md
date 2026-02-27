# Agents Documentation

## Agent-Based Components

This document describes the autonomous agents and automated components in the VLM Challenge system.

## Data Processing Agent

### Purpose

Automates the data preparation pipeline from raw OpenPack data to training-ready shards.

### Workflow

1. **Validation**: Checks data integrity and annotation format
2. **Parsing**: Extracts annotations and metadata
3. **Clipping**: Builds temporal clips with annotations
4. **Sampling**: Extracts frames at optimal intervals
5. **Sharding**: Writes data to WebDataset format

### Usage

```python
from src.data import annotation_parser, clip_builder, frame_sampler, shard_writer

# Automated pipeline
annotations = annotation_parser.parse_annotations("path/to/annotations.json")
clips = clip_builder.build_clips("data/raw", annotations)
frames = frame_sampler.sample_frames("path/to/video")
shard_writer.write_shards(frames, "data/shards")
```

## Training Agent

### Purpose

Manages model training with automatic hyperparameter optimization.

### Components

- **Config Manager**: Loads and validates training configurations
- **Memory Optimizer**: Calculates optimal batch sizes based on hardware
- **Training Loop**: Handles forward/backward passes and optimization
- **Checkpoint Manager**: Saves and restores model states

### Configuration

```python
from src.training.finetune_config import FinetuneConfig

config = FinetuneConfig(
    learning_rate=1e-4,
    batch_size=32,
    num_epochs=10
)
```

## Evaluation Agent

### Purpose

Automatically evaluates model performance across multiple metrics.

### Metrics

- **Accuracy**: Overall prediction accuracy
- **Precision/Recall/F1**: Per-class metrics
- **mAP**: Mean Average Precision for localization tasks

### Usage

```python
from src.evaluation.evaluator import ModelEvaluator

evaluator = ModelEvaluator(model)
results = evaluator.evaluate(test_dataset)
print(results)
```

## API Server Agent

### Purpose

Provides automated inference through REST API.

### Endpoints

- `GET /health` - Health check
- `POST /predict` - Single prediction
- `GET /models` - List available models

### Features

- Automatic request validation
- Error handling and logging
- Rate limiting (optional)
- Model caching

## VRAM Optimization Agent

### Purpose

Automatically optimizes training configuration for available hardware.

### Calculations

- Estimates memory usage per batch
- Calculates maximum batch size
- Adjusts gradient accumulation steps
- Recommends mixed precision training

### Usage

```python
from src.training.vram_math import optimize_config_for_vram

optimized_config = optimize_config_for_vram(config, vram_available=24)
```

## Future Agent Extensions

- **Hyperparameter Tuning Agent**: Automated search for optimal hyperparameters
- **Data Augmentation Agent**: Intelligent data preprocessing and augmentation
- **Model Selection Agent**: Recommends best model architecture for task
- **Monitoring Agent**: Real-time performance tracking and alerts
