# VLM Challenge

Vision Language Model Challenge Implementation using OpenPack dataset.

## Overview

This project implements a flexible pipeline for training and deploying Vision Language Models (VLMs) on the OpenPack dataset.

## Project Structure

```
vlm-openpack/
├── data/                    # Data storage
│   ├── raw/                # Original OpenPack dataset
│   ├── processed/          # Extracted frames and processed data
│   └── shards/            # WebDataset tar shards
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── training/          # Training pipeline
│   ├── evaluation/        # Evaluation metrics and evaluator
│   └── api/              # FastAPI inference server
├── notebooks/            # Jupyter notebooks
├── training_data_samples/ # Sample data for testing
├── docker-compose.yml    # Docker composition
├── Dockerfile           # Docker image definition
└── requirements.txt    # Python dependencies
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.1 (for GPU support)
- Docker & Docker Compose (optional)

### Local Setup

```bash
# Clone the repository
git clone <repo-url>
cd vlm-openpack

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Docker Setup

```bash
docker-compose up -d vlm-api
```

## Quick Start

### 1. Data Processing

```python
from src.data import annotation_parser, clip_builder, frame_sampler

# Parse annotations
annotations = annotation_parser.parse_annotations("data/raw/annotations.json")

# Build clips
clips = clip_builder.build_clips("data/raw", annotations)

# Sample frames
frames = frame_sampler.sample_frames("path/to/video.mp4")
```

### 2. Training

```python
from src.training import dataset, finetune_config
from src.training.finetune_config import FinetuneConfig

# Create config
config = FinetuneConfig(
    model_name="base_vlm",
    batch_size=32,
    num_epochs=10
)

# Initialize dataset
train_dataset = dataset.VLMDataset("data/processed", config)

# Train model
# python -m src.training.train --config config.yaml
```

### 3. Inference API

```bash
# Start API server
python -m src.api.main

# Make predictions
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_data": "...", "prompt": "..."}'
```

## Configuration

Edit `src/training/finetune_config.py` to customize training parameters:

- Model architecture
- Learning rate and optimization
- Batch size and hardware settings
- Data splitting ratios

## Evaluation

```python
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation import metrics

evaluator = ModelEvaluator(model, device="cuda")
results = evaluator.evaluate(test_dataset)
```

## Development

See [ARCHITECTURE.md](ARCHITECTURE.md) for technical details.
See [AGENTS.md](AGENTS.md) for agent-based components.

## License

TBD
