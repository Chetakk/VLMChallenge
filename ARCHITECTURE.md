# Architecture Documentation

## System Overview

The VLM Challenge system is composed of several interconnected modules:

### 1. Data Pipeline (`src/data/`)

**Annotation Parser** (`annotation_parser.py`)

- Parses annotation files from OpenPack dataset
- Validates annotation structure and content
- Provides structured access to annotations

**Clip Builder** (`clip_builder.py`)

- Constructs video clips from raw data
- Associates clips with corresponding annotations
- Handles frame extraction and temporal alignment

**Frame Sampler** (`frame_sampler.py`)

- Extracts frames at specified intervals
- Supports variable sampling rates
- Optimizes storage through frame selection

**Shard Writer** (`shard_writer.py`)

- Writes data to WebDataset tar shards
- Enables distributed and parallel loading
- Creates indices for efficient access

### 2. Training Pipeline (`src/training/`)

**Dataset** (`dataset.py`)

- PyTorch Dataset implementation
- Handles data loading and preprocessing
- Supports batch processing

**VRAM Math** (`vram_math.py`)

- Calculates optimal batch sizes
- Estimates memory requirements
- Optimizes configurations for hardware

**Finetune Config** (`finetune_config.py`)

- Centralized configuration management
- Dataclass-based parameter definitions
- Easy serialization and validation

### 3. Evaluation System (`src/evaluation/`)

**Metrics** (`metrics.py`)

- Accuracy calculation
- Precision, Recall, F1 score
- Mean Average Precision (mAP)

**Evaluator** (`evaluator.py`)

- Orchestrates evaluation on datasets
- Applies multiple metrics
- Generates comprehensive reports

### 4. API Server (`src/api/`)

**Main** (`main.py`)

- FastAPI application
- Health checks
- Prediction endpoints

**Inference** (`inference.py`)

- Model loading
- Single and batch inference
- Optimization for production

## Data Flow

```
Raw Data (OpenPack)
    ↓
Annotation Parser
    ↓
Clip Builder
    ↓
Frame Sampler
    ↓
Shard Writer
    ↓
WebDataset Shards
    ↓
Training Dataset
    ↓
Model Training
    ↓
Evaluation
    ↓
API Deployment
```

## Component Interactions

### Training Flow

1. Load configuration from `finetune_config.py`
2. Initialize `VLMDataset` with processed data
3. Calculate optimal batch size using `vram_math.py`
4. Train model with PyTorch
5. Evaluate using `ModelEvaluator`

### Inference Flow

1. Load model via `VLMInference`
2. Process input image and prompt
3. Run inference
4. Return predictions via FastAPI

## Key Design Decisions

1. **Modular Architecture**: Each component has a single responsibility
2. **WebDataset Support**: Enables efficient distributed training
3. **Configuration-Driven**: Easy to experiment with different settings
4. **GPU Optimization**: Built-in VRAM calculation and optimization
5. **Docker Support**: Easy deployment and reproducibility

## Performance Considerations

- **Batch Size Optimization**: Automatically calculated based on VRAM
- **Data Sharding**: Supports parallel data loading
- **Mixed Precision Training**: Reduces memory and speeds up training
- **Gradient Accumulation**: Allows larger effective batch sizes
