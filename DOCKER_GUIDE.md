# Docker Deployment Guide

## Quick Start

### Prerequisites

- Docker & Docker Compose installed
- **8–12 GB RAM allocated to Docker** (Settings → Resources) — required for build
- NVIDIA Docker runtime (for GPU support)
- NVIDIA GPU with CUDA 12.1 support

### Build the Docker Image

```bash
# Build the image
docker build -t vlm-challenge:latest .

# Or use docker-compose to build
docker-compose build
```

### Run with Docker Compose

```bash
# Start the API service
docker-compose up -d

# View logs
docker-compose logs -f vlm-api

# Stop the service
docker-compose down
```

### Test the API

```bash
# Check health
curl http://localhost:8000/health

# View API docs
open http://localhost:8000/docs

# Get available models
curl http://localhost:8000/models

# Upload a video for prediction
curl -X POST -F "file=@your_video.mp4" http://localhost:8000/predict
```

### Environment Variables

- `CUDA_VISIBLE_DEVICES`: Which GPU to use (default: 0)
- `LOG_LEVEL`: Logging level (default: INFO)
- `PYTHONUNBUFFERED`: Set to 1 for unbuffered output

### Volumes & Mounting

The docker-compose.yml mounts:

- `./data` → `/app/data` - Synthetic video data
- `./checkpoints` → `/app/checkpoints` - LoRA fine-tuned models
- `./training_data_samples` → `/app/training_data_samples` - Training samples

### GPU Configuration

To use a different GPU or multiple GPUs:

```bash
# Single GPU (0)
CUDA_VISIBLE_DEVICES=0 docker-compose up

# Multiple GPUs (0, 1)
CUDA_VISIBLE_DEVICES=0,1 docker-compose up
```

### Direct Docker Run (No Compose)

```bash
docker run --rm \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -e CUDA_VISIBLE_DEVICES=0 \
  vlm-challenge:latest
```

### Troubleshooting

**Build fails with "Bus error (core dumped)" or "rpc error: EOF":**

This is a memory exhaustion issue during `pip install`. Fixes:

1. **Increase Docker memory:** Docker Desktop → Settings → Resources → Memory → set to **8–12 GB**
2. The Dockerfile uses staged pip installs with `--prefer-binary` to reduce peak memory
3. Close other apps to free RAM before building

**GPU not detected:**

```bash
# Check NVIDIA GPU status
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi

# Ensure nvidia-docker runtime is installed
docker ps --no-trunc | grep nvidia
```

**Model loading fails:**

- Ensure checkpoints are in `/checkpoints` directory
- Check HuggingFace model is accessible or cached
- Monitor memory: `nvidia-smi` inside container

**Port already in use:**

```bash
# Change port in docker-compose.yml or
docker-compose up -e PORT=8001 vlm-api
```

### Production Deployment

For production, modify Dockerfile:

1. Remove `--reload` flag from CMD
2. Add proper logging configuration
3. Use `gunicorn` with multiple workers:

```dockerfile
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", \
     "-b", "0.0.0.0:8000", "src.api.main:app"]
```

### Monitoring

```bash
# Container stats
docker stats vlm-api

# View logs in real-time
docker-compose logs -f vlm-api --tail 50

# Check health status
docker-compose ps
```
