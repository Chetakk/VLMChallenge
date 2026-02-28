FROM ubuntu:22.04

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    ffmpeg \
    curl \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Update pip
RUN python3 -m pip install --upgrade pip

# Install Python dependencies in stages to avoid OOM (Bus error) during build
# Stage 1: Light packages
RUN python3 -m pip install --no-cache-dir --prefer-binary \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    pillow \
    numpy \
    webdataset \
    pydantic \
    python-dotenv

# Stage 2: PyTorch (pre-built wheels, large download)
RUN python3 -m pip install --no-cache-dir --prefer-binary \
    torch \
    torchvision

# Stage 3: Transformers and PEFT
RUN python3 -m pip install --no-cache-dir --prefer-binary \
    transformers \
    peft

# Stage 4: OpenCV and decord
RUN python3 -m pip install --no-cache-dir --prefer-binary \
    opencv-python \
    decord

# Stage 5: bitsandbytes last (often heaviest, can compile)
RUN python3 -m pip install --no-cache-dir --prefer-binary \
    bitsandbytes

# Copy application code
COPY src/ ./src/

# Create directories for data and checkpoints (will be mounted via volumes)
RUN mkdir -p /app/data /app/checkpoints /app/training_data_samples

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API server
CMD ["python3", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
