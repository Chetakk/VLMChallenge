# VLM Challenge — Temporal Operation Intelligence

> A Vision-Language Model system that understands warehouse packaging operations from video: what's happening now, when it starts and ends, and what comes next.

---

## What does this do?

Given a short video clip of a warehouse worker, this system predicts:

| Output | Description |
|--------|-------------|
| **Dominant operation** | What the worker is doing (e.g., Tape, Pack, Label) |
| **Temporal segment** | Start and end frames of that operation |
| **Anticipated next operation** | What will likely happen next in the workflow |

**Metrics:** OCA (operation accuracy), tIoU@0.5 (temporal localization), AA@1 (anticipation accuracy)

### Operation classes (10)

The model predicts from these warehouse operations:

| Class | Description |
|-------|-------------|
| Box Setup | Preparing the box for packing |
| Inner Packing | Placing items inside the box |
| Tape | Sealing with tape |
| Put Items | Adding items to the box |
| Pack | General packing activity |
| Wrap | Wrapping items |
| Label | Applying labels |
| Final Check | Final inspection |
| Idle | No active operation |
| Unknown | Unrecognized |

### Video format

- **Clip length:** ~5 seconds
- **Frame rate:** 25 fps (OpenPack standard)
- **Resolution:** 480×640 (Kinect) or 336×336 (model input, auto-scaled)
- **Formats:** MP4, AVI supported

---

## Quick start

### Option A: Docker (recommended)

```bash
# 1. Build the image (allow 8–12 GB RAM for Docker)
docker build -t vlm-challenge:latest .

# 2. Start the API
docker-compose up -d

# 3. Test it
curl http://localhost:8000/health
```

**First prediction:** Upload a video via the API docs at http://localhost:8000/docs, or:

```bash
curl -X POST -F "file=@your_video.mp4" http://localhost:8000/predict

# No GPU? Use baseline model:
curl -X POST -F "file=@your_video.mp4" "http://localhost:8000/predict?use_baseline=true"
```

> 💡 **Build fails with "Bus error"?** Increase Docker memory to 8–12 GB. See [DOCKER_GUIDE.md](DOCKER_GUIDE.md) for details.

---

### Option B: Local setup

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd VLMChallengeCode

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the API
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/docs to try the API.

---

## Project structure

```
VLMChallengeCode/
├── src/
│   ├── api/           # FastAPI inference server
│   ├── data/          # Annotation parsing, clip building, frame sampling
│   ├── training/      # Dataset, VRAM math, fine-tune config
│   └── evaluation/    # OCA, tIoU, AA@1 metrics
├── notebooks/         # finetune.ipynb (guide), qwen_training.ipynb (Kaggle)
├── data/              # Raw/processed data (mount for Docker)
├── checkpoints/       # LoRA checkpoints (mount for Docker)
├── evaluate.py        # Run evaluation → results.json
├── generate_synthetic_data.py
├── create_training_samples.py
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Common tasks

### Run evaluation

```bash
python evaluate.py
# Output: results.json with OCA, tIoU@0.5, AA@1
```

### Generate synthetic training data

```bash
python generate_synthetic_data.py
python create_training_samples.py
```

### Verify setup

```bash
python test_phase1.py
```

---

## Training on Kaggle

Fine-tune Qwen2.5-VL on Kaggle's free T4 GPU:

1. **Upload training data** — Create a Kaggle dataset from `training_data_samples/` (20 videos + index.json)
2. **Upload notebook** — Copy `notebooks/qwen_training.ipynb` into a new Kaggle notebook
3. **Enable GPU** — Set notebook to use T4 (Settings → Accelerator)
4. **Run all cells** — Training takes ~8–10 hours
5. **Download checkpoint** — Save the LoRA checkpoint and place it in `checkpoints/`

See [PHASE3_KAGGLE_GUIDE.txt](PHASE3_KAGGLE_GUIDE.txt) for step-by-step instructions.

---

## Using the fine-tuned (LoRA) model

Place your LoRA checkpoint in `checkpoints/` (e.g. `checkpoints/qwen-lora-checkpoint/`). The Docker compose mounts this folder. The inference engine supports loading LoRA adapters; ensure the API is configured to use the checkpoint path when initializing the model.

---

## Data source

- **Synthetic data (default):** Run `generate_synthetic_data.py` to create 100 procedural warehouse videos. No external download required.
- **OpenPack (real data):** Download from [Zenodo](https://zenodo.org/records/11059235). Use subjects U0101–U0106 for training, U0107 for validation, U0108 for test. Requires `openpack-toolkit` for annotation parsing.

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health and GPU status |
| GET | `/models` | Available models (baseline, Qwen2.5-VL) |
| POST | `/predict` | Upload video → get predictions |
| POST | `/batch_predict` | Upload multiple videos |

**Predict request:** `POST /predict` with `file` (video) as multipart form data.

**Query parameters:**
- `use_baseline=true` — Use rule-based baseline (no GPU, fast)
- `clip_id` — Optional clip identifier for the response

**Response example:**
```json
{
  "clip_id": "U0108_S0500_t0035",
  "dominant_operation": "Tape",
  "temporal_segment": { "start_frame": 14, "end_frame": 98 },
  "anticipated_next_operation": "Put Items",
  "confidence": 0.87
}
```

---

## Requirements

- **Python** 3.10+
- **GPU** (optional): NVIDIA with CUDA 12.1 for full model inference
- **Docker** (optional): 8–12 GB RAM allocated for image build

---

## Documentation

| Document | Purpose |
|----------|---------|
| [DOCKER_GUIDE.md](DOCKER_GUIDE.md) | Docker build, run, and troubleshooting |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Model choice, frame sampling, failure analysis |
| [AGENTS.md](AGENTS.md) | AI agent usage and development log |
| [SETUP_GUIDE.md](SETUP_GUIDE.md) | Step-by-step execution plan |
| [PHASE3_KAGGLE_GUIDE.txt](PHASE3_KAGGLE_GUIDE.txt) | Kaggle fine-tuning walkthrough |
| [DOCKER_SETUP_SUMMARY.md](DOCKER_SETUP_SUMMARY.md) | Docker deployment checklist |
| [project.txt](project.txt) | Full assignment specification |

---

## License

TBD
