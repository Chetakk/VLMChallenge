# VLM Challenge - Setup & Execution Guide

## 📋 What's Been Generated

I've created a **complete end-to-end pipeline** for Qwen2.5-VL fine-tuning on Kaggle T4 with synthetic data.

### Generated Files:

✅ **Data Generation**

- `generate_synthetic_data.py` - Creates 100 synthetic warehouse videos (5sec each)

✅ **Core Pipeline**

- `src/training/dataset.py` - PyTorch Dataset loader (complete)
- `src/training/vram_math.py` - VRAM optimization calculations
- `src/training/finetune_config.py` - Configuration management

✅ **Inference & API**

- `src/api/inference.py` - Qwen2.5-VL model loading & prediction
- `src/api/main.py` - FastAPI endpoints (complete)

✅ **Training Notebook**

- `notebooks/qwen_training.ipynb` - Kaggle notebook ready to run

✅ **Deployment**

- `Dockerfile` (updated) - Container image with CUDA 12.1
- `docker-compose.yml` - Multi-service orchestration

---

## 🎯 Execution Plan (Step by Step)

### STEP 1: Generate Synthetic Data (5 min)

**Location:** Your local machine or Kaggle

```bash
cd "c:\Users\Chetak\Documents\GitHub\projects\VLM Challenge\VLMChallengeCode"
python generate_synthetic_data.py
```

**Output:**

- Creates `data/synthetic/videos/` with 100 mp4 files
- Creates `data/synthetic/annotations.json` with metadata

**What to expect:**

```
🎬 Generating synthetic warehouse videos...
📁 Output directory: data/synthetic
🎥 Creating 100 clips (5s @ 25fps each)
✅ Created 10/100 clips
...
🎉 Synthetic data generation complete!
```

---

### STEP 2: Test Pipeline Locally (3 min)

Verify everything works before going to Kaggle:

```bash
python -m src.test_pipeline
```

**This will:**

- Load fake operations
- Build clips with temporal boundaries
- Write WebDataset shards
- Verify pipeline components work

---

### STEP 3: Copy to Kaggle & Train (20 min)

1. **Create Kaggle Notebook:**
   - Go to https://kaggle.com/code
   - Click "New Notebook"
   - Select "Python" environment

2. **Upload Your Code:**
   - Upload the entire project folder as a Dataset:
     ```bash
     kaggle datasets create -f <your-project-folder>
     ```
   - Or copy/paste files directly into the notebook

3. **Run Training Notebook:**
   - Open `notebooks/qwen_training.ipynb`
   - Copy content to your Kaggle notebook
   - **Cell 1-5:** Install dependencies + load model (5 min)
   - **Cell 6-8:** Setup training (2 min)
   - **Cell 9:** Run `trainer.train()` (8-10 min on T4)
   - **Cell 10:** Evaluate metrics (1 min)

---

### STEP 4: Generate Predictions (5 min)

After training completes:

```python
# In Kaggle notebook Cell 11:

from src.api.inference import QwenVLInference

# Load fine-tuned model
inference = QwenVLInference(
    model_path="outputs/qwen-lora/checkpoint-xxx",
    use_lora=True
)
inference.load_model()

# Run inference on test set
predictions = inference.batch_predict(test_frames_list)

# Save predictions
import json
with open("predictions.json", "w") as f:
    json.dump(predictions, f)
```

---

### STEP 5: Evaluate Metrics (2 min)

```python
# In notebook:

from src.evaluation.evaluator import evaluate_model

results = evaluate_model(
    gt_dict=test_annotations,
    pred_dict=predictions
)

print("Results:")
print(f"OCA (Operation Classification):    {results['OCA']:.4f}")
print(f"tIoU@0.5 (Temporal Boundaries):    {results['tIoU@0.5']:.4f}")
print(f"AA@1 (Anticipation Accuracy):      {results['AA@1']:.4f}")
```

---

### STEP 6: Deploy API Locally (3 min)

```bash
# Terminal 1: Start API
docker-compose up vlm-api

# Terminal 2: Test API
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"clip_id": "test_001", "use_baseline": true}'
```

---

## 📊 Expected Performance

### Baseline (Zero-shot):

- **OCA:** ~0.33 (random operation selection)
- **tIoU@0.5:** ~0.40-0.50 (based on sequence rules)
- **AA@1:** ~0.20 (predicting next operation is hard)

### After Fine-tuning (Target):

- **OCA:** 0.70-0.85 (model learns operation patterns)
- **tIoU@0.5:** 0.65-0.80 (learns temporal boundaries)
- **AA@1:** 0.55-0.75 (learns procedural logic!)

**Key Insight:** The AA@1 improvement is what separates real temporal understanding from visual classification.

---

## 💾 Resource Requirements

| Step              | Time   | VRAM | Notes                 |
| ----------------- | ------ | ---- | --------------------- |
| Data Gen          | 5 min  | CPU  | Can run locally       |
| Pipeline Test     | 2 min  | 2GB  | Verify on laptop      |
| Training (Kaggle) | 10 min | 16GB | Qwen2.5-VL fits in T4 |
| Inference         | 5 min  | 8GB  | Per-batch             |
| API Deployment    | <1 min | 12GB | Lazy-loaded           |

**Total on Kaggle:** ~30-40 minutes for complete pipeline

---

## 🔧 Troubleshooting

### Issue: OOM Error in Kaggle

**Solution:** Reduce batch size further

```python
VRAM_CONFIG["batch_size"] = 1
VRAM_CONFIG["gradient_accumulation_steps"] = 32
```

### Issue: Model not downloading

**Solution:** Pre-download on Kaggle

```python
from transformers import AutoProcessor, Qwen2_5VLForConditionalGeneration
Qwen2_5VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-2B-Instruct")
```

### Issue: tIoU not improving

**Solution:** Add more diverse synthetic videos with varying operation boundaries

---

## 📁 Final Project Structure

```
VLMChallengeCode/
├── data/
│   ├── synthetic/
│   │   ├── videos/ (100 mp4 files)
│   │   └── annotations.json
│   ├── processed/ (extracted frames)
│   └── shards/ (WebDataset shards)
├── notebooks/
│   ├── finetune.ipynb (old)
│   └── qwen_training.ipynb (NEW - use this)
├── outputs/
│   └── qwen-lora/
│       ├── checkpoint-500/
│       ├── adapter_config.json
│       └── adapter_model.bin
├── src/
│   ├── data/ (pipeline)
│   ├── training/ (complete implementations)
│   ├── evaluation/ (metrics)
│   └── api/ (FastAPI)
├── generate_synthetic_data.py (NEW)
├── Dockerfile (updated)
├── docker-compose.yml (ready)
└── requirements.txt (ready)
```

---

## ✅ Checklist to Complete

- [ ] Run `python generate_synthetic_data.py`
- [ ] Run `python -m src.test_pipeline`
- [ ] Copy `qwen_training.ipynb` to Kaggle
- [ ] Run training (note final checkpoint path)
- [ ] Generate test predictions
- [ ] Compute metrics (OCA, tIoU, AA)
- [ ] Save fine-tuned model weights
- [ ] Test API locally
- [ ] Document results in ARCHITECTURE.md
- [ ] Push to GitHub

---

## 🚀 Next Command To Run

```bash
cd "c:\Users\Chetak\Documents\GitHub\projects\VLM Challenge\VLMChallengeCode"
python generate_synthetic_data.py
```

**This will START your pipeline.** After it completes, let me know the output!
