

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
import torch
import tempfile
from pathlib import Path

# Import inference engines
try:
    from src.api.inference import Qwen25VLInference, BaselineModel
except ImportError:
    from inference import Qwen25VLInference, BaselineModel

app = FastAPI(
    title="VLM Challenge - Phase 1 API",
    description="Qwen2.5-VL for temporal warehouse operation understanding",
    version="1.0.0",
)

# Global model instances
qwen_engine: Optional[Qwen25VLInference] = None
baseline_model: Optional[BaselineModel] = None


# Request/Response Models
class VideoUploadRequest(BaseModel):
    """Video upload and inference request."""
    use_baseline: bool = Field(False, description="Use baseline model instead of Qwen2.5-VL")
    use_lora: bool = Field(False, description="Use LoRA-adapted checkpoint")


class PredictionResponse(BaseModel):
    """Required JSON output schema (Phase 1)."""
    clip_id: str = Field("unknown", description="Unique clip identifier")
    dominant_operation: str = Field(..., description="Main operation in the clip")
    temporal_segment: Dict[str, int] = Field(..., description="Start and end frame numbers")
    anticipated_next_operation: str = Field(..., description="Predicted next operation")
    confidence: float = Field(0.0, description="Confidence score [0,1]")


class HealthResponse(BaseModel):
    """API health status."""
    status: str
    model_loaded: bool
    gpu_available: bool
    gpu_count: int


# Startup/Shutdown
@app.on_event("startup")
async def startup_event():
    global baseline_model, qwen_engine
    
    print("\n" + "="*60)
    print("[STARTUP] Phase 1: VLM Deployment")
    print("="*60)
    
    # Initialize baseline
    baseline_model = BaselineModel()
    print("[OK] Baseline model initialized")
    
    # Note: Qwen2.5-VL will be lazy-loaded on first inference request
    print("[INFO] Qwen2.5-VL will load on first inference request")
    print("="*60 + "\n")


# Health & Status Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    
    return HealthResponse(
        status="healthy",
        model_loaded=qwen_engine is not None,
        gpu_available=gpu_available,
        gpu_count=gpu_count,
    )


@app.get("/status")
async def status():
    return {
        "phase": "1_base_deployment",
        "models_available": {
            "baseline": "ready",
            "qwen2.5-vl": "lazy-loaded" if qwen_engine is None else "loaded",
        },
        "gpu_info": {
            "available": torch.cuda.is_available(),
            "count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
    }


# Inference Endpoints
@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="Video file (mp4, avi, etc)"),
    use_baseline: bool = Query(False, description="Use baseline model"),
    clip_id: str = Query("", description="Clip identifier (optional)"),
):
    try:
        # Save uploaded file temporarilythrough
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Extract frames from video
        try:
            import decord
            vr = decord.VideoReader(tmp_path)
            # Sample 8 frames uniformly
            frame_indices = torch.linspace(0, len(vr)-1, 8).long().tolist()
            frames_list = [vr[i].asnumpy() for i in frame_indices]
            frames = torch.from_numpy(torch.stack([torch.from_numpy(f) for f in frames_list]).float())
        except ImportError:
            # Fallback to OpenCV if decord not available
            import cv2
            cap = cv2.VideoCapture(tmp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = torch.linspace(0, total_frames-1, 8).long().tolist()
            
            frames_list = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_list.append(torch.from_numpy(frame).float() / 255.0)
            
            cap.release()
            frames = torch.stack(frames_list) if frames_list else torch.randn(8, 3, 336, 336)
        
        # Run inference
        if use_baseline:
            result = baseline_model.predict(clip_id or "unknown")
        else:
            global qwen_engine
            if qwen_engine is None:
                qwen_engine = Qwen25VLInference()
                qwen_engine.load_model()
            
            result = qwen_engine.predict(frames)
        
        result["clip_id"] = clip_id or "unknown"
        
        # Cleanup
        Path(tmp_path).unlink()
        
        return PredictionResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/batch_predict")
async def batch_predict(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        result = await predict(file, use_baseline=False)
        results.append(result.dict())
    return {"predictions": results}


# Model Management
@app.get("/models")
async def list_models():
    return {
        "models": [
            {
                "name": "baseline",
                "type": "rule-based",
                "phase": 1,
                "description": "Zero-shot rule-based baseline",
                "ready": True,
            },
            {
                "name": "qwen2.5-vl-2b",
                "type": "vision-language",
                "phase": 1,
                "description": "Qwen2.5-VL base model (zero-shot)",
                "ready": qwen_engine is not None,
                "parameters": "2.4B",
                "from_huggingface": "Qwen/Qwen2.5-VL-2B-Instruct",
            }
        ],
        "phase": "1_base_deployment",
        "next_phase": "2_temporal_data_pipeline",
    }


# Root
@app.get("/")
async def root():
    return {
        "message": "VLM Challenge - Phase 1 API",
        "phase": "Base VLM Deployment (0-3 hours)",
        "objective": "Deploy base Qwen2.5-VL without fine-tuning for zero-shot inference",
        "endpoints": {
            "POST /predict": "Run inference on single video",
            "POST /batch_predict": "Batch inference on multiple videos",
            "GET /health": "Health check",
            "GET /status": "Detailed status",
            "GET /models": "List available models",
        },
        "deliverables": ("docker-compose.yml", "Dockerfile"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


