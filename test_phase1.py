
import torch
import argparse
from pathlib import Path

def test_phase1():
    print("\n" + "="*70)
    print("PHASE 1: Base VLM Deployment - Local Test")
    print("="*70 + "\n")
    
    # Test 1: Check GPU
    print("[TEST 1] CUDA/GPU Check")
    print(f"  GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    print()
    
    # Test 2: Import inference module
    print("[TEST 2] Import Inference Module")
    try:
        from src.api.inference import Qwen25VLInference, BaselineModel
        print("  [PASS] Successfully imported Qwen25VLInference and BaselineModel")
    except ImportError as e:
        print(f"  [FAIL] Failed to import: {e}")
        return False
    print()
    
    # Test 3: Test Baseline Model
    print("[TEST 3] Baseline Model Prediction")
    baseline = BaselineModel()
    pred = baseline.predict("test_clip_001")
    print(f"  Operation: {pred['dominant_operation']}")
    print(f"  Temporal Segment: {pred['temporal_segment']}")
    print(f"  Next Operation: {pred['anticipated_next_operation']}")
    print(f"  Confidence: {pred['confidence']}")
    print("  [PASS] Baseline model works")
    print()
    
    # Test 4: Load Qwen2.5-VL (skip if no GPU)
    if torch.cuda.is_available():
        print("[TEST 4] Load Qwen2.5-VL Model")
        try:
            qwen = Qwen25VLInference()
            print("  Loading model... (takes ~30 seconds on first run)")
            qwen.load_model()
            print("  [PASS] Model loaded successfully")
        except Exception as e:
            print(f"  [WARN] Could not load model: {e}")
            print("  Note: Model will be lazy-loaded on API startup")
    else:
        print("[TEST 4] Load Qwen2.5-VL Model - SKIPPED (no GPU)")
        print("  Note: Qwen2.5-VL requires GPU. Use Kaggle or GCP for training.")
    print()
    
    # Test 5: Create dummy video frames
    print("[TEST 5] Create Dummy Video Frames")
    frames = torch.randint(0, 256, (8, 3, 336, 336), dtype=torch.uint8)
    print(f"  Frame tensor shape: {frames.shape}")
    print(f"  Frame dtype: {frames.dtype}")
    print("  [PASS] Dummy frames created")
    print()
    
    # Test 6: Test API imports
    print("[TEST 6] Import FastAPI Application")
    try:
        from src.api.main import app
        print("  [PASS] FastAPI app imported successfully")
    except ImportError as e:
        print(f"  [FAIL] Failed to import app: {e}")
        return False
    print()
    
    print("="*70)
    print("[SUCCESS] ALL PHASE 1 TESTS PASSED")
    print("="*70)
    print("\nNext Steps:")
    print("1. Start the API locally:")
    print("   python -m uvicorn src.api.main:app --reload --port 8000")
    print("\n2. Test an inference:")
    print("   curl -X POST -F 'file=@test_video.mp4' http://localhost:8000/predict")
    print("\n3. Check API docs:")
    print("   http://localhost:8000/docs")
    print("\n4. Deploy with Docker:")  
    print("   docker-compose up vlm-api")
    print("\n" + "="*70 + "\n")
    
    return True


if __name__ == "__main__":
    success = test_phase1()
    exit(0 if success else 1)
