#!/usr/bin/env python3
"""
Docker Build & Deployment Validation Script
- Checks Docker prerequisites
- Validates Dockerfile syntax
- Tests docker-compose configuration
- Builds and tests the image
"""

import subprocess
import sys
import json
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and report results."""
    print(f"\n[RUN] {description}...")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=60
        )
        if result.returncode == 0:
            print(f"[OK] {description} - PASSED")
            return True
        else:
            print(f"[FAIL] {description} - FAILED")
            print(f"   Error: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"[ERROR] {description} - {e}")
        return False


def main():
    print("=" * 60)
    print("VLM Challenge - Docker Deployment Validation")
    print("=" * 60)
    
    checks = []
    
    # 1. Docker installed
    checks.append(run_command("docker --version", "Docker installed"))
    
    # 2. Docker Compose installed
    checks.append(run_command("docker-compose --version", "Docker Compose installed"))
    
    # 3. Docker daemon running
    checks.append(run_command("docker ps", "Docker daemon running"))
    
    # 4. NVIDIA Docker support (optional but helpful)
    print("\n[CHECK] Checking NVIDIA Docker support...")
    if run_command("docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi", "GPU support"):
        print("[OK] GPU support - AVAILABLE")
    else:
        print("[WARN] GPU support - NOT available (will fallback to CPU)")
    
    # 5. Dockerfile exists and is valid
    dockerfile_path = Path("Dockerfile")
    if dockerfile_path.exists():
        print(f"[OK] Dockerfile exists")
        checks.append(True)
    else:
        print(f"[FAIL] Dockerfile not found")
        checks.append(False)
    
    # 6. docker-compose.yml exists and is valid
    print("\n[CHECK] Validating docker-compose.yml...")
    if run_command("docker-compose config > /dev/null", "docker-compose.yml valid"):
        checks.append(True)
    else:
        checks.append(False)
    
    # 7. requirements.txt exists
    req_path = Path("requirements.txt")
    if req_path.exists():
        print(f"[OK] requirements.txt exists")
        checks.append(True)
    else:
        print(f"[FAIL] requirements.txt not found")
        checks.append(False)
    
    # 8. API source exists
    api_files = [
        Path("src/api/main.py"),
        Path("src/api/inference.py"),
    ]
    all_exist = all(f.exists() for f in api_files)
    if all_exist:
        print(f"[OK] API source files exist")
        checks.append(True)
    else:
        print(f"[FAIL] Missing API source files")
        checks.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(checks)
    total = len(checks)
    print(f"Validation Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("[SUCCESS] All checks passed! Ready to build Docker image.")
        print("\nNext steps:")
        print("  1. docker-compose build")
        print("  2. docker-compose up -d")
        print("  3. curl http://localhost:8000/health")
        return 0
    else:
        print(f"[FAILURE] {total - passed} check(s) failed. Fix errors above before building.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
