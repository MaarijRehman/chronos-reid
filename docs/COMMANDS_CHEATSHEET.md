# Step-by-Step Command Reference (Windows & Linux)

This document contains every command needed to install, test, benchmark, and run the **Person-Aware Video Event Extractor** across Windows PowerShell and Linux Bash.

---

## 1. Environment Setup Commands

### Windows (PowerShell)
```powershell
git clone https://github.com/MaarijRehman/chronos-reid.git
cd chronos-reid

# Option A: One-Click Automated Setup (CPU)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
.\scripts\setup_windows.ps1

# Option B: One-Click Automated Setup (NVIDIA RTX GPU / CUDA 12.6)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
.\scripts\setup_windows.ps1 -UseCuda

# Option C: Manual Step-by-Step Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install --only-binary=:all: numpy opencv-python ultralytics PyYAML pytest psutil gdown lap
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Linux (Ubuntu / Debian Bash)
```bash
sudo apt update && sudo apt install -y python3-venv python3-pip libgl1 libglib2.0-0
git clone https://github.com/MaarijRehman/chronos-reid.git
cd chronos-reid

# Option A: One-Click Automated Setup (GPU / Auto)
chmod +x scripts/setup_linux.sh
./scripts/setup_linux.sh

# Option B: One-Click Automated Setup (CPU-Only)
chmod +x scripts/setup_linux.sh
./scripts/setup_linux.sh --cpu
```

---

## 2. Verification & Benchmarking Commands

```bash
# Run the 18-Scenario Automated Unit Test Suite
python -m pytest -v

# Run the Synthetic Smoke Test & Efficiency Benchmark
python benchmark.py

# Run Benchmark on Sample Video with 10s Return Threshold
python benchmark.py --video "people_test.mp4" --return-threshold 10
```

---

## 3. Production Execution Commands

```bash
# Standard Execution (Default Config)
python main.py --video "input.mp4"

# Scenario Presets
python main.py --video "crowd_clip.mp4" --config "configs/crowd.yaml"
python main.py --video "night_cctv.mp4" --config "configs/low_light_cctv.yaml"
python main.py --video "phone_walk.mp4" --config "configs/moving_handheld.yaml"

# CLI Overrides
python main.py --video "input.mp4" --mode full
python main.py --video "input.mp4" --mode sampled --sample-fps 10
python main.py --video "input.mp4" --return-threshold 120
python main.py --video "input.mp4" --grouping-window 2.0
python main.py --video "input.mp4" --save-debug
```
