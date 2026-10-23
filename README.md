# Person-Aware Video Event Extractor

A modular, production-grade computer vision pipeline in Python that processes local video files (`.mp4`, `.webm`, `.avi`, `.mkv`) from fixed CCTV or moving handheld cameras and extracts clean, non-redundant screenshots when:
1. **A genuinely new person appears** (confirmed across multiple observations).
2. **A previously identified person returns** after a configurable absence duration (default: `60s`).
3. **Multiple new people arrive simultaneously**, grouping them into a single optimal screenshot.

Continuously visible people, bounding-box shifts, camera pans, temporary occlusions, and short-term tracker ID switches are automatically suppressed so they never generate duplicate screenshots.

---

## Quick Start (Windows & Linux)

### Windows (PowerShell)
```powershell
git clone https://github.com/MaarijRehman/chronos-reid.git
cd chronos-reid
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
.\scripts\setup_windows.ps1          # Add -UseCuda if you have an NVIDIA RTX GPU
```

### Linux (Ubuntu / Debian)
```bash
git clone https://github.com/MaarijRehman/chronos-reid.git
cd chronos-reid
chmod +x scripts/setup_linux.sh
./scripts/setup_linux.sh             # Add --cpu to force CPU-only PyTorch
```

### Verify Tests & Run Benchmark
```bash
python -m pytest -v
python benchmark.py
```

---

## Usage & Scenario Presets (`configs/`)

```bash
# Standard run
python main.py --video "input.mp4"

# Scenario presets
python main.py --video "input.mp4" --config "configs/default.yaml"
python main.py --video "input.mp4" --config "configs/crowd.yaml"
python main.py --video "input.mp4" --config "configs/low_light_cctv.yaml"
python main.py --video "input.mp4" --config "configs/moving_handheld.yaml"

# CLI overrides
python main.py --video "input.mp4" --mode full
python main.py --video "input.mp4" --mode sampled --sample-fps 10
python main.py --video "input.mp4" --return-threshold 120 --save-debug
```

---

## Documentation
- **[Complete Windows & Linux Commands Cheat Sheet](docs/COMMANDS_CHEATSHEET.md)**
- **[Scenario & Threshold Tuning Guide](docs/TUNING_GUIDE.md)**

## Author
Developed and maintained by **[Maarij Rehman](https://github.com/MaarijRehman)**.
