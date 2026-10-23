#!/usr/bin/env bash
set -e
echo "Setting up Person-Aware Video Event Extractor (Linux)..."
PYTHON_BIN=${PYTHON_BIN:-python3}
if [ ! -d ".venv" ]; then $PYTHON_BIN -m venv .venv; fi
source .venv/bin/activate
python -m pip install --upgrade pip
pip install --only-binary=:all: numpy opencv-python ultralytics PyYAML pytest psutil gdown lap
if [ "$1" == "--cpu" ]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
    pip install torch torchvision
fi
pip install torchreid || echo "Using built-in Torchvision MobileNetV3 ReID."
python -c "import cv2, torch, ultralytics, numpy, yaml; print('SUCCESS! NumPy:', numpy.__version__, '| PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
