param([switch]$UseCuda = $false)
$ErrorActionPreference = "Stop"
Write-Host "Setting up Person-Aware Video Event Extractor (Windows)..." -ForegroundColor Cyan
if (-not (Test-Path ".venv")) { python -m venv .venv }
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install --only-binary=:all: numpy opencv-python ultralytics PyYAML pytest psutil gdown lap
if ($UseCuda) {
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
} else {
    pip install torch torchvision
}
python -c "import cv2, torch, ultralytics, numpy, yaml; print('SUCCESS! NumPy:', numpy.__version__, '| PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
