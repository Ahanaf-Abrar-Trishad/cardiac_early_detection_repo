#!/usr/bin/env bash
set -euo pipefail

# Basic system prep (Ubuntu 20.04/22.04/24.04)
sudo apt update
sudo apt install -y build-essential git git-lfs wget unzip ffmpeg libjpeg-turbo8-dev     libpng-dev libtiff-dev libopenblas-dev

# (Recommended) Install Mambaforge if you don't have conda/mamba
if ! command -v mamba &> /dev/null; then
  echo "Installing Mambaforge..."
  wget -qO ~/Mambaforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
  bash ~/Mambaforge.sh -b -p $HOME/mambaforge
  eval "$($HOME/mambaforge/bin/conda shell.bash hook)"
  conda init
fi

# Create and activate environment
eval "$(conda shell.bash hook)"
conda deactivate || true
mamba create -y -n cardio-dl python=3.11 pip git cmake ninja pkg-config ffmpeg libjpeg-turbo libpng libtiff libopenblas
conda activate cardio-dl

# Install PyTorch with CUDA 12.4 wheels (no separate CUDA toolkit needed, only NVIDIA driver)
# If you need CPU-only, comment the CUDA line and uncomment the CPU line.
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Project Python deps
pip install -r requirements.txt

# Quick sanity checks
python - <<'PY'
import torch, platform, sys
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime:", getattr(torch.version, "cuda", None))
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
PY

echo "Environment ready. Activate with: conda activate cardio-dl"
