#!/bin/bash
# Simple GPU setup script for Windows 11 with Python 3.11
# Run: bash scripts/setup_py311_gpu.sh

set -e

echo "=== GPU Setup: Python 3.11 + CUDA 13.1 ==="

# Step 1: Remove old venv
echo -e "\n[1/6] Removing old venv..."
rm -rf .venv 2>/dev/null || true
sleep 1

# Step 2: Create venv with Python 3.11
echo "[2/6] Creating venv with Python 3.11..."
py -3.11 -m venv .venv

# Step 3: Upgrade pip
echo "[3/6] Upgrading pip..."
.venv/Scripts/python -m pip install --upgrade pip -q 2>/dev/null || true

# Step 4: Install PyTorch CUDA 12.1 (cu121)
echo "[4/6] Installing PyTorch with CUDA support..."
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q

# Step 5: Verify CUDA
echo "[5/6] Verifying CUDA support..."
.venv/Scripts/python -c "import torch; print('  PyTorch:', torch.__version__); print('  CUDA:', torch.cuda.is_available())"

# Step 6: Install project requirements
echo "[6/6] Installing project requirements..."
.venv/Scripts/python -m pip install -r config/requirements.txt -q

echo -e "\n✓ GPU Setup Complete!"
echo -e "✓ Ready to train with GPU\n"
echo "Run:"
echo "  make start_train_gpu"
echo ""
