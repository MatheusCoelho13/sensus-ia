# PowerShell script to setup GPU training - Windows 11 CUDA 13.1
# Run: powershell -ExecutionPolicy Bypass -File fix_gpu_setup.ps1

Write-Host "=== GPU Setup for Assistiva IA ===" -ForegroundColor Cyan

# Step 1: Remove old venv
Write-Host "`nStep 1: Removing old venv..." -ForegroundColor Yellow
if (Test-Path ".venv") {
    Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 500
    Write-Host "✓ Old venv removed" -ForegroundColor Green
} else {
    Write-Host "✓ No existing venv" -ForegroundColor Green
}

# Step 2: Create venv with Python 3.11
Write-Host "`nStep 2: Creating venv with Python 3.11..." -ForegroundColor Yellow
try {
    & py -3.11 -m venv .venv
    Write-Host "✓ Venv created with Python 3.11" -ForegroundColor Green
} catch {
    Write-Host "✗ Failed to create venv with Python 3.11" -ForegroundColor Red
    Write-Host "Make sure Python 3.11 is installed: py -3.11 --version" -ForegroundColor Yellow
    exit 1
}

# Step 3: Upgrade pip
Write-Host "`nStep 3: Upgrading pip..." -ForegroundColor Yellow
& .venv\Scripts\python -m pip install --upgrade pip --quiet
Write-Host "✓ Pip upgraded" -ForegroundColor Green

# Step 4: Install PyTorch with CUDA 13.1
Write-Host "`nStep 4: Installing PyTorch with CUDA 13.1..." -ForegroundColor Yellow
& .venv\Scripts\python -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu131 --quiet
Write-Host "✓ PyTorch installed" -ForegroundColor Green

# Step 5: Verify CUDA
Write-Host "`nStep 5: Verifying CUDA support..." -ForegroundColor Yellow
$verify = & .venv\Scripts\python -c 'import torch; print(f"Version: {torch.__version__}"); print(f"CUDA: {torch.cuda.is_available()}")'
Write-Host $verify

# Step 6: Install requirements
Write-Host "`nStep 6: Installing project requirements..." -ForegroundColor Yellow
& .venv\Scripts\python -m pip install -r config/requirements.txt --quiet
Write-Host "✓ Requirements installed" -ForegroundColor Green

Write-Host "`n=== Setup Complete ===" -ForegroundColor Cyan
Write-Host "Ready to train GPU! Run:`n  make start_train_gpu`n" -ForegroundColor Green
