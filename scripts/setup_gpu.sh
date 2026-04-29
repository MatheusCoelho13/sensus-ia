#!/usr/bin/env bash
# Script de setup GPU (WSL / Linux / Git Bash)
# - detecta versão CUDA via nvidia-smi
# - cria/ativa .venv (Python 3.11 recomendado)
# - instala PyTorch compatível com CUDA e as dependências do projeto
# Uso: bash scripts/setup_gpu.sh
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "== Setup GPU para Assistiva IA =="

# Detect nvidia-smi and CUDA version
CUDA_VERSION=""
if command -v nvidia-smi >/dev/null 2>&1; then
  # Parse nvidia-smi output to find CUDA Version: X.X line
  # The format is usually: CUDA Version: XX.X
  CUDA_VERSION="$(nvidia-smi 2>/dev/null | awk '/CUDA Version:/ { print $NF }' | head -1 || true)"
  if [ -n "$CUDA_VERSION" ]; then
    echo "GPU NVIDIA detectada. CUDA Version: $CUDA_VERSION"
  else
    echo "nvidia-smi encontrado mas CUDA não detectado; tentando pytorch cu131 como padrão..." >&2
    CUDA_VERSION="13.1"  # Try cu131 as default for modern systems
  fi
else
  echo "nvidia-smi não encontrado — ambiente sem GPU NVIDIA detectada. Instalações seguirão em modo CPU." >&2
  CUDA_VERSION=""
fi

# Decide torch index URL
INDEX_URL=""
if [ -n "$CUDA_VERSION" ]; then
  case "$CUDA_VERSION" in
    13.1*) INDEX_TAG="cu131";;
    12.1*) INDEX_TAG="cu121";;
    12.*) INDEX_TAG="cu12";;
    11.8*) INDEX_TAG="cu118";;
    11.*) INDEX_TAG="cu11";;
    *) INDEX_TAG="cpu";;
  esac
else
  INDEX_TAG="cpu"
fi

if [ "$INDEX_TAG" = "cpu" ]; then
  echo "Instalando PyTorch (CPU-only)..."
  PYTORCH_INDEX="https://download.pytorch.org/whl/cpu"
else
  echo "Detectado CUDA "$CUDA_VERSION" -> usando roda PyTorch com tag $INDEX_TAG"
  PYTORCH_INDEX="https://download.pytorch.org/whl/$INDEX_TAG"
fi

# Find suitable python executable - PREFER Python 3.11
PY_EXEC=""
for p in python3.11 py.exe python3 python; do
  if command -v "$p" >/dev/null 2>&1; then
    # For py.exe, try to get version to prefer 3.11
    if [ "$p" = "py.exe" ]; then
      Vers="$("$p" -3.11 --version 2>/dev/null | awk '{print $2}' | head -c 4 || true)"
      if [ "$Vers" = "3.11" ]; then
        PY_EXEC="$p -3.11"
        break
      fi
    else
      Vers="$("$p" --version 2>/dev/null | awk '{print $2}' | head -c 4 || true)"
      if [ "$Vers" = "3.11" ] || [ "$Vers" = "3.12" ] || [ "$Vers" = "3.10" ]; then
        PY_EXEC="$p"
        break
      fi
    fi
  fi
done
# Fallback if no suitable version found
if [ -z "$PY_EXEC" ]; then
  for p in python3.12 python3.10 python3.9 python3 python; do
    if command -v "$p" >/dev/null 2>&1; then
      PY_EXEC="$p"
      break
    fi
  done
fi
if [ -z "$PY_EXEC" ]; then
  echo "Erro: nenhum python encontrado. Instale Python 3.11+ e reexecute." >&2
  exit 1
fi

# Create venv
if [ ! -d .venv ]; then
  echo "Criando virtualenv com $PY_EXEC ..."
  "$PY_EXEC" -m venv .venv
else
  echo ".venv já existe — pulando criação"
fi

# Activate venv in-script
# shellcheck disable=SC1091
if [ -f .venv/bin/activate ]; then
  . .venv/bin/activate
elif [ -f .venv/Scripts/activate ]; then
  . .venv/Scripts/activate
else
  echo "Arquivo de ativação do venv não encontrado" >&2
  exit 1
fi

python -m pip install --upgrade pip

# Install torch + torchvision
echo "== Limpando instalações anteriores de torch (se houver) e numpy/opencv (se existirem) =="
python -m pip uninstall -y torch torchvision torchaudio numpy opencv-python || true

# Remove possíveis leftovers em paths Windows/Unix dentro do venv (torch, numpy, opencv, e artefatos cp314)
if [ -d ".venv/Lib/site-packages" ]; then
  echo "Limpando .venv/Lib/site-packages/..."
  rm -rf .venv/Lib/site-packages/torch* .venv/Lib/site-packages/~orch* || true
  rm -rf .venv/Lib/site-packages/numpy* .venv/Lib/site-packages/*cp314* .venv/Lib/site-packages/cv2* .venv/Lib/site-packages/opencv_python* || true
fi
if [ -d ".venv/lib" ]; then
  echo "Limpando .venv/lib/..."
  rm -rf .venv/lib/*/site-packages/torch* .venv/lib/*/site-packages/~orch* || true
  rm -rf .venv/lib/*/site-packages/numpy* .venv/lib/*/site-packages/*cp314* .venv/lib/*/site-packages/cv2* || true
fi

echo "== Instalando PyTorch ($INDEX_TAG) =="
if [ "$INDEX_TAG" = "cpu" ]; then
  pip install --no-cache-dir torch torchvision --index-url "$PYTORCH_INDEX"
else
  pip install --no-cache-dir torch torchvision --index-url "$PYTORCH_INDEX"
fi

# Verify torch installation
echo "== Verificando PyTorch e CUDA =="
python - <<'PY'
import sys
try:
    import torch
    print('torch', torch.__version__)
    try:
        print('cuda_available', torch.cuda.is_available())
        if torch.cuda.is_available():
            print('device name:', torch.cuda.get_device_name(0))
    except Exception as e:
        print('cuda check failed:', e)
except Exception as e:
    print('torch import failed:', e)
    sys.exit(1)
PY

# Install project requirements
if [ -f config/requirements.txt ]; then
  # Ensure numpy/opencv/pillow are installed first to avoid import-time binary mismatches
  echo "== Instalando numpy, opencv-python, Pillow (compatíveis com numpy nova) =="
  pip install --no-cache-dir --force-reinstall numpy opencv-python Pillow || true
  echo "== Instalando dependências do projeto =="
  pip install --no-cache-dir -r config/requirements.txt
else
  echo "Atenção: config/requirements.txt não encontrado; pulando instalação de requirements" >&2
fi

echo
echo "== Setup concluído =="
echo "Ative o venv (se ainda não estiver ativado):"
if [ -f .venv/bin/activate ]; then
  echo "  source .venv/bin/activate"
else
  echo "  source .venv/Scripts/activate"
fi

echo "Para iniciar treino usando GPU (se detectada):"
echo "  make start_train_gpu"

echo "Se quiser forçar cuda:0 diretamente:"
echo "  make start_train_cuda"

exit 0
