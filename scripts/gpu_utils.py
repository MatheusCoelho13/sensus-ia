"""Helpers para detecção e exigência de GPU/CUDA.

Fornece utilitários usados por `train.py` e scripts auxiliares.
"""
from shutil import which
import subprocess
import sys


def nvidia_smi_available():
    return which('nvidia-smi') is not None


def detect_cuda_version():
    """Tenta detectar versão CUDA via `nvidia-smi`.
    Retorna string como '13.1' ou empty string se não encontrada.
    """
    if not nvidia_smi_available():
        return ''
    try:
        out = subprocess.check_output(['nvidia-smi'], stderr=subprocess.DEVNULL, text=True)
        for line in out.splitlines():
            if 'CUDA Version' in line:
                parts = line.split('CUDA Version:')
                if len(parts) > 1:
                    v = parts[1].strip().split()[0]
                    return v
    except Exception:
        pass
    return ''


def torch_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def require_cuda_or_exit(msg=None):
    """Verifica se CUDA está disponível (torch or nvidia-smi). Sai do processo se não estiver."""
    ok = torch_cuda_available() or nvidia_smi_available()
    if not ok:
        if msg:
            print(msg)
        else:
            print('CUDA / NVIDIA GPU não detectada. Abortando (require-cuda).')
        sys.exit(1)


def choose_device(prefer_gpu=True):
    """Retorna string device para passar ao ultralytics/torch: 'cuda:0' ou 'cpu'."""
    if prefer_gpu and torch_cuda_available():
        return 'cuda:0'
    # if torch not available but nvidia-smi present, still try cuda
    if prefer_gpu and nvidia_smi_available():
        return 'cuda:0'
    return 'cpu'
