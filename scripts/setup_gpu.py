#!/usr/bin/env python3
"""Wrapper Python para setup de GPU.

Este script tenta delegar para `scripts/setup_gpu.sh` se presente,
caso contrário imprime instruções básicas.
"""
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).parent.parent
    sh = root / 'scripts' / 'setup_gpu.sh'
    if sh.exists():
        print('Executando scripts/setup_gpu.sh ...')
        rc = subprocess.call(['bash', str(sh)])
        sys.exit(rc)
    else:
        print('scripts/setup_gpu.sh não encontrado.')
        print('Tente rodar manualmente ou usar o helper `ia/scripts/gpu_utils.py` para checar a GPU.')


if __name__ == '__main__':
    main()
