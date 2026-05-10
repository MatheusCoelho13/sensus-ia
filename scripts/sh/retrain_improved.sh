#!/usr/bin/env bash
# Retreinamento melhorado: yolov8s, 200 epochs, 768px
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IA_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$IA_DIR"

echo "=== Sensus IA — Retreinamento Melhorado ==="
echo "Diretório: $IA_DIR"
echo ""

# Verificar GPU
if command -v nvidia-smi &>/dev/null; then
    echo "GPU detectada:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    DEVICE="0"
else
    echo "GPU não encontrada. Usando CPU (pode demorar 2-4h)."
    DEVICE="cpu"
fi
echo ""

# Backup do best.pt atual
if [ -f "models/best.pt" ]; then
    BACKUP="models/best_backup_$(date +%Y%m%d_%H%M%S).pt"
    cp "models/best.pt" "$BACKUP"
    echo "Backup do modelo atual salvo em: $BACKUP"
fi

echo ""
echo "Iniciando treino: yolov8s | 200 epochs | 768px | batch 16"
echo "--------------------------------------------------------------"

python3 scripts/train/train.py \
    --model yolov8s.pt \
    --data config/data.yaml \
    --max-epochs 200 \
    --imgsz 768 \
    --batch 16 \
    --device "$DEVICE"

echo ""
echo "=== Treino finalizado ==="
echo "Modelo salvo em: models/best.pt"
echo "Resultados em:   runs/detect/train_iter/"
