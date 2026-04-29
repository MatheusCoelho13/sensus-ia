#!/bin/bash
# Script para listar e visualizar modelos disponíveis

echo "=========================================="
echo "🤖 MODELOS DISPONÍVEIS"
echo "=========================================="
echo ""

# Modelos pré-treinados
echo "📦 Modelos Pré-Treinados:"
if [ -d "models" ]; then
    ls -lh models/*.pt 2>/dev/null | awk '{print "  - " $9 " (" $5 ")"}' || echo "  (nenhum)"
else
    echo "  (pasta models não existe)"
fi

echo ""

# Modelos treinados
echo "🏆 Modelos Treinados:"
if [ -d "runs/detect" ]; then
    for run_dir in runs/detect/*/; do
        if [ -d "$run_dir/weights" ]; then
            run_name=$(basename "$run_dir")
            echo "  📍 $run_name"
            
            if [ -f "$run_dir/weights/best.pt" ]; then
                size=$(ls -lh "$run_dir/weights/best.pt" | awk '{print $5}')
                echo "     - best.pt ($size)"
            fi
            
            if [ -f "$run_dir/weights/last.pt" ]; then
                size=$(ls -lh "$run_dir/weights/last.pt" | awk '{print $5}')
                echo "     - last.pt ($size)"
            fi
            
            if [ -f "$run_dir/results.csv" ]; then
                epochs=$(tail -1 "$run_dir/results.csv" | awk -F',' '{print $1}')
                map50=$(tail -1 "$run_dir/results.csv" | awk -F',' '{print $7}')
                echo "     - Épocas: $epochs | mAP50: $map50"
            fi
        fi
    done
else
    echo "  (nenhum modelo treinado ainda)"
fi

echo ""
echo "=========================================="
