#!/bin/bash
# Visualizar modelos treinados e métricas

echo "=========================================="
echo "🤖 MODELOS TREINADOS"
echo "=========================================="

if [ ! -d "runs/detect" ]; then
    echo "❌ Nenhum modelo treinado encontrado (runs/detect não existe)"
    exit 1
fi

# Listar runs
echo ""
echo "📂 Runs disponíveis:"
ls -d runs/detect/*/ 2>/dev/null | while read run; do
    name=$(basename "$run")
    echo "  • $name"
done

# Mostrar modelos
echo ""
echo "📊 Modelos (.pt):"
find runs/detect -name "*.pt" -type f | while read model; do
    size=$(du -h "$model" | cut -f1)
    echo "  • $model ($size)"
done

# Mostrar resultados se existirem
echo ""
echo "📈 Resultados (CSV):"
find runs/detect -name "results.csv" -type f | while read csv; do
    dir=$(dirname "$csv")
    name=$(basename "$dir")
    if [ -f "$csv" ]; then
        echo ""
        echo "  Run: $name"
        echo "  Epochs: $(wc -l < "$csv") (excl. header)"
        
        # Últimas métricas
        tail -1 "$csv" | awk -F',' '{
            print "    - Train Loss: " $3
            print "    - Val Loss: " $10
            print "    - mAP50: " $8
            print "    - mAP50-95: " $9
        }'
    fi
done

echo ""
echo "=========================================="
echo ""
echo "🚀 Para usar o melhor modelo:"
echo "   model = YOLO('runs/detect/custom_run/weights/best.pt')"
echo ""
