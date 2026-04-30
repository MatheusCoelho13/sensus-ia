#!/bin/bash
# Pipeline Completo: Treina → Usa Melhor Modelo → Detecta com Hitbox

set -e
cd "$(dirname "$0")/.."

echo "=========================================="
echo "🚀 PIPELINE COMPLETO - TREINAR E DETECTAR"
echo "=========================================="
echo ""

# Ativar venv
if [ ! -d ".venv" ]; then
    echo "Criando venv..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# Instalar dependências
echo "✓ Instalando dependências..."
pip install -q -r config/requirements.txt 2>/dev/null || true

# PASSO 1: Treinar modelo
echo ""
echo "=========================================="
echo "🤖 PASSO 1: TREINANDO MODELO"
echo "=========================================="
echo ""

if [ -f "runs/detect/custom_run/weights/best.pt" ]; then
    echo "✓ Modelo já existe!"
    echo ""
    read -p "Deseja retreinar? (s/n) [padrão: n]: " treinar
    if [ "$treinar" != "s" ] && [ "$treinar" != "S" ]; then
        echo "✓ Usando modelo existente..."
    else
        echo "Iniciando novo treinamento..."
        python3 scripts/train.py
    fi
else
    echo "Iniciando treinamento..."
    python3 scripts/train.py
fi

# PASSO 2: Identificar melhor modelo
echo ""
echo "=========================================="
echo "🏆 PASSO 2: IDENTIFICANDO MELHOR MODELO"
echo "=========================================="
echo ""

BEST_MODEL="runs/detect/custom_run/weights/best.pt"
if [ -f "$BEST_MODEL" ]; then
    SIZE=$(ls -lh "$BEST_MODEL" | awk '{print $5}')
    echo "✓ Melhor modelo encontrado: $BEST_MODEL ($SIZE)"
    echo ""
    echo "Atualizando app.py para usar este modelo..."
    
    # Atualizar app.py
    sed -i "s|model = YOLO(.*)|model = YOLO('$BEST_MODEL')|g" app.py
    echo "✓ app.py atualizado"
else
    echo "❌ Modelo best.pt não encontrado!"
    exit 1
fi

# PASSO 3: Testar detecção
echo ""
echo "=========================================="
echo "👁️  PASSO 3: TESTANDO DETECÇÃO COM HITBOX"
echo "=========================================="
echo ""

echo "Opções:"
echo "  1. Testar com webcam (tempo real + hitbox)"
echo "  2. Testar com arquivo de imagem"
echo "  3. Testar com URL"
echo "  4. Ir para API"
echo ""
read -p "Escolha [1-4]: " choice

case $choice in
    1)
        echo "Abrindo câmera..."
        echo "Pressione 'q' para sair"
        echo ""
        python3 scripts/detect_and_visualize.py webcam --model "$BEST_MODEL" --output output
        ;;
    2)
        echo ""
        read -p "Digite o caminho da imagem: " IMG_PATH
        if [ -f "$IMG_PATH" ]; then
            python3 scripts/detect_and_visualize.py "$IMG_PATH" --model "$BEST_MODEL" --output output
            echo ""
            echo "✓ Imagem salva em output/"
        else
            echo "❌ Arquivo não encontrado: $IMG_PATH"
        fi
        ;;
    3)
        echo ""
        read -p "Digite a URL da imagem: " IMG_URL
        python3 scripts/detect_and_visualize.py "$IMG_URL" --model "$BEST_MODEL" --output output
        echo ""
        echo "✓ Imagem salva em output/"
        ;;
    4)
        echo ""
        echo "Iniciando API..."
        echo "Acesse: http://localhost:8000/docs"
        uvicorn app:app --host 0.0.0.0 --port 8000 --reload
        ;;
    *)
        echo "Opção inválida"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "✅ PIPELINE COMPLETO!"
echo "=========================================="
