#!/bin/bash
# Full pipeline: fetch URLs -> download images -> prepare dataset -> train -> evaluate

set -e  # parar em caso de erro
cd "$(dirname "$0")/.."  # ir para raiz do projeto

echo "=========================================="
echo "🚀 PIPELINE COMPLETO DE TREINAMENTO"
echo "=========================================="

# Ativar venv
if [ -d ".venv" ]; then
    echo "✓ Ativando venv..."
    source .venv/bin/activate
else
    echo "⚠ Venv não encontrado. Criando..."
    python3 -m venv .venv
    source .venv/bin/activate
    python3 -m pip install -U pip
    python3 -m pip install -r config/requirements.txt
fi

# Instalar Pillow
echo "✓ Instalando Pillow..."
python3 -m pip install Pillow > /dev/null 2>&1

# Verificar se já tem imagens
IMG_COUNT=$(find datasets/images -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)

if [ "$IMG_COUNT" -gt 10 ]; then
    echo "✓ Encontradas $IMG_COUNT imagens. Pulando download..."
else
    # 1. Buscar URLs do Wikimedia
    echo ""
    echo "=========================================="
    echo "📥 PASSO 1: Gerando lista de URLs"
    echo "=========================================="
    if [ ! -f "data/urls/urls.txt" ]; then
        echo "✓ Fetch de URLs do Wikimedia..."
        python3 scripts/fetch_wikimedia_urls.py --out data/urls/urls.txt --count 300
    else
        echo "✓ urls.txt já existe, usando..."
    fi

    # 2. Baixar imagens
    echo ""
    echo "=========================================="
    echo "⬇️  PASSO 2: Baixando imagens"
    echo "=========================================="
    echo "✓ Baixando com delay e auto-retries..."
    python3 scripts/download_images.py data/urls/urls.txt datasets \
        --max 5000 \
        --split 0.8 \
        --delay 2.0 \
        --wikimedia-delay 3.0 \
        --contact matheuscf6@gmail.com \
        --per-class 50
fi

# 3. Preparar dataset
echo ""
echo "=========================================="
echo "🧹 PASSO 3: Preparando dataset"
echo "=========================================="
echo "✓ Limpando imagens corrompidas e gerando labels..."
python3 scripts/prepare_dataset.py

# 4. Treinar modelo
echo ""
echo "=========================================="
echo "🤖 PASSO 4: Treinando modelo"
echo "=========================================="

# Verificar se já existe modelo treinado
if [ -f "runs/detect/custom_run/weights/best.pt" ]; then
    echo "✓ Modelo treinado já existe!"
    echo ""
    echo "Opções:"
    echo "  1. Usar modelo existente (pular treino)"
    echo "  2. Treinar modelo novo (sobrescrever)"
    echo ""
    read -p "Escolha [1/2]: " choice
    
    if [ "$choice" == "2" ] || [ "$choice" == "" ]; then
        echo "✓ Iniciando novo treinamento (50 épocas)..."
        python3 scripts/train.py --max-epochs 50 --step 5 --target-map 0.60
    else
        echo "✓ Pulando treino, usando modelo existente"
    fi
else
    echo "✓ Iniciando treinamento (50 épocas)..."
    python3 scripts/train.py --max-epochs 50 --step 5 --target-map 0.60
fi

# 5. Avaliar modelo
echo ""
echo "=========================================="
echo "📊 PASSO 5: Avaliando modelo"
echo "=========================================="
echo "✓ Executando validação..."
python3 scripts/evaluate.py --weights runs/detect/custom_run/weights/best.pt --data config/data.yaml

echo ""
echo "=========================================="
echo "✅ PIPELINE COMPLETO!"
echo "=========================================="

echo ""
echo "📁 Modelo treinado: runs/detect/*/weights/best.pt"
echo "📈 Resultados: runs/detect/*/results.csv"

echo "✓ Não iniciaremos a API automaticamente (executado somente o treino)."
