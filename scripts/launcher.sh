#!/bin/bash
# Smart launcher: tenta Docker, fallback para Python local

set -e
cd "$(dirname "$0")/.."

echo "=========================================="
echo "🚀 ASSISTIVA IA - AUTO LAUNCHER"
echo "=========================================="
echo ""

# Verificar se docker está disponível
check_docker() {
    if command -v docker &> /dev/null && docker ps &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Verificar se docker-compose está disponível
check_docker_compose() {
    if command -v docker-compose &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Opção 1: Docker Compose (se disponível)
if check_docker_compose; then
    echo "✓ Docker Compose detectado"
    echo ""
    echo "Opções:"
    echo "  1. Rodar com Docker Compose (isolado)"
    echo "  2. Rodar com Python local (rápido)"
    echo ""
    read -p "Escolha [1/2] (padrão: 1): " choice
    choice=${choice:-1}
    
    if [ "$choice" == "1" ]; then
        echo ""
        echo "✓ Iniciando com Docker Compose..."
        docker-compose up --build
        exit 0
    fi
fi

# Opção 2: Docker direto (build + run)
if check_docker; then
    echo "✓ Docker detectado"
    echo ""
    echo "Opções:"
    echo "  1. Rodar com Docker (build + run)"
    echo "  2. Rodar com Python local (rápido)"
    echo ""
    read -p "Escolha [1/2] (padrão: 1): " choice
    choice=${choice:-1}
    
    if [ "$choice" == "1" ]; then
        echo ""
        echo "✓ Building Docker image..."
        docker build -t assistiva-ia .
        
        if [ $? -eq 0 ]; then
            echo "✓ Iniciando container..."
            docker run --rm -p 8000:8000 \
                -v $(pwd)/datasets:/app/datasets \
                -v $(pwd)/runs:/app/runs \
                assistiva-ia
            exit 0
        else
            echo "⚠ Build falhou, usando Python local..."
        fi
    fi
fi

# Fallback: Python local
echo ""
echo "⚠ Docker não disponível ou desabilitado"
echo "✓ Usando Python local..."
echo ""

bash scripts/full_pipeline.sh
