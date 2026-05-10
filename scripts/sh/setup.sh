#!/usr/bin/env bash
set -euo pipefail

echo "== Setup script: instalar Python3, criar venv e instalar dependências =="

# detect python3
if command -v python3 >/dev/null 2>&1; then
  PY=python3
  echo "python3 encontrado: $(python3 --version)"
else
  echo "python3 não encontrado — tentando instalar via gerenciador de pacotes (requer sudo)"
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y python3 python3-venv python3-pip
  elif command -v yum >/dev/null 2>&1; then
    sudo yum install -y python3 python3-venv python3-pip
  elif command -v apk >/dev/null 2>&1; then
    sudo apk add --no-cache python3 py3-pip
  else
    echo "Nenhum gerenciador de pacotes suportado detectado. Instale Python3 manualmente." >&2
    exit 1
  fi
  PY=python3
fi

# criar virtualenv
if [ -d "venv" ]; then
  echo "Virtualenv já existe em ./venv"
else
  echo "Criando virtualenv em ./venv..."
  $PY -m venv venv
fi

echo "Ativando venv e instalando dependências..."
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
else
  echo "Aviso: requirements.txt não encontrado, pulando instalação de dependências"
fi

echo "Setup concluído. Para ativar o ambiente: source venv/bin/activate"
echo "Iniciar servidor: ./start.sh"
