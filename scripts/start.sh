#!/usr/bin/env bash
set -euo pipefail

if [ -f venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
else
  echo "Virtualenv não encontrado. Execute ./setup.sh primeiro." >&2
  exit 1
fi

echo "Iniciando API (uvicorn) na porta 8000..."
exec uvicorn app:app --host 0.0.0.0 --port 8000
