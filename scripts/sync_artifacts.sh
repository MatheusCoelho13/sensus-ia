#!/usr/bin/env bash
set -euo pipefail

# scripts/sync_artifacts.sh
# Sincroniza `runs/` e `models/` para um destino usando rsync.
# Uso:
#   ./scripts/sync_artifacts.sh /path/to/dest   # copia local->dest
#   SYNC_SRC=/some/src ./scripts/sync_artifacts.sh /path/to/dest

DEST=${1:-}
SRC=${SYNC_SRC:-$(pwd)}

if [ -z "$DEST" ]; then
  echo "Usage: $0 /path/to/destination"
  exit 2
fi

mkdir -p "$DEST"

echo "🔁 Sincronizando artifacts from $SRC to $DEST"

rsync -av --delete --exclude='*.tmp' --exclude='__pycache__/' "$SRC/models/" "$DEST/models/"
rsync -av --delete --exclude='*.tmp' --exclude='__pycache__/' "$SRC/runs/" "$DEST/runs/"

echo "✅ Sincronização concluída"

exit 0
