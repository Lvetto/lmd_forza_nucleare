#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./generate_docs.sh [OUTDIR]
# Env:
#   DOCFORMAT (optional) e.g. "google" or "numpy" (default: google)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
LIB_DIR="${PROJECT_ROOT}/utils/lib"
UTILS_DIR="${PROJECT_ROOT}/utils"
OUTDIR="${1:-${PROJECT_ROOT}/docs/api}"
DOCFORMAT="${DOCFORMAT:-google}"

if [ ! -d "$LIB_DIR" ]; then
  echo "Directory not found: $LIB_DIR" >&2
  exit 1
fi

# Ensure pdoc is available (try to install to user site if missing)
if ! command -v pdoc >/dev/null 2>&1; then
  echo "pdoc not found — installing with pip (user site)..." >&2
  python -m pip install --user pdoc
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v pdoc >/dev/null 2>&1; then
    echo "pdoc still not found. Add $HOME/.local/bin to PATH or install pdoc manually." >&2
    exit 1
  fi
fi

mkdir -p "$OUTDIR"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# === MODIFICA INIZIO ===
# Crea o aggiorna __init__.py inserendo il contenuto del README come docstring
README_FILE="${UTILS_DIR}/README.md"
INIT_FILE="${LIB_DIR}/__init__.py"

if [ -f "$README_FILE" ]; then
  echo "Trovato README.md, iniezione in $INIT_FILE come main docstring..."
  echo '"""' > "$INIT_FILE"
  cat "$README_FILE" >> "$INIT_FILE"
  echo '"""' >> "$INIT_FILE"
else
  echo "Attenzione: README.md non trovato in $UTILS_DIR"
  touch "$INIT_FILE" # Crea comunque il file vuoto per renderlo un pacchetto Python
fi

echo "Generating docs for package $LIB_DIR -> $OUTDIR (docformat=$DOCFORMAT)"

# Invece di passare la lista di file, passiamo l'intera directory a pdoc.
# In questo modo pdoc riconoscerà __init__.py e userà il README come index.html
pdoc --docformat "$DOCFORMAT" --math --mermaid -o "$OUTDIR" "$LIB_DIR"
# === MODIFICA FINE ===

echo "Docs generated in: $OUTDIR"