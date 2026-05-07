#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./generate_docs.sh [OUTDIR]
# Env:
#   DOCFORMAT (optional) e.g. "google" or "numpy" (default: google)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
LIB_DIR="${PROJECT_ROOT}/utils/lib"
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

# Collect python files (exclude backups)
mapfile -t FILES < <(find "$LIB_DIR" -maxdepth 1 -type f -name '*.py' ! -name '*bak*' -print)

if [ "${#FILES[@]}" -eq 0 ]; then
  echo "No .py files found in $LIB_DIR" >&2
  exit 1
fi

echo "Generating docs for ${#FILES[@]} modules -> $OUTDIR (docformat=$DOCFORMAT)"
pdoc --docformat "$DOCFORMAT" --math --mermaid -o "$OUTDIR" "${FILES[@]}"

echo "Docs generated in: $OUTDIR"