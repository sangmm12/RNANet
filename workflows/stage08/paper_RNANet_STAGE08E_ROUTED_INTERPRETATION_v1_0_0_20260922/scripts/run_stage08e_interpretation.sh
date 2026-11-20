#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$PWD}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "$SCRIPT_DIR/src/stage08e_interpretation.py" "$ROOT"
