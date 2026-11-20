#!/usr/bin/env bash
set -euo pipefail
PROJECT="${1:-$PWD}"
TOOL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT"
bash "$TOOL_ROOT/scripts/00_selftest.sh"
python "$TOOL_ROOT/scripts/01_train_cyc_surf_hnm_test17.py" --project-root "$PROJECT"
