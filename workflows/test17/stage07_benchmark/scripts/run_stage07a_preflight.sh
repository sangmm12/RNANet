#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${1:-$HOME/BioData/hgq/paper_RNANet}"
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$PROJECT_ROOT/07_test17_external_benchmark/v1_0_0/logs"
python "$PKG_ROOT/scripts/01_stage07a_preflight.py" \
  --project-root "$PROJECT_ROOT" \
  --package-root "$PKG_ROOT" \
  2>&1 | tee "$PROJECT_ROOT/07_test17_external_benchmark/v1_0_0/logs/stage07a_preflight.log"
echo
cat "$PROJECT_ROOT/07_test17_external_benchmark/v1_0_0/STAGE07A_TEST17_PREFLIGHT_GATE.json"
