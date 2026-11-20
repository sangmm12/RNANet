#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${1:-$HOME/BioData/hgq/paper_RNANet}"
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$PROJECT_ROOT/07_test17_external_benchmark/v1_0_0"
mkdir -p "$OUT/logs"
python "$PKG_ROOT/scripts/02_extract_rnanet_cyc.py" \
  --project-root "$PROJECT_ROOT" \
  --package-root "$PKG_ROOT" \
  2>&1 | tee "$OUT/logs/stage07b_extract_cyc.log"
python "$PKG_ROOT/scripts/03_train_rnanet_test17.py" \
  --project-root "$PROJECT_ROOT" \
  --package-root "$PKG_ROOT" \
  2>&1 | tee "$OUT/logs/stage07b_train_rnanet_test17.log"
echo
cat "$OUT/STAGE07B_RNANET_TEST17_GATE.json"
