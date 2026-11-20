#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${1:-$PWD}"
PKG_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$PROJECT_ROOT/08_v2_systematic_selection/source_forensic_r3_v1_0_0"
mkdir -p "$OUT"
python "$PKG_ROOT/tests/test_source_contract_forensic.py"
python "$PKG_ROOT/scripts/01_source_contract_forensic.py" \
  --project-root "$PROJECT_ROOT" \
  --out "$OUT" \
  2>&1 | tee "$OUT/STAGE08A_R3_SOURCE_CONTRACT.log"
echo
echo '=== R3 GATE ==='
cat "$OUT/STAGE08A_R3_SOURCE_CONTRACT_GATE.json"
echo
echo '=== FULL CORE MATRIX DIR ==='
cat "$OUT/FULL_CORE_MATRIX_DIR_FILES.tsv"
echo
echo '=== SURF AGGREGATE CANDIDATES (top 30) ==='
head -n 31 "$OUT/SURF_AGGREGATE_CANDIDATES.tsv"
echo
echo '=== SOURCE IO EVIDENCE (top 120) ==='
head -n 121 "$OUT/SOURCE_IO_EVIDENCE.tsv"
