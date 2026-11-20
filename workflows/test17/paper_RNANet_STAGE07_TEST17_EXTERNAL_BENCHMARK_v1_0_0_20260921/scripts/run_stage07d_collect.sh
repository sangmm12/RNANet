#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${1:-$HOME/BioData/hgq/paper_RNANet}"
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$PROJECT_ROOT/07_test17_external_benchmark/v1_0_0"
mkdir -p "$OUT/logs"
python "$PKG_ROOT/scripts/05_collect_external_benchmark.py" \
  --project-root "$PROJECT_ROOT" \
  2>&1 | tee "$OUT/logs/stage07d_collect.log"

SMALL="$OUT/_small_results_stage07"
rm -rf "$SMALL"
mkdir -p "$SMALL/comparator_results" "$SMALL/rnanet_results" "$SMALL/logs"
cp -f "$OUT"/STAGE07*.json "$SMALL/" 2>/dev/null || true
cp -f "$OUT"/BENCHMARK_COMMIT.lock "$OUT"/BENCHMARK_CRITICAL_SHA256.json "$OUT"/DATASET_MANIFEST.csv "$SMALL/"
cp -f "$OUT"/TEST17_11_METHOD_COMPARISON.csv "$OUT"/TEST17_11_METHOD_MANUSCRIPT_TABLE.csv "$OUT"/MANUSCRIPT_READY_EXTERNAL_BENCHMARK.md "$SMALL/"
cp -f "$OUT"/CY-C_FEATURES_42.txt "$OUT"/CY-C_MODEL_FEATURES_46.txt "$OUT"/CY-C_CHAIN_EXTRACTION_AUDIT.csv "$SMALL/"
cp -f "$OUT"/results/RNANet_CY-C/test17_results.json "$SMALL/rnanet_results/RNANet_CY-C_test17_results.json"
cp -f "$OUT"/results/RNANet_CY-C+HNM/test17_results.json "$SMALL/rnanet_results/RNANet_CY-C+HNM_test17_results.json"
BS="$OUT/vendor/rna-small-molecule-review/BindingSite"
for m in RLBind RNet RBind Rsite Rsite2 RNAsite RNABind MultiModRLBP MVRBind; do
  cp -f "$BS/results/$m/test17_results.json" "$SMALL/comparator_results/${m}_test17_results.json"
done
cp -f "$OUT"/logs/*.log "$SMALL/logs/" 2>/dev/null || true

PARENT="$(dirname "$PROJECT_ROOT")"
ARCHIVE="$PARENT/paper_RNANet_STAGE07_TEST17_EXTERNAL_BENCHMARK_RESULTS_v1_0_0_20260921.tar.gz"
tar -czf "$ARCHIVE" -C "$OUT" "$(basename "$SMALL")"
sha256sum "$ARCHIVE" | tee "$ARCHIVE.sha256"
rm -rf "$SMALL"

echo "[DONE] $ARCHIVE"
echo "[DONE] $ARCHIVE.sha256"
cat "$OUT/STAGE07D_FINAL_EXTERNAL_BENCHMARK_GATE.json"
