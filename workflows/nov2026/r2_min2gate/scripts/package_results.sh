#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$HOME/BioData/hgq}"
P="$ROOT/paper_RNANet"
TAG="${RNANET_CHALLENGE_TAG:-20261112_NC_MIN2GATE_R2_LENGTH20}"
OUT="$P/exports/$TAG"
test -f "$OUT/FINAL_GATE.json" || { echo '[FATAL] final gate missing'; exit 2; }
mkdir -p "$P/exports"
ARCH="$P/exports/RNANET_NC_MIN2GATE_REVIEW_${TAG}.tar.gz"
# Include results, CSVs and short reports; do not copy full exported CIF/PDB structural coordinates.
tar -czf "$ARCH" -C "$OUT" \
    FINAL_GATE.json inputs.sha256 \
    rotation/rotation_gate.json rotation/baseline_anchor.csv \
    rotation/rotation_feature_detail.csv rotation/rotation_feature_summary.csv \
    rotation/rotation_query_rank_overlap.csv \
    ablation/retrieval_contract.json ablation/candidates_C_ONLY.csv \
    ablation/candidates_FM_ONLY.csv ablation/candidates_C_FM.csv \
    ablation/union_candidates.csv \
    structural/chain_export_manifest.csv structural/union_usalign.csv \
    summary/comparison_gate.json summary/length_gate.json summary/length_stratified_comparison.csv summary/TopK_comparison.csv \
    summary/all_candidate_structural_detail.csv logs/run_two_gates.log
sha256sum "$ARCH" > "$ARCH.sha256"
ls -lh "$ARCH" "$ARCH.sha256"
