#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$HOME/BioData/hgq}"
ROOT="$(cd "$ROOT" && pwd)"
P="$ROOT/paper_RNANet"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="${RNANET_CHALLENGE_TAG:-20261112_NC_MIN2GATE_R2_LENGTH20}"
OUT="$P/exports/$STAMP"
C="$P/04_surf_ablation/masters/master_C.csv"
SEQ="$P/02_retrieval/geometry/strict_rna_sequences.csv"
CIF="$ROOT/RNANet/rna_cif_data"
USALIGN="${USALIGN:-/home/hgq/conda-envs/usalign_env/bin/USalign}"
FROZEN="$P/04_surf_ablation/B1_dimension_neutral/candidates/candidates_C.csv"
for f in "$C" "$SEQ" "$USALIGN"; do
    if test ! -f "$f"; then echo "[FATAL] file absent: $f" >&2; exit 2; fi
done
if test ! -d "$CIF"; then echo "[FATAL] CIF directory absent: $CIF" >&2; exit 2; fi
mkdir -p "$OUT"/{rotation,ablation,structural,summary,logs}
exec > >(tee -a "$OUT/logs/run_two_gates.log") 2>&1
printf '[ROOT] %s\n[OUT] %s\n[PYTHON] %s\n[USALIGN] %s\n' "$ROOT" "$OUT" "$(command -v python)" "$USALIGN"
python - <<'PY'
import numpy, pandas, scipy, sklearn, Bio
print('[ENV PASS] numpy',numpy.__version__,'pandas',pandas.__version__,'scipy',scipy.__version__,'sklearn',sklearn.__version__,'biopython',Bio.__version__)
PY
sha256sum "$C" "$SEQ" "$USALIGN" > "$OUT/inputs.sha256"
sha256sum "$HERE"/src/*.py >> "$OUT/inputs.sha256"
echo '[1/4] Coordinate-rotation probe using frozen CY-C source'
python "$HERE/src/rotation_probe.py" --master "$C" --seq "$SEQ" --cif-dir "$CIF" \
    --out "$OUT/rotation" --n-chains "${ROTATION_CHAINS:-12}" \
    --n-rot "${ROTATION_TRIALS:-3}" --max-length "${ROTATION_MAX_LENGTH:-600}" --seed 20261112
echo '[2/4] Fair three-arm modality ablation (new standalone challenge)'
python "$HERE/src/ablation_retrieval.py" --master "$C" --seq "$SEQ" \
    --out "$OUT/ablation" --kNN 100 --percent 20 --topN 100 --min-length 20
if test -f "$FROZEN"; then
    cp "$FROZEN" "$OUT/summary/frozen_C_reference_DO_NOT_COMPARE_AS_IDENTICAL_PROTOCOL.csv"
fi
echo '[3/4] Export exact CIF chains and resume structural alignment'
python "$HERE/src/frozen_extract_candidate_chains.py" \
    --candidates "$OUT/ablation/union_candidates.csv" --cif-dir "$CIF" \
    --outdir "$OUT/structural"
python "$HERE/src/align_resume.py" \
    --candidates "$OUT/ablation/union_candidates.csv" \
    --chain-manifest "$OUT/structural/chain_export_manifest.csv" \
    --usalign "$USALIGN" --out "$OUT/structural/union_usalign.csv" \
    --raw-dir "$OUT/structural/usalign_raw"
echo '[4/4] Frozen-threshold descriptive TopK comparison'
ARGS=(--candidate-dir "$OUT/ablation" --align "$OUT/structural/union_usalign.csv" --out "$OUT/summary")
if test -f "$FROZEN"; then ARGS+=(--frozen-c "$FROZEN"); fi
python "$HERE/src/summarize_challenge.py" "${ARGS[@]}"
echo "[COMPLETE] results remain unchanged under $P/04_surf_ablation and $P/02_retrieval"
echo "[COMPLETE] independent challenge outputs: $OUT"
