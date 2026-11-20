#!/usr/bin/env bash
set -uo pipefail
ROOT="${1:-/mnt/202001Main/BioData/hgq}"
P="RNANET_NC_FROZEN_CORE_ROTATION_R102_20261112"
HERE="$ROOT/$P"
OUT="$ROOT/paper_RNANet/exports/NC_FROZEN_CORE_ROTATION_R102_20261112"
mkdir -p "$OUT"
printf '[INFO] root=%s\n[INFO] out=%s\n' "$ROOT" "$OUT"
python "$HERE/src/run_core_rotation.py" \
  --root "$ROOT" --seed "${SEED:-3407}" --n-chains "${N_CHAINS:-12}" \
  --n-rot "${N_ROT:-3}" --max-len "${MAX_LEN:-500}" --out "$OUT" \
  > "$OUT/run.log" 2>&1
CODE=$?
cat "$OUT/GATE.json" 2>/dev/null || true
printf '[INFO] run exit code=%s\n' "$CODE"
STAMP="20261112_$(date -u +%H%M%S)"
ARCHIVE="$ROOT/paper_RNANet/exports/RNANET_NC_FROZEN_CORE_ROTATION_R102_${STAMP}.tar.gz"
tar -czf "$ARCHIVE" -C "$ROOT/paper_RNANet/exports" "NC_FROZEN_CORE_ROTATION_R102_20261112"
if [ $? -eq 0 ]; then
  sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
  printf '[RESULT] %s\n[SHA256] %s.sha256\n' "$ARCHIVE" "$ARCHIVE"
fi
exit 0
