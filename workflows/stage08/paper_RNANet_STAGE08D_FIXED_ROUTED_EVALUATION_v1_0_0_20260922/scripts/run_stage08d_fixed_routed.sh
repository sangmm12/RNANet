#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$PWD}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/08_v2_systematic_selection/routed_fixed_v1_0_0"
mkdir -p "$OUT"
LOG="$OUT/stage08d_fixed_routed.log"
export XGB_DEVICE="${XGB_DEVICE:-cuda}"
export THREADS="${THREADS:-32}"
"$HERE/scripts/00_selftest.sh"
echo "[INFO] Stage08D is selection-free: fixed Core=C+HNM, Complete=ACS+HNM."
echo "[INFO] Route: frozen Stage05A SURF-AD -> Complete; otherwise -> Core."
echo "[INFO] One shared outer-validation PDB partition is used: early-stop vs calibration."
echo "[INFO] Primary pooled route uses calibrated scores; raw pooled route is diagnostic only."
echo "[INFO] Calibration-matched Core control isolates Complete-representation gain from calibration gain."
echo "[INFO] Completed outer seeds are resumable from exact source-bound checkpoints."
echo "[INFO] Full verbose output -> $LOG"
set +e
python "$HERE/src/stage08d_fixed_routed.py" --root "$ROOT" >"$LOG" 2>&1
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  echo "[FAIL] Stage08D rc=$rc; last 120 log lines:"
  tail -n 120 "$LOG"
  exit $rc
fi
cat "$OUT/STAGE08D_FIXED_ROUTED_GATE.json"
echo
echo "===== FROZEN ROUTED SYSTEM ====="
cat "$OUT/FROZEN_ROUTED_SYSTEM.json"
echo
echo "[DONE] Full log: $LOG"
