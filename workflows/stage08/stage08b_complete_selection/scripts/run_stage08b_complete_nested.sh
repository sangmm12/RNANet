#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$PWD}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/08_v2_systematic_selection/complete_nested_v1_0_1"
mkdir -p "$OUT"
LOG="$OUT/stage08b_complete_nested.log"
export XGB_DEVICE="${XGB_DEVICE:-cuda}"
export THREADS="${THREADS:-32}"
"$HERE/scripts/00_selftest.sh"
echo "[INFO] Stage08B v1.0.1 running. Full verbose output -> $LOG"
echo "[INFO] Screen output is intentionally suppressed; only checkpoints/final summary are shown."
set +e
python "$HERE/src/stage08b_complete_nested.py" --root "$ROOT" >"$LOG" 2>&1
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  echo "[FAIL] Stage08B v1.0.1 rc=$rc; last 60 log lines:"
  tail -n 60 "$LOG"
  exit $rc
fi
cat "$OUT/STAGE08B_COMPLETE_NESTED_GATE.json"
echo
echo "===== CONSENSUS COMPLETE ARM ====="
cat "$OUT/CONSENSUS_COMPLETE_ARM.json"
echo
echo "[DONE] Full log: $LOG"
