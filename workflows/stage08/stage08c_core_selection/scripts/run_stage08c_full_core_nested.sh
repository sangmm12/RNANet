#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$PWD}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/08_v2_systematic_selection/full_core_nested_v1_0_0"
mkdir -p "$OUT"
LOG="$OUT/stage08c_full_core_nested.log"
export XGB_DEVICE="${XGB_DEVICE:-cuda}"
export THREADS="${THREADS:-32}"
"$HERE/scripts/00_selftest.sh"
echo "[INFO] Stage08C running on the exact 6,252,187-row full Core universe."
echo "[INFO] Candidates: A/C/AC x standard/HNM; SURF and Test17 are excluded from selection."
echo "[INFO] Full verbose output -> $LOG"
echo "[INFO] The run is resumable at completed inner-fold and outer-seed checkpoints."
set +e
python "$HERE/src/stage08c_full_core_nested.py" --root "$ROOT" >"$LOG" 2>&1
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  echo "[FAIL] Stage08C rc=$rc; last 80 log lines:"
  tail -n 80 "$LOG"
  exit $rc
fi
cat "$OUT/STAGE08C_FULL_CORE_NESTED_GATE.json"
echo
echo "===== CONSENSUS CORE ARM ====="
cat "$OUT/CONSENSUS_CORE_ARM.json"
echo
echo "[DONE] Full log: $LOG"
