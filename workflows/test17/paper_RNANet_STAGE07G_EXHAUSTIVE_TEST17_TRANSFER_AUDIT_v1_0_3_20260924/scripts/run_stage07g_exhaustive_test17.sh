#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then echo "usage: $0 PROJECT_ROOT" >&2; exit 2; fi
ROOT="$(cd "$1" && pwd)"
PKG="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$PKG/src${PYTHONPATH:+:$PYTHONPATH}"
BASE="$ROOT/07_test17_external_benchmark/v1_0_0"
OLD="$BASE/exhaustive_14arm_transfer_v1_0_2"
OUT="$BASE/exhaustive_14arm_transfer_v1_0_3"

echo "[STAGE07G v1.0.3] exhaustive Test17 representation-family transfer audit"
echo "[BOUNDARY] canonical 14-arm family uses Stage08 feature order; historical Stage07E/F CS order is audited separately."
bash "$PKG/scripts/00_selftest.sh"

# Reuse the expensive completed v1.0.2 fit set if available. The v1.0.2 failure occurred only after all 70 fits were present, during an over-strict loaded-model metric replay check.
if [[ ! -e "$OUT/STAGE07G_CYA_MATRIX_GATE.json" && -e "$OLD/STAGE07G_CYA_MATRIX_GATE.json" ]]; then
  echo "[MIGRATE] copying completed v1.0.2 matrix/checkpoints/models into v1.0.3"
  rm -rf "$OUT"
  cp -a "$OLD" "$OUT"
fi

if [[ ! -e "$OUT/STAGE07G_CYA_MATRIX_GATE.json" ]]; then
  "$PYTHON" "$PKG/scripts/01_build_cya_acs_matrix.py" --project-root "$ROOT"
else
  echo "[RESUME] Stage07G CY-A/ACS matrix already present"
fi

"$PYTHON" "$PKG/scripts/02_run_14arm_transfer.py" --project-root "$ROOT"

echo
cat "$OUT/STAGE07G_EXHAUSTIVE_TEST17_GATE.json"
echo
echo "===== FEATURE-ORDER PROVENANCE AUDIT ====="
cat "$OUT/FEATURE_ORDER_PROVENANCE_AUDIT.json"
echo
echo "===== 14-ARM COMPLETE TABLE ====="
column -t -s, "$OUT/TEST17_14ARM_COMPLETE_TABLE.csv" || cat "$OUT/TEST17_14ARM_COMPLETE_TABLE.csv"
echo
echo "===== CY-A INCREMENT CONTRASTS ====="
column -t -s, "$OUT/TEST17_CYA_INCREMENT_CONTRASTS.csv" || cat "$OUT/TEST17_CYA_INCREMENT_CONTRASTS.csv"
