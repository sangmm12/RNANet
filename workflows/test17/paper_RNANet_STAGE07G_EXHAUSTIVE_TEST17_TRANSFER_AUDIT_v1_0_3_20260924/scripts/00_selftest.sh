#!/usr/bin/env bash
set -euo pipefail
PKG="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$PKG/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" - <<'PY'
from stage07g_common import *
import numpy as np
assert REPS==['A','C','S','AC','AS','CS','ACS']
assert ARMS==['A','A+HNM','C','C+HNM','S','S+HNM','AC','AC+HNM','AS','AS+HNM','CS','CS+HNM','ACS','ACS+HNM']
assert DIMS=={'A':50,'C':46,'S':61,'AC':90,'AS':107,'CS':103,'ACS':147}
y=np.array([0,0,1,0,0,0,1,0]); p=np.array([.1,.9,.8,.7,.2,.4,.6,.3]); c=np.array(['a']*4+['b']*4)
m=hard_negative_mask(y,p,c,2)
assert np.where(m)[0].tolist()==[1,3,5,7]
assert not np.any(y[m]!=0)
print('[PASS] Stage07G feature-family and HNM selftest')
PY
python -m py_compile "$PKG/scripts/01_build_cya_acs_matrix.py" "$PKG/scripts/02_run_14arm_transfer.py" "$PKG/src/stage07g_common.py"
echo "[PASS] Stage07G Python compile selftest"
if grep -q "row with all CY-A features missing" "$PKG/scripts/01_build_cya_acs_matrix.py"; then
  echo "[FAIL] obsolete all-CY-A-missing hard-stop still present" >&2; exit 1
fi
grep -q "CYA_MISSINGNESS_AUDIT.csv" "$PKG/scripts/01_build_cya_acs_matrix.py"
grep -q "NaN preserved and handled natively by XGBoost" "$PKG/scripts/01_build_cya_acs_matrix.py"
echo "[PASS] Stage07G v1.0.1 missing-value contract selftest"
grep -q "Stage08-canonical C42+SURF57+nt4" "$PKG/scripts/02_run_14arm_transfer.py"
grep -q "FEATURE_ORDER_PROVENANCE_AUDIT.json" "$PKG/scripts/02_run_14arm_transfer.py"
grep -q "historical_CS_order" "$PKG/scripts/02_run_14arm_transfer.py"
echo "[PASS] Stage07G v1.0.2 feature-order provenance audit selftest"

grep -q "auc_half_positive_negative_pair_unit" "$PKG/scripts/02_run_14arm_transfer.py"
grep -q "archived_prediction_replay_AUC_max_abs_diff" "$PKG/scripts/02_run_14arm_transfer.py"
grep -q "Loaded saved-model re-prediction is retained as a diagnostic" "$PKG/PATCH_NOTES_v1_0_3.md"
echo "[PASS] Stage07G v1.0.3 historical-evidence provenance selftest"
