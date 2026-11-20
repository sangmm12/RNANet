#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -m py_compile "$ROOT/src/stage08e_interpretation.py"
grep -q "core_partition_calibrated_control" "$ROOT/src/stage08e_interpretation.py"
grep -q "historical_outer_test_results_previously_seen" "$ROOT/src/stage08e_interpretation.py"
grep -q "exact_two_sided_sign_probability_descriptive" "$ROOT/src/stage08e_interpretation.py"
echo "[PASS] Stage08E routed interpretation selftest"
