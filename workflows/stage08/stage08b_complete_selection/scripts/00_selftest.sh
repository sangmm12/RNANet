#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -m py_compile "$HERE/src/stage08b_complete_nested.py"
python "$HERE/src/stage08b_complete_nested.py" --selftest
bash -n "$HERE/scripts/run_stage08b_complete_nested.sh"
echo "[PASS] Stage08B v1.0.1 package selftest"
