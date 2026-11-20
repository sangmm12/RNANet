#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -m py_compile "$HERE/src/stage08c_full_core_nested.py"
python "$HERE/src/stage08c_full_core_nested.py" --selftest
bash -n "$HERE/scripts/run_stage08c_full_core_nested.sh"
echo "[PASS] Stage08C package selftest"
