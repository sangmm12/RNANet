#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "$HERE/src/stage08d_fixed_routed.py" --selftest
