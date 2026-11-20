#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -m py_compile "$HERE/scripts/01_extract_test17_surf.py" "$HERE/scripts/02_train_cyc_surf_test17.py" "$HERE/../../../src/rnanet/io.py" "$HERE/../../../src/rnanet/surf.py"
[[ "$(sha256sum "$HERE/../../../src/rnanet/surf.py" | awk '{print $1}')" == "7cab38601388a6a66d6250c4d00e0b0c027abb2d5f1c12634f5785ed9a0a47aa" ]]
[[ "$(sha256sum "$HERE/../../../src/rnanet/io.py" | awk '{print $1}')" == "0f4554f67d2770598c89491ef1fa28197c57ec5db087700299d042a4096429dc" ]]
grep -q 'FIELDS=("Keff"' "$HERE/../../../src/rnanet/surf.py"
echo '[PASS] Stage07E post-hoc SURF package selftest'
