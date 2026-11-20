#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${1:?usage: run_stage07e_surf.sh PROJECT_ROOT}"
TOOL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MSMS_BIN="${MSMS_BIN:-$(command -v msms || true)}"
JOBS="${JOBS:-8}"
[[ -n "$MSMS_BIN" && -x "$MSMS_BIN" ]] || { echo '[FAIL] set MSMS_BIN to executable msms' >&2; exit 2; }
OUT="$PROJECT_ROOT/07_test17_external_benchmark/v1_0_0"
mkdir -p "$OUT/logs"
bash "$TOOL_ROOT/scripts/00_selftest.sh"
python "$TOOL_ROOT/scripts/01_extract_test17_surf.py" --project-root "$PROJECT_ROOT" --tool-root "$TOOL_ROOT" --msms-bin "$MSMS_BIN" --jobs "$JOBS" 2>&1 | tee "$OUT/logs/stage07e_posthoc_surf_extract_20260921.log"
python "$TOOL_ROOT/scripts/02_train_cyc_surf_test17.py" --project-root "$PROJECT_ROOT" --tool-root "$TOOL_ROOT" 2>&1 | tee "$OUT/logs/stage07e_posthoc_cyc_surf_train_20260921.log"
cat "$OUT/posthoc_surf_v1_0_0/STAGE07E_SURF_EXTRACT_GATE.json"
cat "$OUT/posthoc_surf_v1_0_0/STAGE07E_POSTHOC_CYC_SURF_TEST17_GATE.json"
