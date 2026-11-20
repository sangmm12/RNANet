#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${1:-$HOME/BioData/hgq/paper_RNANet}"
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$PROJECT_ROOT/07_test17_external_benchmark/v1_0_0"
BS="$OUT/vendor/rna-small-molecule-review/BindingSite"
mkdir -p "$OUT/logs"
if [[ ! -d "$BS" ]]; then
  echo "[FAIL] Missing benchmark checkout: $BS" >&2
  exit 2
fi

CREATE_ENVS="${CREATE_ENVS:-0}"
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
export CONDA_EXE="$CONDA_BIN"
if [[ -z "$CONDA_BIN" ]]; then
  echo "[FAIL] conda not found. Activate conda or set CONDA_EXE." >&2
  exit 2
fi

has_env() {
  "$CONDA_BIN" env list --json | python -c 'import json,sys,os; n=sys.argv[1]; d=json.load(sys.stdin); print(int(any(os.path.basename(x)==n for x in d.get("envs",[]))))' "$1" | grep -q '^1$'
}

if [[ "$CREATE_ENVS" == "1" ]]; then
  cd "$BS"
  if ! has_env core; then
    "$CONDA_BIN" env create -f environments/core.yml
  else
    echo "[SKIP] conda env core already exists"
  fi
  if ! has_env MVRBind; then
    "$CONDA_BIN" env create -f environments/mvrbind.yml
  else
    echo "[SKIP] conda env MVRBind already exists"
  fi
  if ! has_env Mul; then
    "$CONDA_BIN" env create -f environments/multimodrlbp.yml
  else
    echo "[SKIP] conda env Mul already exists"
  fi
fi

for e in core MVRBind Mul; do
  if ! has_env "$e"; then
    echo "[FAIL] conda env $e missing. Re-run with CREATE_ENVS=1." >&2
    exit 2
  fi
done

cd "$BS"
if [[ "$(uname -s)" == "Linux" && -d MultiModRLBP/RnaBert ]]; then
  echo "[BUILD] MultiModRLBP alignment_C extension"
  ( cd MultiModRLBP/RnaBert && "$CONDA_BIN" run --no-capture-output -n Mul python setup.py build_ext --inplace ) \
    2>&1 | tee "$OUT/logs/stage07c_multimod_extension.log"
fi

python validate_review.py 2>&1 | tee "$OUT/logs/stage07c_validate_before_full.log"
python run_all_test17.py --methods all --continue-on-error \
  2>&1 | tee "$OUT/logs/stage07c_all_9_methods_full.log"
python collect_results.py 2>&1 | tee "$OUT/logs/stage07c_collect_vendor.log"
python "$PKG_ROOT/scripts/04_audit_comparators.py" \
  --project-root "$PROJECT_ROOT" \
  2>&1 | tee "$OUT/logs/stage07c_comparator_audit.log"
echo
cat "$OUT/STAGE07C_COMPARATOR_AUDIT.json"
