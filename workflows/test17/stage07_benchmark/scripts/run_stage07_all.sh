#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${1:-$HOME/BioData/hgq/paper_RNANet}"
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$PKG_ROOT/scripts/run_stage07a_preflight.sh" "$PROJECT_ROOT"
bash "$PKG_ROOT/scripts/run_stage07b_rnanet.sh" "$PROJECT_ROOT"
CREATE_ENVS="${CREATE_ENVS:-0}" bash "$PKG_ROOT/scripts/run_stage07c_comparators_full.sh" "$PROJECT_ROOT"
bash "$PKG_ROOT/scripts/run_stage07d_collect.sh" "$PROJECT_ROOT"
