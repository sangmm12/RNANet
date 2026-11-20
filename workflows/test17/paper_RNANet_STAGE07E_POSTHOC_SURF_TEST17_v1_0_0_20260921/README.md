# RNANet Stage07E post-hoc CY-C+SURF Test17 sensitivity

This package evaluates exactly one exploratory arm: frozen CY-C + frozen SURF57 on the already-inspected Test17 benchmark. It is explicitly post-hoc and cannot replace the prespecified Stage07B external arms.

Run from `~/BioData/hgq/paper_RNANet` after Stage07B-R5 has PASSed:

```bash
conda activate bioh200
export MSMS_BIN="$(which msms)"
export XGB_DEVICE=cuda
export THREADS=32
export JOBS=8
bash paper_RNANet_STAGE07E_POSTHOC_SURF_TEST17_v1_0_0_20260921/scripts/run_stage07e_surf.sh "$PWD"
```

Expected pipeline gates (performance is never a gate):

- `PASS_STAGE07E_POSTHOC_SURF_EXTRACT`
- `PASS_STAGE07E_POSTHOC_CYC_SURF_TEST17`

Outputs are written under `07_test17_external_benchmark/v1_0_0/posthoc_surf_v1_0_0/`.
