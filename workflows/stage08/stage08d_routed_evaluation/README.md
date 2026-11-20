# RNANet Stage08D — fixed Core/Complete routed evaluation v1.0.0

Stage08B froze the Complete consensus arm `ACS+HNM`. Stage08C froze the full-universe Core consensus arm `C+HNM` with 5/5 outer-seed selection agreement. Stage08D performs **no candidate or arm selection**.

## Frozen routed system

- Core branch: `C+HNM`, 46 dimensions including the frozen 4D base one-hot.
- Complete branch: `ACS+HNM`, 147 dimensions including the frozen 4D base one-hot.
- Route: rows in the frozen, label-blind Stage05A SURF applicability domain use Complete; all other full-Core rows use Core.
- B-factor: excluded.
- Test17: not used.

## Why calibration is part of Stage08D

Core and Complete are separately trained models with different applicability domains and HNM weighting. Their raw XGBoost outputs are therefore not assumed to share a common probability scale. Directly concatenating raw branch scores can distort pooled AUPRC/AUC through arbitrary cross-branch score ordering.

For each frozen outer seed, Stage08D therefore:

1. fits each fixed arm on its original outer-train rows;
2. creates **one shared PDB-disjoint 50/50 split of the frozen outer-validation PDBs** into early-stop and calibration PDBs, with stratification chosen to retain positive and negative SURF-domain validation PDBs on both sides;
3. uses only the shared early-stop PDBs for XGBoost early stopping in both branches;
4. fits low-dimensional Platt calibration on the shared calibration PDBs using natural, unweighted labels;
5. applies branch/domain-specific calibration before combining branch scores on the untouched outer test.

The raw pooled routed score is retained only as a diagnostic. The primary pooled routed metrics are the calibrated routed metrics.

## Calibration-matched Core control

To prevent a domain-calibration gain from being misattributed to the `ACS` Complete representation, Stage08D also evaluates a matched control:

- the same `C+HNM` Core model is used on **both** non-SURF and SURF test rows;
- Core scores are calibrated separately for non-SURF and SURF domains using the same shared calibration PDBs;
- this is reported as `core_partition_calibrated_control`.

Therefore the main representation substitution contrast is:

`routed_calibrated - core_partition_calibrated_control`

The ordinary globally calibrated `C+HNM` system is also retained as `core_only_global_calibrated`.

## Fixed HNM rule

For both branches:

- hard-negative ranking uses outer-train rows only;
- top 20 predicted negatives per training chain;
- selected hard negatives receive weight 5.0;
- outer-validation calibration rows, outer-test rows and Test17 are never used for HNM ranking.

## Source binding

The run refuses to proceed unless:

- Stage08B gate is exactly `PASS_STAGE08B_COMPLETE_NESTED_SELECTION_V101`;
- Stage08B consensus is exactly `ACS+HNM` and did not use outer-test/Test17 for consensus selection;
- Stage08C gate is exactly `PASS_STAGE08C_FULL_CORE_NESTED_SELECTION_V100`;
- Stage08C consensus is exactly `C+HNM`, did not use SURF, outer-test or Test17 for Core selection, and is bound to the Stage08B Complete consensus;
- Stage05A full-Core and SURF-domain contracts, dimensions and raw memmap sizes match exactly.

All source contracts, SURF domain index, SURF57 matrix, upstream gates/consensus files and outer split arrays are SHA256-bound into `STAGE08D_SOURCE_BINDING.json`.

The run is resumable at completed outer-seed checkpoints.

## Run

```bash
cd ~/BioData/hgq/paper_RNANet
conda activate bioh200

sha256sum \
  stage08d_routed_evaluation.tar.gz

tar -xzf \
  stage08d_routed_evaluation.tar.gz

export XGB_DEVICE=cuda
export THREADS=32

bash \
  stage08d_routed_evaluation/scripts/run_stage08d_fixed_routed.sh \
  "$PWD"
```

## Return after completion

```bash
cat \
  08_v2_systematic_selection/routed_fixed_v1_0_0/STAGE08D_FIXED_ROUTED_GATE.json

echo

cat \
  08_v2_systematic_selection/routed_fixed_v1_0_0/FROZEN_ROUTED_SYSTEM.json

echo
echo "===== ROUTED RESULTS ====="
column -t -s, \
  08_v2_systematic_selection/routed_fixed_v1_0_0/ROUTED_RESULTS.csv

echo
echo "===== SURF BRANCH COMPARISON ====="
column -t -s, \
  08_v2_systematic_selection/routed_fixed_v1_0_0/SURF_BRANCH_COMPARISON.csv

echo
echo "===== CALIBRATION AUDIT ====="
column -t -s, \
  08_v2_systematic_selection/routed_fixed_v1_0_0/CALIBRATION_AUDIT.csv
```

## Interpretation boundary

Stage08D outer-test results evaluate the already-frozen routed system and must **not** be used to reopen arm selection. A negative or null routed increment is a valid result. Historical outer-test results were seen earlier in the project, so a future untouched external confirmation remains required for a fully independent fixed-system claim.
