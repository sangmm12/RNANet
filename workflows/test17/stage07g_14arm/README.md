# RNANet Stage07G — exhaustive Test17 representation-family transfer audit

Release: `v1.0.3_20260924`

Stage07G is a **descriptive external-transfer completeness audit**. It runs the entire representation family

`A, C, S, AC, AS, CS, ACS x {standard, HNM}`

on the already-established corrected Train60 -> Test17 benchmark using the same five frozen Train60 train/validation splits, the same XGBoost contract, and the same frozen HNM rule.


## v1.0.3 hotfix: CY-A structural missingness

The v1.0.0 matrix builder incorrectly treated a residue with all newly extracted CY-A values missing as a fatal error. That guard was stricter than the frozen Stage08 modeling contract. Stage08 permits structure-derived `NaN` values and relies on XGBoost's native missing-value routing; only `+/-inf` is forbidden.

v1.0.3 therefore:

- distinguishes a **true row-key merge failure** from legitimate descriptor missingness;
- preserves all 3,054 Train60 rows and all 583 Test17 rows (no availability filtering);
- performs no imputation and adds no missingness-derived predictor;
- records split-level and chain-level CY-A missingness audits;
- reuses the frozen C-derived `ContactCount` / `ContactOrder` columns for shared contacts;
- rejects `+/-inf` but allows `NaN`, matching the Stage08 XGBoost contract.

This is an implementation-contract correction only. It does not change the 14-arm family, Train60/Test17 split, HNM rule, model hyperparameters, or reporting boundary.

## Scientific boundary

This stage is intentionally **not** Test17 model selection. Test17 has already been inspected historically, so Stage07G:

- does not select a winner from Test17;
- does not alter Stage08B Complete selection (`ACS+HNM`);
- does not alter Stage08C Core selection (`C+HNM`);
- does not tune thresholds, hyperparameters, feature definitions, HNM, or arm definitions on Test17;
- reports **all 14 arms regardless of performance**;
- is intended to remove selective-reporting ambiguity and quantify the transfer of CY-A, CY-C and SURF combinations.

The previously evaluated Test17 arms (`C`, `C+HNM`, `CS`, `CS+HNM`) are replayed inside the same exhaustive program and are checked against their historical output files when those files are present.

## Prerequisites already present under the project root

The current project should already contain the successful Stage07 chain:

- `07_test17_external_benchmark/v1_0_0/STAGE07A_TEST17_PREFLIGHT_GATE.json`
- `07_test17_external_benchmark/v1_0_0/STAGE07B_R5_ROW_CONTRACT_AUDIT.json`
- `07_test17_external_benchmark/v1_0_0/STAGE07B_R5_ROW_CONTRACT_AUDIT.csv`
- `07_test17_external_benchmark/v1_0_0/rnanet_test17_cyc_matrix.csv.gz`
- `07_test17_external_benchmark/v1_0_0/RNANET_TRAIN60_SPLIT_AUDIT.csv`
- `07_test17_external_benchmark/v1_0_0/posthoc_surf_v1_0_0/STAGE07E_SURF_EXTRACT_GATE.json`
- `07_test17_external_benchmark/v1_0_0/posthoc_surf_v1_0_0/rnanet_test17_cyc_surf_matrix.csv.gz`

The R5 row-contract helper must remain installed in the original Stage07 package at:

`stage07_benchmark/scripts/stage07b_row_contract_r5.py`

## Run

```bash
cd ~/BioData/hgq/paper_RNANet
conda activate bioh200

sha256sum \
  stage07g_14arm.tar.gz

tar -xzf \
  stage07g_14arm.tar.gz

export XGB_DEVICE=cuda
export THREADS=32

bash \
  stage07g_14arm/scripts/run_stage07g_exhaustive_test17.sh \
  "$PWD"
```

Stage07G reuses the existing SURF matrix, so **MSMS is not rerun**.

## Primary outputs

Under:

`07_test17_external_benchmark/v1_0_0/exhaustive_14arm_transfer_v1_0_1/`

- `STAGE07G_CYA_MATRIX_GATE.json`
- `STAGE07G_EXHAUSTIVE_TEST17_GATE.json`
- `TEST17_14ARM_COMPLETE_TABLE.csv`
- `TEST17_14ARM_PER_SEED.csv`
- `TEST17_CYA_INCREMENT_CONTRASTS.csv`
- `HISTORICAL_ARM_REPLAY_AUDIT.csv`
- `REPORTING_BOUNDARY.json`
- per-arm saved models and Test17 prediction files

## What to return

```bash
cat \
  07_test17_external_benchmark/v1_0_0/exhaustive_14arm_transfer_v1_0_1/STAGE07G_EXHAUSTIVE_TEST17_GATE.json

echo
column -t -s, \
  07_test17_external_benchmark/v1_0_0/exhaustive_14arm_transfer_v1_0_1/TEST17_14ARM_COMPLETE_TABLE.csv

echo
column -t -s, \
  07_test17_external_benchmark/v1_0_0/exhaustive_14arm_transfer_v1_0_1/TEST17_CYA_INCREMENT_CONTRASTS.csv
```

## v1.0.3 feature-order provenance fix

Historical Stage07E/F C+S used feature order `C42 + nt4 + SURF57`, while the Stage08-canonical exhaustive family uses `C42 + SURF57 + nt4`. Because XGBoost feature subsampling is enabled (`colsample_bytree=0.8`), these are different deterministic fit contracts despite identical feature sets. v1.0.3 audits both contracts explicitly and preserves the Stage08-canonical order for the 14-arm table. See `PATCH_NOTES_v1_0_2.md`.


## v1.0.3 provenance hotfix
Historical Stage07E/F prediction CSVs, rather than a later saved-model re-prediction, are the authoritative metric replay artifacts. See `PATCH_NOTES_v1_0_3.md`.
