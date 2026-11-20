# Stage07G v1.0.2 patch notes

## Failure addressed
v1.0.1 completed all 70 canonical fits and then stopped at the historical replay guard because CS and CS+HNM did not numerically reproduce Stage07E/F.

## Root cause
This was not a row, label, SURF, CY-C, or HNM data mismatch. It was a feature-order contract difference:

- Historical Stage07E/F CS: `C42 + nt4 + SURF57`
- Stage08 / Stage07G canonical CS: `C42 + SURF57 + nt4`

Both contain the same 103 predictors. XGBoost uses `colsample_bytree=0.8`; feature subsampling is index based, so permuting predictor columns can change the deterministic fitted model and its Test17 metrics.

## v1.0.2 behavior
- Keeps the Stage08-canonical order for the exhaustive 14-arm transfer table.
- Requires exact numerical replay for C and C+HNM, whose historical and canonical orders are identical.
- For historical CS and CS+HNM, verifies the exact expected feature-set/order permutation and replays the byte-frozen Stage07E/F saved models on the exact current Test17 matrix under the historical feature order.
- Writes `FEATURE_ORDER_PROVENANCE_AUDIT.json` and an expanded `HISTORICAL_ARM_REPLAY_AUDIT.csv`.
- Does not use Test17 to select or tune any arm and does not reopen Stage08 Core/Complete selection.
- Automatically copies the already-completed v1.0.1 matrix/checkpoints/models into `exhaustive_14arm_transfer_v1_0_2`, so the 70 fits should resume rather than retrain.

## Reporting rule
The complete 14-arm Stage07G table is the canonical Stage08-order descriptive transfer audit. Historical Stage07E/F C+S values remain a separate chronology-preserving external sensitivity result and must not be silently replaced by the canonical Stage07G CS numbers.
