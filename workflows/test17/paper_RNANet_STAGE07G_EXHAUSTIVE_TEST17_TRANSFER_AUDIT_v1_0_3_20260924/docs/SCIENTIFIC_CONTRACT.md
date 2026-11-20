# Stage07G scientific contract

## Question

Does the complete frozen RNANet representation family transfer to corrected Test17, when every predefined representation/training combination is evaluated and reported?

## Frozen arms

Representations:

- A = CY-A
- C = CY-C
- S = SURF57
- AC = CY-A + CY-C
- AS = CY-A + SURF57
- CS = CY-C + SURF57
- ACS = CY-A + CY-C + SURF57

Training modes:

- standard
- HNM: top 20 predicted negative nucleotides per **training chain**, weight 5.0

Total: 14 arms.

Four A/C/G/U one-hot variables are appended exactly once to each arm. ContactCount and ContactOrder are shared by CY-A and CY-C and are included only once in AC/ACS.

Expected model dimensions:

- A = 50
- C = 46
- S = 61
- AC = 90
- AS = 107
- CS = 103
- ACS = 147

## Data and fitting

- Train60 only is used for fitting and early stopping.
- Test17 contains 17 RNA chains / 583 nucleotides.
- Same frozen five seeds: 3407, 3420, 3433, 3446, 3459.
- Exact existing `RNANET_TRAIN60_SPLIT_AUDIT.csv` is reused.
- XGBoost contract is read from the frozen Stage07 contract.
- No Test17 labels/scores are used for training, HNM ranking, early stopping, hyperparameter choice or threshold choice.
- HNM stage-1 rankings use only the training partition for the same representation.

## Reporting status

Stage07G is `POSTHOC_EXHAUSTIVE_DESCRIPTIVE_TRANSFER_AUDIT` because Test17 had already been inspected before this 14-arm audit was frozen. The audit is exhaustive and non-selective: all 14 arms must be present in the final table whether favorable or unfavorable.

Stage07G cannot reopen Stage08 selection. It is supporting external-transfer completeness evidence only.

## v1.0.1 missing-value contract

Structure-derived CY-A descriptors may be NaN for residues where the required atoms/geometric neighborhood are unavailable. This is not a row-contract failure. Stage07G v1.0.1 follows the frozen Stage08 modeling contract: NaN values are preserved for XGBoost native missing-value routing, no Train60/Test17 row is removed on representation availability, no imputation or missingness-derived predictor is introduced, and +/-inf remains fatal. Row-key merge completeness and descriptor missingness are audited separately.
