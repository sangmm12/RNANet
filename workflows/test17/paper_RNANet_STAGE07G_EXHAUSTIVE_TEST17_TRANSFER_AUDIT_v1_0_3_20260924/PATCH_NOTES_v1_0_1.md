# Stage07G v1.0.1 hotfix

The v1.0.0 failure `RuntimeError: row with all CY-A features missing` was caused by an over-strict matrix-build guard, not by a row-contract failure. The frozen Stage08 predictors allow structure-derived NaN values and XGBoost handles them natively. v1.0.1 preserves the entire Test17 universe, audits missingness explicitly, forbids only +/-inf, and introduces no imputation or availability-based row filtering.
