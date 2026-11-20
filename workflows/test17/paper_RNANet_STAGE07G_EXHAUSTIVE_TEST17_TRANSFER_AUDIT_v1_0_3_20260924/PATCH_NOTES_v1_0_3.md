# Stage07G v1.0.3 patch notes

v1.0.2 correctly separated the Stage08-canonical CS feature order from the historical Stage07E/F order, but it still treated a newly loaded saved-model re-prediction as an exact historical evidence replay.

The observed CS AUROC difference was `6.678063895648734e-06`, while AUPRC differed only by floating-point epsilon. This AUROC value is not arbitrary: with 191 Test17 positives and 392 negatives,

`0.5 / (191 * 392) = 6.6780638957153545e-06`.

Thus the discrepancy is exactly one half positive-negative pair in ROC rank accounting, consistent with one historical score tie becoming an infinitesimal ordering difference during a later XGBoost saved-model re-prediction on a different numerical execution path.

v1.0.3 therefore freezes the correct provenance hierarchy:

1. The original Stage07E/F prediction CSVs are the authoritative frozen evidence artifacts because the historical metrics were computed from those predictions during the original run.
2. Recomputing AUROC/AUPRC from those archived predictions must reproduce the historical JSON exactly (1e-12 tolerance).
3. Loaded saved-model re-prediction is retained as a diagnostic, including metric drift, score drift, and AUROC half-pair-equivalent drift, but is not used as the historical evidence gate.
4. Stage07G canonical CS/CS+HNM remains a new Stage08-order descriptive transfer fit and is not forced to equal Stage07E/F.

No model, feature, split, HNM setting, Test17 row, or Stage08 selection rule is changed by this patch.
