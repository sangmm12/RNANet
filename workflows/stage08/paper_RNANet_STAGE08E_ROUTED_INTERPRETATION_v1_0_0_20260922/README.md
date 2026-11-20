# paper_RNANet Stage08E — routed-system interpretation and manuscript integration

Stage08E is analysis-only. It does **not** train, tune, select, or rescore models.
It consumes the frozen Stage08D outputs and produces an auditable interpretation bundle for manuscript integration.

Frozen system expected from Stage08D:

- Core branch: `C+HNM`
- Complete branch: `ACS+HNM`
- Route: frozen Stage05A SURF applicability-domain rows -> Complete; all other full-Core rows -> Core
- Primary comparator: Core `C+HNM` on both domains with matched separate SURF/non-SURF Platt calibration
- Test17: unused
- Stage08D candidate selection: forbidden / absent

Stage08E reports:

1. Primary routed-minus-matched-Core deltas for AUPRC/AUC/Brier/log loss.
2. Same-row SURF-domain Complete-minus-Core ranking deltas.
3. SURF coverage and positive-enrichment summaries.
4. Split-level directional consistency. With only five outer seeds, exact sign probabilities are reported as descriptive resampling-consistency statistics, not biological-unit inferential tests.
5. Claim boundaries caused by historical reuse of the frozen outer-test splits.
6. Ready-to-integrate Methods/Results text and machine-readable source-data tables.

## Run

```bash
cd ~/BioData/hgq/paper_RNANet
conda activate bioh200

sha256sum \
  paper_RNANet_STAGE08E_ROUTED_INTERPRETATION_v1_0_0_20260922.tar.gz

tar -xzf \
  paper_RNANet_STAGE08E_ROUTED_INTERPRETATION_v1_0_0_20260922.tar.gz

bash \
  paper_RNANet_STAGE08E_ROUTED_INTERPRETATION_v1_0_0_20260922/scripts/run_stage08e_interpretation.sh \
  "$PWD"
```

## Return

```bash
cat 08_v2_systematic_selection/routed_interpretation_v1_0_0/STAGE08E_INTERPRETATION_GATE.json

echo
column -t -s, 08_v2_systematic_selection/routed_interpretation_v1_0_0/ROUTED_EFFECT_SUMMARY.csv

echo
column -t -s, 08_v2_systematic_selection/routed_interpretation_v1_0_0/SURF_BRANCH_EFFECT_SUMMARY.csv

echo
cat 08_v2_systematic_selection/routed_interpretation_v1_0_0/MANUSCRIPT_RESULTS_DRAFT.md
```
