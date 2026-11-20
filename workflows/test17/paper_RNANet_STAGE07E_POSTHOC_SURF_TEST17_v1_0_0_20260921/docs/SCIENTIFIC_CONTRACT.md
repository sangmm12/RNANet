# Stage07E post-hoc SURF sensitivity contract

This is an explicitly post-hoc exploratory sensitivity requested after the frozen Stage07B CY-C / CY-C+HNM Test17 results had already been inspected.

It MUST NOT replace or redefine the prespecified Stage07 external benchmark arms.

Frozen before this Stage07E run:

- dataset: Train60 -> Test17 from benchmark commit `f1cea5cfc04541dda22d0c027f87cb0e20067013`;
- Test17: 17 chains / 583 nucleotide rows / 191 positives;
- row universe: Stage07B-R5 resolved-coordinate contract;
- CY-C matrix and 46D model feature set from frozen Stage07B;
- SURF implementation: exact RNANet GitHub v2.0.0 public `surf.py`;
- MSMS: probe radius 1.5 A, density 3.0, no convex-hull fallback;
- SURF57: 14 fields x {mean,std,min,max} + `n_verts`;
- classifier: same XGBoost hyperparameters as Stage07B;
- seeds and Train60 train/validation chain assignments: exact reuse of Stage07B;
- no Test17 tuning, threshold selection, feature selection, rescue, or parameter search.

Only one exploratory arm is evaluated: `CY-C + SURF` (46 + 57 = 103 model variables).

The Test17 result is descriptive/post-hoc and is not eligible for the prespecified primary external-comparison claim.
