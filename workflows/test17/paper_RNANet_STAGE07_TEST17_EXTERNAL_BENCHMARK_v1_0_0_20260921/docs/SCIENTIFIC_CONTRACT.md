# Stage07 Test17 external benchmark — scientific contract

This stage is **external validation only**. It does not reopen or replace the frozen RNANet primary ligand analysis on 10,454 chains / 6,252,187 residue rows.

Only two RNANet arms are evaluated on Test17:

- `CY-C`: frozen public 42D CY-C geometry + A/C/G/U one-hot (46 predictors total), no B-factor, no SURF.
- `CY-C+HNM`: the identical predictor space with the historically frozen two-stage hard-negative mining rule: top 20 negative residues per **training chain**, hard-negative weight 5.

The external benchmark uses the benchmark-native Train60 and annotation-corrected Test17 labels. Test17 is never used for feature selection, HNM selection, early stopping, threshold tuning, or hyperparameter tuning.

For each frozen RNANet seed (3407, 3420, 3433, 3446, 3459), 10% of Train60 chains are reserved deterministically for validation/early stopping and the remaining Train60 chains are used for model fitting. Test17 is fixed across seeds. Within RNANet, AUPR remains the primary external metric and AUC is secondary. For the **cross-method headline**, AUC is the safest common metric. The benchmark-native `AUPR` field is retained as supporting evidence but is not used for a formal cross-method ranking unless the runtime metric-implementation audit confirms a uniform definition across all methods. Thresholded metrics use a fixed probability threshold of 0.5 for RNANet and remain supplementary because comparator methods have heterogeneous native thresholding/model-selection policies.

The nine comparator methods are taken from the external benchmark repository. Manuscript-ready collection is blocked unless all nine result JSONs are present and the known trainable methods are full configured runs rather than one-epoch smoke tests.
