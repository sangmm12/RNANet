# Stage08D protocol notes

The primary Stage08D estimand is the performance of a fixed, label-blind applicability-domain router on each frozen outer test split.

The direct within-SURF comparison (`complete_on_surf_raw` versus `core_on_surf_raw`) is valid for AUPRC/AUC without cross-model calibration because both scores are evaluated on exactly the same SURF test rows and ranking metrics are invariant to monotone within-model calibration.

The pooled full-test routed score requires cross-branch comparability. Therefore the headline pooled route uses Platt calibration fitted only on a shared, PDB-disjoint calibration half of the frozen outer-validation PDBs. The other half is reserved for early stopping. No outer-test row is used for fitting, early stopping, HNM ranking, calibration or routing-rule construction.

A second control keeps `C+HNM` on both domains while applying the same non-SURF/SURF domain-specific calibration. This calibration-matched Core control is the primary comparator for deciding whether substituting `ACS+HNM` on the SURF applicability domain adds predictive value beyond domain calibration itself.

A negative or null routed increment is admissible and must not trigger re-selection of Core/Complete arms using Stage08D outer-test performance.
