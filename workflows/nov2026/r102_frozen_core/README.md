# RNANet NC frozen Core prediction rotation audit - R102

This package is the exact Stage08D **CY-C+HNM Core** frozen model prediction rotation check, **not** new model training or a rerun of RNA-FM retrieval. It uses the V101 source-audit evidence to bind the original 46D Stage08D model and saved TEST `core_raw` probabilities directly to the Stage05A `X_intrinsic.f32` matrix.

## What was verified before creating the package

- Stage08D source `stage08d_fixed_routed.py`: `core_subset` consumes 42 ordered CY-C-related columns `[10,11,56,...,95]` + four A/C/G/U one-hot columns. It does **not** fit a scaler or use an RNA-FM embedding.
- Stage08D stores exact held-out test row indices, labels, `core_raw`, and `core_global_calibrated` probabilities in `predictions/seed3407_routed_test.npz` (and other seed files).
- Stage05A `prepare_full_core_input.py` freezes **case-sensitive chain-specific symlinks** named `input_pdb/{pdb.lower()}_{chain}.pdb`. R102 uses those exact links; it never looks up a different public PDB version.
- Stage05A CY-C extractor is preserved unchanged (SHA256 `a4be2d6d040f43c3eb42d98af27ac201e37520bbe5a6d2156f0ed57167a18c29`) and contains the same local feature formulas as the late Stage02 source. Original 40D CY-C feature values must match the frozen matrix before any rotation results are scored.
- Stage08D five Core model SHA256 hashes were copied from the uploaded V101 inventory; each is verified at runtime.

## Run on original as-4125 server

Copy the **ZIP** to `/mnt/202001Main/BioData/hgq/paper_RNANet`, and run:

```bash
conda activate bioh200
ROOT=/mnt/202001Main/BioData/hgq
PROJECT="$ROOT/paper_RNANet"
P=RNANET_NC_FROZEN_CORE_ROTATION_R102_20261112
cd "$PROJECT"
# Put the checksum printed alongside the download into this file if it did not download.
sha256sum -c "$P.zip.sha256"
python -m zipfile -e "$P.zip" "$ROOT"
python "$ROOT/$P/tests/smoke.py"
bash "$ROOT/$P/scripts/run_frozen_core_rotation.sh" "$ROOT"
```

The wrapper **does not exit your SSH session** on a scientific HOLD. It always packages `GATE.json`, the log and any partial evidence for review. Return `paper_RNANet/exports/RNANET_NC_FROZEN_CORE_ROTATION_R102_*.tar.gz` and `.sha256`.

## Scientific contract

1. Select up to 12 full held-out TEST chains by a fixed seed and length strata (20-80, 81-200, 201-500 nt). Only source availability and baseline feature matching may exclude a chain; all exclusions are reported. No labels/predictions are used in selection.
2. Verify PDB chain identity; use frozen Stage05A source symlinks. Re-extract original 40 CY-C features at eight scales. Anchor them, along with the two contact descriptors, to the saved Stage05A matrix. Identity, translation and proper SO(3) controls are mandatory.
3. Bind the exact Stage08D `core_C_HNM.json` and 46D feature order. Before rotation inference, require original raw predictions and frozen global Platt calibrated outputs to replay the stored TEST arrays within 2e-6. No test set fitting is allowed.
4. For three random rotations, only 40 CY-C feature values are re-extracted from rotated physical RNA backbone atoms. Other Stage08D model inputs are unchanged. Compute three paired interventions: all 40 features, two rotation-sensitive families only, and the remaining three families only.
5. Report same-row probability differences, within-chain top-ranked residues, and subset AUPRC/AUROC if both labels are present. **Do not** compare subset metrics to the full-universe manuscript AUPRC. An empirical sensitivity result is not an unconditional claim of rotation invariance.

## Failure policy

If the existing frozen checkpoint differs from the V101 inventory, any original feature cannot be reproduced from its original chain symlink, raw or calibrated TEST predictions cannot be replayed, or fewer than four source chains are technically verifiable, the script outputs a **HOLD** with details. It does not retrain, silently substitute a different model, or claim PASS.

The R102 result is the last **targeted** rotation check. It does not redesign the CY-C method, add feature selection, or replace any original Stage04B/R15 results.
