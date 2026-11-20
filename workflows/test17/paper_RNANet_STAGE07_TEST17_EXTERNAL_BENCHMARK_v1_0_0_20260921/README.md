# paper_RNANet Stage07 — Test17 external benchmark

Adds one deliberately narrow external validation to RNANet:

**Test17 unified benchmark + 9 existing methods + RNANet CY-C / CY-C+HNM.**

The benchmark repository is cloned at runtime and its exact Git commit is frozen on first successful preflight. Third-party code is not redistributed in this archive.

## Quick start

From the RNANet project root:

```bash
cd ~/BioData/hgq/paper_RNANet
conda activate bioh200

tar -xzf paper_RNANet_STAGE07_TEST17_EXTERNAL_BENCHMARK_v1_0_0_20260921.tar.gz

bash paper_RNANet_STAGE07_TEST17_EXTERNAL_BENCHMARK_v1_0_0_20260921/scripts/00_selftest.sh

bash paper_RNANet_STAGE07_TEST17_EXTERNAL_BENCHMARK_v1_0_0_20260921/scripts/run_stage07a_preflight.sh \
  "$PWD"
```

After Stage07A PASS, run RNANet first:

```bash
bash paper_RNANet_STAGE07_TEST17_EXTERNAL_BENCHMARK_v1_0_0_20260921/scripts/run_stage07b_rnanet.sh \
  "$PWD"
```

Then create/check the three external benchmark environments and run all nine comparator methods at their full configured settings:

```bash
CREATE_ENVS=1 \
bash paper_RNANet_STAGE07_TEST17_EXTERNAL_BENCHMARK_v1_0_0_20260921/scripts/run_stage07c_comparators_full.sh \
  "$PWD"
```

Finally collect the manuscript-ready 11-method comparison:

```bash
bash paper_RNANet_STAGE07_TEST17_EXTERNAL_BENCHMARK_v1_0_0_20260921/scripts/run_stage07d_collect.sh \
  "$PWD"
```

Outputs live only under:

```text
~/BioData/hgq/paper_RNANet/07_test17_external_benchmark/v1_0_0/
```

No Stage05/Stage06 primary files are modified.

## Reporting contract

The existing 10,454-chain / 6,252,187-residue PDB-disjoint RNANet ligand analysis remains the paper primary analysis. Test17 is an orthogonal external validation only. RNANet evaluates exactly two frozen arms: CY-C and CY-C+HNM.

For cross-method reporting, **AUC is the headline common metric**. The benchmark-native `AUPR` values are retained as supporting results because the third-party runners do not necessarily implement PR area identically; Stage07C writes `TEST17_METRIC_IMPLEMENTATION_AUDIT.csv/json` from the exact frozen benchmark checkout. Thresholded Precision/Recall/F1/MCC are supplementary because comparator thresholding/model-selection rules differ.
