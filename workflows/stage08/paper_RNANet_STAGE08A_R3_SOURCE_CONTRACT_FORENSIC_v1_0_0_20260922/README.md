# RNANet Stage08A-R3 — source-contract forensic audit

Purpose: resolve why Stage08A-R2 could not bind a single aggregate Stage05A table.

This package performs **no training** and **no geometry recomputation**. It inventories binary/aggregate assets that R2 did not inspect, audits Stage05A source-code I/O, inspects existing result archives, and determines whether the v2 systematic-selection input should be bound to an existing aggregate matrix or deterministically rebuilt from the frozen Stage05A source contract.

It explicitly keeps Test17 outside v2 model selection.

Run:

```bash
cd ~/BioData/hgq/paper_RNANet
conda activate bioh200

bash \
  paper_RNANet_STAGE08A_R3_SOURCE_CONTRACT_FORENSIC_v1_0_0_20260922/scripts/run_stage08a_r3_forensic.sh \
  "$PWD"
```

Return:

```bash
cat 08_v2_systematic_selection/source_forensic_r3_v1_0_0/STAGE08A_R3_SOURCE_CONTRACT_GATE.json
cat 08_v2_systematic_selection/source_forensic_r3_v1_0_0/FULL_CORE_MATRIX_DIR_FILES.tsv
cat 08_v2_systematic_selection/source_forensic_r3_v1_0_0/SURF_AGGREGATE_CANDIDATES.tsv
head -n 120 08_v2_systematic_selection/source_forensic_r3_v1_0_0/SOURCE_IO_EVIDENCE.tsv
```
