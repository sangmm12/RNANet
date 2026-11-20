# RNANet Stage08B Complete-domain nested systematic selection v1.0.1

This replaces v1.0.0. v1.0.0 aborted before model training because a mapping defined only for outer-train PDBs was applied to all SURF-domain rows, so outer validation/test PDB indices raised `KeyError`.

v1.0.1 fixes that indexing bug and strengthens the nested design:

- inner-fold codes are assigned **only** to original outer-train rows; outer validation/test rows remain `-1`;
- each inner candidate score fold is untouched by fitting and early stopping;
- the remaining outer-train PDBs are split again into PDB-disjoint **fit** and **early-stop** subsets;
- HNM ranking uses fit/train rows only; its stage-1 model uses the separate early-stop subset;
- original outer validation is used only for the final selected-arm early stopping;
- original outer test is used only for final outer evaluation;
- Test17 is never used for Stage08B selection;
- per-inner-fold checkpoints make the long run resumable.

Candidate family on the exact 431,226-row SURF domain:
`A/C/S/AC/AS/CS/ACS x standard/HNM` (14 arms), always with the frozen 4D nucleotide one-hot and no B-factor.

Output directory:
`08_v2_systematic_selection/complete_nested_v1_0_1/`

Run:
```bash
cd ~/BioData/hgq/paper_RNANet
conda activate bioh200
export XGB_DEVICE=cuda
export THREADS=32
bash stage08b_complete_selection/scripts/run_stage08b_complete_nested.sh "$PWD"
```
