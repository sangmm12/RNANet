# RNANet Stage08C — full-Core nested systematic selection v1.0.0

Stage08B is closed with the Complete-domain consensus arm `ACS+HNM`. Stage08C now selects the **Core** arm on the exact Stage05A full residue universe without using SURF.

## Frozen candidate family

- representations: `A`, `C`, `AC`
- training modes: `standard`, `HNM`
- total arms: 6
- frozen 4D nucleotide/base one-hot appended to every arm
- dimensions: A=50, C=46, AC=90 after base one-hot
- B-factor excluded
- SURF excluded from Core selection
- Test17 excluded from selection

## Nested protocol

For each frozen outer seed `3407, 3420, 3433, 3446, 3459`:

1. original frozen outer train/validation/test split is loaded from `05_ligand_full_core/splits/`;
2. only original outer-train PDBs receive 3-fold inner-score assignments;
3. the non-score PDBs are split again into PDB-disjoint fit and early-stop subsets;
4. standard and HNM variants are scored only on the untouched inner-score fold;
5. HNM ranking uses fit/train rows only, top 20 predicted negatives per training chain with weight 5.0;
6. the winning arm is selected by mean inner AUPRC;
7. original outer validation is used only for final selected-arm early stopping;
8. original outer test is used only for final outer evaluation.

The consensus Core arm is selected from the 15 untouched inner-score results only. Historical outer-test metrics are never used for consensus selection.

## Exact upstream binding

The runner requires:

- `05_ligand_full_core/matrix/matrix_contract.json`
- full-core memmaps with exact shape `6,252,187 x 96`
- the five frozen PDB split arrays
- Stage08B v1.0.1 PASS gate
- Stage08B Complete consensus `ACS+HNM`

The source audit writes SHA256 bindings before model fitting.

## Run

```bash
cd ~/BioData/hgq/paper_RNANet
conda activate bioh200

sha256sum \
  stage08c_core_selection.tar.gz

tar -xzf \
  stage08c_core_selection.tar.gz

export XGB_DEVICE=cuda
export THREADS=32

bash \
  stage08c_core_selection/scripts/run_stage08c_full_core_nested.sh \
  "$PWD"
```

## Return after completion

```bash
cat \
  08_v2_systematic_selection/full_core_nested_v1_0_0/STAGE08C_FULL_CORE_NESTED_GATE.json

echo

cat \
  08_v2_systematic_selection/full_core_nested_v1_0_0/CONSENSUS_CORE_ARM.json

echo

column -t -s, \
  08_v2_systematic_selection/full_core_nested_v1_0_0/SELECTION_FREQUENCY.csv

echo

column -t -s, \
  08_v2_systematic_selection/full_core_nested_v1_0_0/CONSENSUS_INNER_CV_RANKING.csv
```

After PASS, Stage08D will freeze the routed system: SURF-domain rows use the Stage08B Complete arm `ACS+HNM`; all remaining full-core rows use the Stage08C consensus Core arm.
