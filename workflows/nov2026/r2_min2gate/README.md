# RNANet NC Minimal 2-Gate R2: source-anchor and length-eligibility correction

## Motivation: targeted repair only, NO new research arms

R1 is a valid computer execution but its 100 C_ONLY candidates comprise 98 pairs of 3-nt chains and 2 pairs of 4-nt chains; their high TM scores are not meaningful RNA global topology retrieval. R1 rotation also compared CY-C-derived B_avg to the 33D retrieval master whose base columns were sourced from the frozen CY-A aggregation. Neither is evidence of a broken frozen Stage04B experiment.

## Frozen R2 policy

- Entire original 1,406-chain master is retained as the immutable input and used for z-score fits.
- Before kNN and p20 thresholds, restrict each of the *same three retrieval arms* to chains with sequence length >= 20 nt. All arms use the same eligible subset. No manual selection of favorable candidates.
- Confirm exported 3D structures also have >=20 residues on both sides. This is a QC gate, not a second scoring filter. A length-stratified descriptive table is generated without additional alignment runs.
- Keep kNN=100, p20=20, Jaccard<=0.35, topN=100, original fixed 67/33 geometry weight, no extra conditional Geo/FM gates, same fixed descriptive Top-K and US-align.
- Rotation source reconstruction: CY-A origin for ContactCount, ContactOrder, B_avg (original CY-A 6-atom B average); CY-C origin for 30 multiscale geometry columns. Each re-extracted 33D vector must agree with the frozen C master before the effect is interpreted.
- Original Stage04B 90 candidate set is unchanged and remains the primary historical result. Do not replace its interpretation with R2 descriptive retrieval.
- No matched-null p-value claim for R2. No new model training, no ligand re-label, no parameter search.

## Execution (place R2 ZIP and SHA256 in paper_RNANet)

```bash
conda activate bioh200
ROOT="$HOME/BioData/hgq"
cd "$ROOT/paper_RNANet" || exit 1
P=RNANET_NC_MIN2GATE_R2_LENGTH20_SOURCEANCHOR_20261112
sha256sum -c "$P.zip.sha256" || exit 1
python -m zipfile -e "$P.zip" "$ROOT" || exit 1
bash "$ROOT/$P/scripts/run_two_gates.sh" "$ROOT" && bash "$ROOT/$P/scripts/package_results.sh" "$ROOT"
```

Upload `paper_RNANet/exports/RNANET_NC_MIN2GATE_REVIEW_20261112_NC_MIN2GATE_R2_LENGTH20.tar.gz` and its `.sha256`.

## Scientific decision rule

The rotation gate is expected to report coordinate-frame sensitivity for the 12 fixed-J multiscale indicators; this is an important *negative* finding, not a software failure. If source anchor passes, the existing data can be interpreted cautiously with limitations explicitly stating non-invariance. Stronger invariance claims are not supported. If anchor fails, inspect the precise frozen source files before continuing. R2 high TM on short chains is prevented by common 20-nt eligibility.
