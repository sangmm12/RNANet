# RNANet

Source code for multiscale RNA geometric descriptors and their application to sequence-divergent structural retrieval, RNA–small-molecule binding-site prediction and RNA–protein interface prediction.

- **CY-A** — conformational geometry in periodic angle space (η, θ, χ, ribose pseudorotation)
- **CY-C** — Cartesian backbone geometry (P, C4′, C3′, O3′)
- **SURF** — molecular-surface geometry (MSMS surface)

## Layout

```
src/rnanet/   geometry library (io, cya, cyc, surf, retrieval, splits, metrics)
scripts/      feature extraction, kNN retrieval, XGBoost training
workflows/    analysis pipelines used in the manuscript
config/       frozen protocols and feature contracts
tests/        unit tests
```

## Installation

```bash
conda env create -f environment.yml && conda activate rnanet-repro
pip install -e .
python tests/test_core.py
```

External tools: [MSMS](https://ccsb.scripps.edu/msms/) (SURF) and [US-align](https://zhanggroup.org/US-align/) (structural alignment).

## Usage

```bash
python scripts/extract_cya.py  1abc.cif --chain A -o cya.csv                          # 46 features/nt
python scripts/extract_cyc.py  1abc.cif --chain A -o cyc.csv                          # 42 features/nt
python scripts/extract_surf.py 1abc.cif --chain A --msms-bin /path/to/msms -o surf.csv # 57 features/nt
python scripts/retrieval_knn.py chain_features.csv -k 100 -o knn.csv
python scripts/train_grouped_xgb.py matrix.parquet --features features.txt --group-col pdb_id --seed 3407 -o run/
```

Chain identifiers are case-sensitive. Training uses the frozen XGBoost settings in `config/frozen_protocol.json` with PDB-disjoint splits (72/8/20).

## Feature dimensions

| | Residue level (site prediction) | Chain level (retrieval) |
| --- | --- | --- |
| CY-A | 46 | 34 |
| CY-C | 42 (scales 4–32, step 4) | 33 (scales 4–44, step 8) |
| SURF | 57 | 57 |
| RNA-FM | — | 640 |

See `config/nov2026_frozen_contract.json`.

## Workflows

| Folder | Analysis |
| --- | --- |
| `retrieval_stage04/` | Structural retrieval feature contracts |
| `nov2026/r2_min2gate/` | Retrieval sensitivity analysis (CY-C / RNA-FM / combined) |
| `nov2026/r102_frozen_core/` | Rotation sensitivity of the frozen CY-C + HNM model |
| `test17/` | Train60→Test17 external benchmark (Stage07, 07E–07G) |
| `stage08/` | Nested Core/Complete selection and routed evaluation (Stage08A–E) |

Each pipeline has its own README and self-test. The `run_*.sh` wrappers contain the authors' server paths.

## Data sources

- **RNA structures**: [RCSB PDB](https://www.rcsb.org) (mmCIF) — retrieval set of 1,406 chains (resolution < 3.0 Å); ligand set of 10,454 chains; RNA–protein interface set of 11,700 chains from 2,919 complexes.
- **Train60→Test17 benchmark**: [rna-small-molecule-review](https://github.com/cschen-y/rna-small-molecule-review) (`BindingSite/data/common`).
- **RNA-FM**: `multimolecule/rnafm` pretrained model (retrieval only).

