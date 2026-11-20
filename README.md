# RNANet

RNANet describes RNA three-dimensional structures with three families of nucleotide-level geometric descriptors and applies them to sequence-divergent structural retrieval, RNA–small-molecule binding-site prediction and RNA–protein interface prediction.

| Descriptor | Geometry | Features per nucleotide |
| --- | --- | --- |
| **CY-A** | Conformational angles (η, θ, χ, ribose pseudorotation) in periodic angle space | 46 |
| **CY-C** | Cartesian backbone coordinates (P, C4′, C3′, O3′), eight neighborhood scales | 42 |
| **SURF** | Molecular surface (MSMS), area-distortion and curvature fields | 57 |

## Repository structure

```
RNANet/
├── src/rnanet/                  Descriptor library: io, cya, cyc, surf
├── scripts/                     Command-line feature extraction
├── config/frozen_protocol.json  Scales, contact cutoff, XGBoost settings
├── tests/                       Unit tests
└── workflows/                   Analysis pipelines used in the manuscript
    ├── retrieval_stage04/       Structural retrieval
    ├── nov2026/                 Retrieval sensitivity and rotation analyses
    ├── stage08/                 RNA–small-molecule binding-site prediction
    └── test17/                  Train60→Test17 external benchmark
```

## Installation

```bash
conda env create -f environment.yml
conda activate rnanet-repro
pip install -e .
python tests/test_core.py
```

Requires Python ≥ 3.10. SURF requires [MSMS](https://ccsb.scripps.edu/msms/); structural alignment requires [US-align](https://zhanggroup.org/US-align/).

## Feature extraction

```bash
python scripts/extract_cya.py  1abc.cif --chain A -o cya.csv
python scripts/extract_cyc.py  1abc.cif --chain A -o cyc.csv
python scripts/extract_surf.py 1abc.cif --chain A --msms-bin /path/to/msms -o surf.csv
```

Each command writes one row per nucleotide. Chain identifiers are case-sensitive, and nucleotides lacking both P and C4′ are skipped.

## Analysis workflows

Each pipeline folder contains its own `README.md`, a `run_*.sh` entry script (written for the authors' server paths) and a self-test; `retrieval_stage04/` contains only the feature contracts and the master builder.

**Structural retrieval**

| Folder | Content |
| --- | --- |
| `retrieval_stage04/` | Chain-level feature contracts and SURF-combination master builder |
| `nov2026/r2_min2gate/` | Sensitivity analysis of CY-C-only, RNA-FM-only and combined retrieval with US-align evaluation |

**RNA–small-molecule binding-site prediction**

| Folder | Content |
| --- | --- |
| `stage08/stage08b_complete_selection/` | Nested PDB-disjoint selection of the Complete arm (with SURF) |
| `stage08/stage08c_core_selection/` | Nested PDB-disjoint selection of the Core arm (without SURF) |
| `stage08/stage08d_routed_evaluation/` | Applicability-aware routed evaluation |
| `stage08/stage08e_interpretation/` | SURF-domain coverage and binding-site enrichment |

**Train60→Test17 external benchmark**

| Folder | Content |
| --- | --- |
| `test17/stage07_benchmark/` | CY-C models and comparison with existing predictors |
| `test17/stage07e_surf/` | CY-C + SURF |
| `test17/stage07f_surf_hnm/` | CY-C + SURF with hard negative mining |
| `test17/stage07g_14arm/` | All 14 representation × training-mode combinations |

**Rotation sensitivity**

| Folder | Content |
| --- | --- |
| `nov2026/r102_frozen_core/` | Frozen CY-C + HNM model evaluated on rigidly rotated structures |

## Testing

```bash
export PYTHONPATH=$PWD/src
python tests/test_core.py
for s in workflows/test17/*/scripts/00_selftest.sh workflows/stage08/*/scripts/00_selftest.sh; do bash "$s"; done
for t in workflows/nov2026/r2_min2gate/tests/*.py; do python "$t"; done
python workflows/nov2026/r102_frozen_core/tests/smoke.py
```

## Data sources

| Data | Source |
| --- | --- |
| RNA structures | [RCSB PDB](https://www.rcsb.org), mmCIF format: retrieval set of 1,406 chains (resolution < 3.0 Å); binding-site set of 10,454 chains; interface set of 11,700 chains from 2,919 RNA–protein complexes |
| Train60/Test17 benchmark | [rna-small-molecule-review](https://github.com/cschen-y/rna-small-molecule-review), `BindingSite/data/common` |
| RNA-FM embeddings | Pretrained `multimolecule/rnafm` model (structural retrieval only) |


