# RNANet NC Minimal Two-Gate R1: independent audit and R2 correction

## Inputs and integrity

- Uploaded archive `RNANET_NC_MIN2GATE_REVIEW_20261112_NC_MIN2GATE_R1.tar.gz`, SHA256 `5bd31fb8816a2c8c72b75b25286d3107b0f2f90d25424af4090d4176ec1c6c8b` agrees with supplied digest.
- Run status in `FINAL_GATE.json`: `HOLD_FOR_SCIENTIFIC_INTERPRETATION`.
- 247 union pairs structurally aligned, 100 ranked candidates for each of C_ONLY, FM_ONLY, C_FM.

## Key confound found in independent audit

For C_ONLY: 98/100 Top100 candidates are exactly 3 nt on both chains; remaining 2 are exactly 4 nt. Its Top100 median TM_min of 0.43639, including 54/100 with TM_min >=0.42 and coverage >=0.5, measures alignment of extremely short oligomers rather than substantive RNA tertiary topology. Do NOT present this as evidence that CY-C-only outperforms RNA-FM. FM_ONLY Top100 minimum pair length median = 20.5 nt; C_FM Top100 median = 19 nt. Different length distributions prevent causal modality comparison.

## Rotation diagnostics

The original 12 chain x 3 rotations x 33 dimensions probe found 431/1188 instabilities, restricted to the 6 integrability_err and 6 kahler_err columns. The 3 base descriptors, 6 omega_norm, 6 logdetG, and 6 ricci_proxy columns remained within test tolerances. Max absolute delta 0.2308194. However 9/12 baseline vectors did not match the frozen C master, each at one scalar column. The Stage02 aggregator takes ContactCount/Order/B_avg from CY-A and its B_avg averages P,O5',C5',C4',C3',O3'; the CY-C source re-extractor uses P,C4',C3',O3'. This difference can explain the one-column mismatch and is explicitly checked (not assumed) in R2.

## R2 frozen scope

- Correct the *source anchor* by recomputing the CY-A-derived three base descriptors using original Stage02 CY-A code and all 30 CY-C descriptors using exact Stage02 CY-C code. Anchor all 33 numbers to the frozen master before interpreting rotation.
- Exclude chains with sequence length below 20 nt in **all three** retrieval arms before kNN. No selective or outcome-dependent sample dropping; no new models. Keep 67/33 geometry weight, 640D RNA-FM block, kNN100, p20, Jaccard <=0.35, topN100 and common TopK structural US-align.
- If structural export reveals any chain shorter than 20 nucleotides, report HOLD and inspect; do not suppress.
- R2 independent output is a controlled descriptive sensitivity analysis; it is NOT the exact Stage04B frozen retrieval (which used additional Geo and FM distance gates and matched nulls).
- Do not publish a new scientific claim until R2 results and coordinate-frame dependence are examined.

## Manuscript boundaries, pending R2

If rotation dependence is verified, describe CY-C as a *coordinate-space* (not intrinsically rotation-invariant) representation, explicitly disclose the fixed J12 gauge dependence in Methods/Limitations, and check the effect on model outputs before making claims of invariance or intrinsic geometry. This is a scientific qualification of the original method, not evidence of new RNA biology.

Do not modify the historical Stage04B source files or candidate lists. Do not infer matched-null significance for new R2 arms, which has not been computed.
