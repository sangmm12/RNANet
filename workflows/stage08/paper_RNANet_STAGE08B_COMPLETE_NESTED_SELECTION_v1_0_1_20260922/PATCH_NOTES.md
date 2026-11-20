# v1.0.1 patch notes

Root cause of v1.0.0 failure:

```python
train_pdbs=np.unique(pdb[tr0])
pdb2fold={... for pp in train_pdbs}
foldcode=np.array([pdb2fold[int(pp)] for pp in pdb])
```

`pdb2fold` intentionally contained only outer-train PDBs, but the list comprehension traversed all SURF rows, including outer validation/test PDBs. This produced `KeyError: 3` before any model fitting.

Fix:

- initialize all rows to `-1`;
- assign inner-fold codes only on `tr0`;
- assert outer validation/test remain `-1`.

Additional methodological hardening:

- inner early stopping is now separated from inner scoring at the PDB level;
- candidate AUPRC is measured only on the untouched inner-score fold;
- HNM ranking is restricted to fit/train rows;
- per-fold checkpoint/resume support added.
