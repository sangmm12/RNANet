#!/usr/bin/env bash
set -euo pipefail
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -m py_compile "$PKG_ROOT"/scripts/*.py "$PKG_ROOT"/../../../src/rnanet/*.py
for f in "$PKG_ROOT"/scripts/*.sh; do bash -n "$f"; done
python - "$PKG_ROOT" <<'PY'
from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np, pandas as pd
root=Path(sys.argv[1])
contract=json.loads((root/'config/stage07_test17_contract.json').read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(root.parents[2]/'src/rnanet/cyc.py') == contract['rnanet']['cyc_source_sha256']
assert sha(root.parents[2]/'src/rnanet/io.py') == contract['rnanet']['io_source_sha256']
assert contract['benchmark']['expected_test_chains']==17
assert contract['benchmark']['expected_test_nucleotides']==583
assert contract['rnanet']['hnm']['top_hard_negatives_per_training_chain']==20
assert contract['rnanet']['hnm']['hard_negative_weight']==5.0
sys.path.insert(0,str(root.parents[2]/'src'))
from rnanet.cyc import compute_cyc,SCALES
rng=np.random.default_rng(1)
X=rng.normal(size=(40,12))
F=compute_cyc(X)
assert tuple(F)==SCALES
assert sum(len(v) for v in F.values())==40
spec=importlib.util.spec_from_file_location('trainmod',root/'scripts/03_train_rnanet_test17.py')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
d=[]
for c in ('A','B'):
    for i in range(30): d.append({'chain_key':c,'label':1 if i in (0,1) else 0,'ResidueIndex':i+1})
df=pd.DataFrame(d)
scores=np.linspace(0,1,len(df))
mask=mod.select_hard_negatives(df,scores,20)
assert mask.sum()==40
assert (df.loc[mask,'label']==0).all()
assert all(df.assign(sel=mask).groupby('chain_key').sel.sum()==20)
print('[PASS] Stage07 package selftest')
PY
