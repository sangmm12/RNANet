#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
python -m py_compile "$HERE/01_train_cyc_surf_hnm_test17.py"
python - <<'PY'
import numpy as np, pandas as pd

def sel(df,score,topk):
    z=np.zeros(len(df),bool); w=df[['chain_key','label','ResidueIndex']].copy().reset_index(drop=True); w['score']=score
    for _,g in w.groupby('chain_key',sort=True):
        n=g[g.label.eq(0)].sort_values(['score','ResidueIndex'],ascending=[False,True],kind='mergesort')
        z[n.head(topk).index.to_numpy()]=True
    return z

d=pd.DataFrame({'chain_key':['A']*4+['B']*4,'label':[0,0,1,0,0,1,0,0],'ResidueIndex':[1,2,3,4,1,2,3,4]})
s=np.array([.2,.9,.8,.4,.1,.8,.7,.6])
h=sel(d,s,2)
assert h.sum()==4
assert not d.loc[h,'label'].astype(bool).any()
assert set(d.loc[h & d.chain_key.eq('A'),'ResidueIndex'])=={2,4}
assert set(d.loc[h & d.chain_key.eq('B'),'ResidueIndex'])=={3,4}
print('[PASS] Stage07F deterministic HNM selector selftest')
PY
echo '[PASS] Stage07F script compile/selftest'
