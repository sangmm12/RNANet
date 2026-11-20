#!/usr/bin/env python3
"""Synthetic smoke test only: verifies wiring, never substitutes for real RNA data."""
import os,sys,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'r102_frozen_core'/'src'))
import frozen_step1_cyc_RNA_chainlocal as cyc
from rotation_probe import feature_vector,rand_rotation,rot_X,COLS
from ablation_retrieval import calculate
from summarize_challenge import pairkey

def test_rotation_functions():
    rng=np.random.default_rng(20261112)
    rr=[]
    for i in range(50):
        # Positive probability of missing atoms is not needed for this simple smoke test.
        origin=np.array([i*0.7,np.sin(i/3),np.cos(i/2)])
        xyz=origin+rng.normal(0,.2,size=(4,3))
        rr.append({'coords':xyz,'c4':xyz[1],'b_avg':25.0})
    x,_=cyc.embedding(rr)
    base=feature_vector(rr,x)
    assert len(base)==33 and np.isfinite(base).all()
    R=rand_rotation(rng)
    xr=rot_X(x,R)
    center=x.reshape(-1,4,3).mean(axis=(0,1))
    transformed=[]
    for r in rr:
        t=dict(r);t['c4']=(r['c4']-center)@R.T+center;transformed.append(t)
    y=feature_vector(transformed,xr)
    assert np.isclose(base[0],y[0]) and np.isclose(base[1],y[1])
    for k in (4,12,20,28,36,44):
        for suffix in ('omega_norm','logdetG','ricci_proxy'):
            col=f'cyc_{suffix}_k{k}'
            j=COLS.index(col)
            assert np.isclose(base[j],y[j],rtol=1e-4,atol=1e-4),(col,base[j],y[j])
    print('[SELFTEST PASS] Source CY-C function, proper 3D rotation, invariant descriptors')

def test_search_wiring():
    rng=np.random.default_rng(20261112)
    with tempfile.TemporaryDirectory() as temp:
        out=Path(temp);n=1406
        m={'Chain_ID':[f'{i:04X}_A' for i in range(n)]}
        for col in COLS:m[col]=rng.normal(size=n)
        for col in [f'fm_{i}' for i in range(640)]:m[col]=rng.normal(size=n)
        master=out/'master_C.csv';pd.DataFrame(m).to_csv(master,index=False)
        alphabet=np.array(list('AUCG'))
        seqs={k:''.join(rng.choice(alphabet,size=26)) for k in m['Chain_ID']}
        res=calculate(master,seqs,out/'retrieval',knn=20,percent=20,top=20)
        assert res['n_chains']==1406 and len(res['mode_summaries'])==3
        for name in ['C_ONLY','FM_ONLY','C_FM']:
            cand=pd.read_csv(out/'retrieval'/f'candidates_{name}.csv')
            assert len(cand)==20 and (cand.SeqSim<=0.35).all()
            assert (cand.PDB_A!=cand.PDB_B).all()
        print('[SELFTEST PASS] Three-arm kNN retrieval, strict Jaccard, PDB-pair dedup')

if __name__=='__main__':
    test_rotation_functions()
    test_search_wiring()
