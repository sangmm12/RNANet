#!/usr/bin/env python3
"""Pure synthetic self-check; does not use server data or train any models."""
from pathlib import Path
import sys
import tempfile
import numpy as np

HERE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(HERE/'src'))
from run_core_rotation import (loaded_cyc, proper_rotation, transform_coords,
                               fixed_model_input, CYC_NAMES, rotation_stats,
                               source_path_for_chain, EXPECTED_CYC_SRC, sha256)

def main():
    cyc=loaded_cyc()
    rng=np.random.default_rng(42)
    rr=[]
    for i in range(38):
        v=rng.normal(size=(4,3))*0.15+np.array([i*1.0,i*.15,i*.2])
        rr.append({'coords':v,'c4':v[1].copy(),'b_avg':0.0})
    rr[0]['coords'][0,:]=np.nan
    XX,_=cyc.embedding(rr)
    assert XX.shape==(38,12)
    cen=XX.reshape(-1,4,3).mean((0,1))
    Q=proper_rotation(rng)
    assert np.allclose(Q.T@Q,np.eye(3),atol=1e-14)
    assert np.linalg.det(Q)>0.99999999
    rr2=transform_coords(rr,Q=Q,center=cen)
    rr3=transform_coords(rr,t=np.array([11.25,-7.5,4.75]))
    assert np.isnan(rr2[0]['coords'][0]).all()
    assert np.isnan(rr3[0]['coords'][0]).all()
    assert np.linalg.norm(rr2[1]['coords'][1]-rr2[7]['coords'][1]) == np.linalg.norm(rr[1]['coords'][1]-rr[7]['coords'][1]) or np.allclose(
        np.linalg.norm(rr2[1]['coords'][1]-rr2[7]['coords'][1]),np.linalg.norm(rr[1]['coords'][1]-rr[7]['coords'][1]))
    F=cyc.local_features(XX,[4],4,1e-6)
    Fr=cyc.local_features(cyc.embedding(rr2)[0],[4],4,1e-6)
    Ft=cyc.local_features(cyc.embedding(rr3)[0],[4],4,1e-6)
    for k in ('omega_norm','logdetG','ricci_proxy'):
        assert np.allclose(F[4][k],Fr[4][k],atol=1e-5,rtol=1e-5),k
    for k in F[4]:
        assert np.allclose(F[4][k],Ft[4][k],atol=1e-4,rtol=1e-4),k
    assert len(CYC_NAMES)==40
    fakeX=np.zeros((12,96),np.float32)
    fakeB=np.arange(12,dtype=np.uint8)%4
    z=fixed_model_input(fakeX,fakeB,[2,5],[10,11]+list(range(56,96)))
    assert z.shape==(2,46) and z[0,-4:].tolist()==[0,0,1,0]
    r=rotation_stats(np.array([.2,.8,.5,.3]),np.array([.2,.75,.55,.3]),np.array([0,1,1,0]),np.array(['a','a','b','b']))
    assert r['max_abs_delta_p']>0
    with tempfile.TemporaryDirectory() as tmp:
        t=Path(tmp); actual=t/'source.pdb';actual.write_text('placeholder')
        (t/'8abc_a.pdb').symlink_to(actual)
        p,status=source_path_for_chain(t,'8ABC','a')
        assert status=='OK' and p.name=='8abc_a.pdb'
        assert source_path_for_chain(t,'8ABC','A')[1]=='EXACT_STAGE05A_CHAIN_SYMLINK_MISSING'
    print('[PASS] frozen source hash, SO(3), missing atoms, invariant controls, model column order, exact input symlink')

if __name__=='__main__':main()
