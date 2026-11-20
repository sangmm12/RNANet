#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from rnanet.cya import pseudorotation_phase,standard_J as JA,a5_features
from rnanet.cyc import standard_J as JC,compute_cyc
from rnanet.surf import surface_fields,FIELDS

def circdiff(a,b): return abs(np.angle(np.exp(1j*(a-b))))
def main():
    P=.73; tau=.42; m=np.arange(5); nu=tau*np.cos(P+4*np.pi*m/5); ph,amp=pseudorotation_phase(nu)
    assert circdiff(ph,P)<1e-10 and abs(amp-tau)<1e-10
    assert np.allclose(JA(8)@JA(8),-np.eye(8)); assert np.allclose(JC(12)@JC(12),-np.eye(12))
    rng=np.random.default_rng(7); Q=rng.uniform(-np.pi,np.pi,(40,4)); Y=np.stack([np.cos(Q[:,0]),np.sin(Q[:,0]),np.cos(Q[:,1]),np.sin(Q[:,1]),np.cos(Q[:,2]),np.sin(Q[:,2]),np.cos(Q[:,3]),np.sin(Q[:,3])],1)
    assert set(a5_features(Y).keys())=={4,8,12,16,20,24,28,32}
    X=rng.normal(size=(40,12)); assert set(compute_cyc(X).keys())=={4,8,12,16,20,24,28,32}
    V=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],float); F=np.array([[0,2,4],[2,1,4],[1,3,4],[3,0,4],[2,0,5],[1,2,5],[3,1,5],[0,3,5]])
    assert set(surface_fields(V,F))==set(FIELDS) and len(FIELDS)==14
    print('[PASS] core geometry selftest')
if __name__=='__main__': main()
