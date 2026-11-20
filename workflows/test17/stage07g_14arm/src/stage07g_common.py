from __future__ import annotations
import hashlib, json, os
from pathlib import Path
import numpy as np

SEEDS=[3407,3420,3433,3446,3459]
REPS=['A','C','S','AC','AS','CS','ACS']
ARMS=[x for r in REPS for x in (r,r+'+HNM')]
DIMS={'A':50,'C':46,'S':61,'AC':90,'AS':107,'CS':103,'ACS':147}
HNM_TOPK=20
HNM_WEIGHT=5.0


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def dumpj(path: Path, obj) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')


def hard_negative_mask(y, pred, chain, topk=HNM_TOPK):
    y=np.asarray(y); pred=np.asarray(pred,float); chain=np.asarray(chain).astype(str)
    out=np.zeros(len(y),dtype=bool)
    neg=np.flatnonzero(y==0)
    if not len(neg): return out
    # stable group order; ties resolved by original local row index
    order=np.argsort(chain[neg],kind='mergesort')
    neg=neg[order]; cneg=chain[neg]
    starts=np.r_[0,np.flatnonzero(cneg[1:]!=cneg[:-1])+1]
    ends=np.r_[starts[1:],len(neg)]
    for a,b in zip(starts,ends):
        idx=neg[a:b]; k=min(int(topk),len(idx))
        oo=np.lexsort((idx,-pred[idx]))[:k]
        out[idx[oo]]=True
    return out


def expected_feature_dimensions():
    return dict(DIMS)
