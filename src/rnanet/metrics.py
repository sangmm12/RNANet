from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def binary_metrics(y_true,y_score):
    y=np.asarray(y_true); s=np.asarray(y_score)
    return {"auroc":float(roc_auc_score(y,s)),"auprc":float(average_precision_score(y,s))}

def hit_at_k(chain, y_true, score, ks=(1,5,10,20,50,100)):
    chain=np.asarray(chain); y=np.asarray(y_true); s=np.asarray(score); out={}
    pos_ch=[c for c in np.unique(chain) if np.any(y[chain==c]>0)]
    for k in ks:
        hits=[]
        for c in pos_ch:
            ids=np.where(chain==c)[0]; order=ids[np.argsort(-s[ids],kind='mergesort')[:k]]
            hits.append(float(np.any(y[order]>0)))
        out[int(k)]=float(np.mean(hits)) if hits else np.nan
    return out
