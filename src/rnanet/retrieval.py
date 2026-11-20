"""Retrieval helpers used by the frozen search/audit contract."""
from __future__ import annotations
import numpy as np

def kmer_set(seq: str, k: int = 3):
    s=''.join(str(seq).split()).upper()
    return {s[i:i+k] for i in range(max(len(s)-k+1,0))}

def kmer_jaccard(seq_a: str, seq_b: str, k: int = 3) -> float:
    a,b=kmer_set(seq_a,k),kmer_set(seq_b,k)
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return len(a&b)/len(a|b)

def kmer_divergence(seq_a: str, seq_b: str, k: int = 3) -> float:
    return 1.0-kmer_jaccard(seq_a,seq_b,k)

def knn_euclidean(X, k=100):
    X=np.asarray(X,float); d2=np.sum((X[:,None,:]-X[None,:,:])**2,axis=2); np.fill_diagonal(d2,np.inf)
    ids=np.argsort(d2,axis=1,kind='mergesort')[:,:min(k,len(X)-1)]
    dist=np.sqrt(np.take_along_axis(d2,ids,axis=1)); return ids,dist
