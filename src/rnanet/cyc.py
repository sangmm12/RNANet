"""CY-C RNA coordinate-space geometry; B-factor-free public implementation."""
from __future__ import annotations
from typing import Dict, List
import numpy as np
from .io import RNAResidue, atom

SCALES=(4,8,12,16,20,24,28,32)
ATOM_ORDER=("P","C4'","C3'","O3'")

def standard_J(dim=12):
    J=np.zeros((dim,dim),float)
    for i in range(0,dim,2): J[i,i+1]=-1.; J[i+1,i]=1.
    return J

def embedding(res: List[RNAResidue], impute=True):
    X=np.full((len(res),12),np.nan,float)
    for i,r in enumerate(res):
        for j,a in enumerate(ATOM_ORDER):
            c=atom(r,a)
            if c is not None: X[i,3*j:3*j+3]=c
    if impute:
        with np.errstate(all='ignore'): mu=np.nanmean(X,axis=0)
        if np.any(~np.isfinite(mu)): raise ValueError("coordinate column entirely missing")
        bad=~np.isfinite(X); X[bad]=np.take(mu,np.where(bad)[1])
    return X

def _neighbors(X,k):
    d2=np.sum((X[:,None,:]-X[None,:,:])**2,axis=2); np.fill_diagonal(d2,np.inf)
    return np.argsort(d2,axis=1,kind="mergesort")[:,:min(k,len(X)-1)]

def _tangent(Y,tan_dim=4):
    Yc=Y-Y.mean(0,keepdims=True); _,s,vt=np.linalg.svd(Yc,full_matrices=False)
    rank=int(np.sum(s>1e-12)); t=min(tan_dim,rank,Y.shape[1])
    return None if t==0 else vt[:t].T

def compute_cyc(X, scales=SCALES, tan_dim=4, eps=1e-6) -> Dict[int,Dict[str,np.ndarray]]:
    X=np.asarray(X,float); n,d=X.shape
    if d!=12 or n<2: raise ValueError("CY-C expects n>=2 and 12D RNA coordinate embedding")
    J=standard_J(d); out={}
    for k in scales:
        nb=_neighbors(X,min(k,n-1)); vals={z:np.full(n,np.nan) for z in ("integrability_err","kahler_err","omega_norm","logdetG","ricci_proxy")}
        for i in range(n):
            Y=X[nb[i]]; Yc=Y-Y.mean(0,keepdims=True); m=len(Y)
            G=(Yc.T@Yc)/max(m-1,1)+eps*np.eye(d)
            T=_tangent(Y,tan_dim)
            if T is not None:
                P=T@np.linalg.pinv(T.T@T)@T.T; JT=J@T; den=np.linalg.norm(JT,'fro')
                vals['integrability_err'][i]=np.linalg.norm((np.eye(d)-P)@JT,'fro')/den if den else np.nan
            vals['kahler_err'][i]=np.linalg.norm(J.T@G@J-G,'fro')/np.linalg.norm(G,'fro')
            vals['omega_norm'][i]=np.linalg.norm(G@J,'fro')
            sign,ld=np.linalg.slogdet(G); vals['logdetG'][i]=ld if sign>0 else np.linalg.slogdet(G+eps*np.eye(d))[1]
        for i in range(n):
            x=vals['logdetG'][nb[i]]; x=x[np.isfinite(x)]
            if np.isfinite(vals['logdetG'][i]) and len(x): vals['ricci_proxy'][i]=abs(vals['logdetG'][i]-x.mean())
        out[k]=vals
    return out

def contact_features(res: List[RNAResidue], cutoff=10.0):
    pts=np.array([atom(r,"C4'") if atom(r,"C4'") is not None else [np.nan]*3 for r in res],float)
    n=len(res); cc=np.zeros(n); co=np.zeros(n)
    for i in range(n):
        if not np.all(np.isfinite(pts[i])): cc[i]=co[i]=np.nan; continue
        d=np.linalg.norm(pts-pts[i],axis=1); ids=np.where((d<=cutoff)&(np.arange(n)!=i)&np.isfinite(d))[0]
        cc[i]=len(ids); co[i]=np.mean(np.abs(ids-i)) if len(ids) else 0.
    return cc,co
