"""CY-A RNA conformational geometry.

Reader-facing definition: eta, theta, glycosidic chi, ribose pseudorotation P,
plus A5 descriptors at k={4,8,12,16,20,24,28,32}. No B-factor predictors.
"""
from __future__ import annotations
import itertools, math
from typing import Dict, Iterable, List, Tuple
import numpy as np
from .io import RNAResidue, atom

SCALES=(4,8,12,16,20,24,28,32)

def dihedral(a,b,c,d):
    if any(x is None for x in (a,b,c,d)): return np.nan
    b0=np.asarray(a)-np.asarray(b); b1=np.asarray(c)-np.asarray(b); b2=np.asarray(d)-np.asarray(c)
    n=np.linalg.norm(b1)
    if n < 1e-12: return np.nan
    b1=b1/n
    v=b0-np.dot(b0,b1)*b1; w=b2-np.dot(b2,b1)*b1
    return float(np.arctan2(np.dot(np.cross(b1,v),w), np.dot(v,w)))

def glycosidic_chi(r: RNAResidue) -> float:
    # Purines use N9/C4; pyrimidines use N1/C2.
    if atom(r,"N9") is not None and atom(r,"C4") is not None:
        return dihedral(atom(r,"O4'"),atom(r,"C1'"),atom(r,"N9"),atom(r,"C4"))
    return dihedral(atom(r,"O4'"),atom(r,"C1'"),atom(r,"N1"),atom(r,"C2"))

def ribose_torsions(r: RNAResidue) -> np.ndarray:
    A=r.atoms
    quads=[("O4'","C1'","C2'","C3'"),("C1'","C2'","C3'","C4'"),
           ("C2'","C3'","C4'","O4'"),("C3'","C4'","O4'","C1'"),
           ("C4'","O4'","C1'","C2'")]
    return np.array([dihedral(A.get(a),A.get(b),A.get(c),A.get(d)) for a,b,c,d in quads],float)

def pseudorotation_phase(nu: Iterable[float]) -> Tuple[float,float]:
    """Least-squares phase convention from manuscript Eqs. S5-S6.

    nu_m ~= tau*cos(P + 4*pi*m/5). Returns (P,tau), P in (-pi,pi].
    """
    y=np.asarray(list(nu),float)
    if y.shape != (5,) or not np.all(np.isfinite(y)): return np.nan,np.nan
    phi=4*np.pi*np.arange(5)/5.0
    X=np.c_[np.cos(phi), np.sin(phi)]
    a,b=np.linalg.lstsq(X,y,rcond=None)[0]  # a=tau*cosP, b=-tau*sinP
    P=float(np.arctan2(-b,a)); tau=float(np.hypot(a,b))
    return P,tau

def circular_coordinates(res: List[RNAResidue]):
    n=len(res); Q=np.full((n,4),np.nan,float)
    for i,r in enumerate(res):
        prev=res[i-1] if i>0 and res[i-1].chain==r.chain else None
        nxt=res[i+1] if i+1<n and res[i+1].chain==r.chain else None
        Q[i,0]=dihedral(atom(prev,"C4'") if prev else None,atom(r,"P"),atom(r,"C4'"),atom(nxt,"P") if nxt else None)
        Q[i,1]=dihedral(atom(r,"P"),atom(r,"C4'"),atom(nxt,"P") if nxt else None,atom(nxt,"C4'") if nxt else None)
        Q[i,2]=glycosidic_chi(r)
        Q[i,3]=pseudorotation_phase(ribose_torsions(r))[0]
    return Q

def torus_embed(Q):
    Q=np.asarray(Q,float)
    return np.stack([np.cos(Q[:,0]),np.sin(Q[:,0]),np.cos(Q[:,1]),np.sin(Q[:,1]),
                     np.cos(Q[:,2]),np.sin(Q[:,2]),np.cos(Q[:,3]),np.sin(Q[:,3])],axis=1)

def standard_J(dim: int):
    if dim%2: raise ValueError("even dimension required")
    J=np.zeros((dim,dim),float)
    for i in range(0,dim,2): J[i,i+1]=-1.; J[i+1,i]=1.
    return J

def _pca(Y):
    Y=np.asarray(Y,float); Yc=Y-Y.mean(0,keepdims=True)
    _,s,vt=np.linalg.svd(Yc,full_matrices=False)
    good=s>1e-12; U=vt[good]
    lam=(s[good]**2/max(len(Y)-1,1)).astype(float)
    return U,lam

def _z(u):
    u=np.asarray(u,float)
    return u[0::2]+1j*u[1::2]

def _center_geom(Y, idx, k):
    kstar=min(max(int(k),5),len(Y))
    d2=np.sum((Y-Y[idx])**2,axis=1); ids=np.argsort(d2,kind="mergesort")[:kstar]
    U,lam=_pca(Y[ids]); J=standard_J(Y.shape[1])
    pairs=[abs(float((J@U[a]).dot(U[b]))) for a,b in itertools.combinations(range(len(U)),2)]
    lag_mean=float(np.mean(pairs)) if pairs else np.nan
    lag_p90=float(np.quantile(pairs,.90)) if pairs else np.nan
    hs=[]
    for comb in itertools.combinations(range(len(U)),4):
        M=np.vstack([_z(U[a]) for a in comb]); hs.append(np.linalg.det(M))
    sl_abs=float(np.mean(np.abs(hs))) if hs else np.nan
    q=np.mean([(h/abs(h))**2 for h in hs if abs(h)>1e-15]) if hs else np.nan+0j
    if np.iscomplexobj(q) and np.isfinite(q.real) and abs(q)>0: q=q/abs(q)
    else: q=np.nan+0j
    eff=float((lam.sum()**2)/(np.sum(lam**2))) if lam.size and np.sum(lam**2)>0 else np.nan
    return ids,lag_mean,lag_p90,sl_abs,q,eff

def a5_features(Y, scales=SCALES) -> Dict[int,Dict[str,np.ndarray]]:
    Y=np.asarray(Y,float); n=len(Y); out={}
    for k in scales:
        rows=[_center_geom(Y,i,k) for i in range(n)]
        q=np.array([r[4] for r in rows],complex)
        phase=np.full(n,np.nan,float)
        for i,r in enumerate(rows):
            qq=q[r[0]]; qq=qq[np.isfinite(qq.real)&np.isfinite(qq.imag)]
            phase[i]=1.0-abs(np.mean(qq)) if len(qq) else np.nan
        out[k]={"lag_mean":np.array([r[1] for r in rows]),"lag_p90":np.array([r[2] for r in rows]),
                "sl_abs":np.array([r[3] for r in rows]),"phase_disp":phase,
                "pca_eff_dim":np.array([r[5] for r in rows])}
    return out

def contact_features(res: List[RNAResidue], cutoff=10.0):
    pts=np.array([atom(r,"C4'") if atom(r,"C4'") is not None else [np.nan]*3 for r in res],float)
    n=len(res); cc=np.zeros(n); co=np.zeros(n)
    for i in range(n):
        if not np.all(np.isfinite(pts[i])): cc[i]=co[i]=np.nan; continue
        d=np.linalg.norm(pts-pts[i],axis=1); ids=np.where((d<=cutoff)&(np.arange(n)!=i)&np.isfinite(d))[0]
        cc[i]=len(ids); co[i]=np.mean(np.abs(ids-i)) if len(ids) else 0.
    return cc,co

def compute_cya(res: List[RNAResidue], scales=SCALES):
    Q=circular_coordinates(res)
    valid=np.all(np.isfinite(Q),axis=1)
    # Geometry descriptors are defined on the complete circular-coordinate subset.
    A={k:{name:np.full(len(res),np.nan) for name in ("lag_mean","lag_p90","sl_abs","phase_disp","pca_eff_dim")} for k in scales}
    if valid.sum()>=5:
        f=a5_features(torus_embed(Q[valid]),scales)
        ids=np.where(valid)[0]
        for k in scales:
            for name in A[k]: A[k][name][ids]=f[k][name]
    cc,co=contact_features(res)
    return Q,cc,co,A
