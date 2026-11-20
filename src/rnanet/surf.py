"""SURF57 boundary-surface geometry.

Final contract: MSMS only (probe 1.5 A, density 3.0), no convex-hull rescue,
nearest RNA-heavy-atom vertex ownership, 14 nonredundant fields x 4 stats + n_verts.
"""
from __future__ import annotations
import os, subprocess, tempfile
from typing import Dict, List, Tuple
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from .io import RNAResidue, load_rna, parser_for

FIELDS=("Keff","Keff_neg","Keff_pos","Keff_r050","Keff_r100","grad_Keff","grad_phi","grad_phi_r050","grad_phi_r100","lap_Keff","phi","phi_abs","phi_r050","phi_r100")
VDW={"H":1.20,"C":1.70,"N":1.55,"O":1.52,"S":1.80,"P":1.80}

def robust_clip(a,k=10.0):
    a=np.asarray(a,float); fin=np.isfinite(a)
    if not fin.any(): return a
    med=np.median(a[fin]); mad=max(float(np.median(np.abs(a[fin]-med))),1e-6)
    out=a.copy(); out[fin]=np.clip(out[fin],med-k*mad,med+k*mad); return out

def _cot(u,v):
    den=np.linalg.norm(np.cross(u,v))
    return 0. if den<1e-12 else float(np.clip(np.dot(u,v)/den,-1e3,1e3))

def areas_weights(V,F):
    n=len(V); A=np.zeros(n); W=[{} for _ in range(n)]
    def add(i,j,w):
        if i!=j: W[i][j]=W[i].get(j,0.)+w
    for tri in np.asarray(F,int):
        i,j,k=map(int,tri); vi,vj,vk=V[i],V[j],V[k]
        area=.5*np.linalg.norm(np.cross(vj-vi,vk-vi))
        if not np.isfinite(area) or area<=0: continue
        A[[i,j,k]]+=area/3.
        ca=max(_cot(vj-vi,vk-vi),0.); cb=max(_cot(vk-vj,vi-vj),0.); cg=max(_cot(vi-vk,vj-vk),0.)
        add(j,k,ca); add(k,j,ca); add(k,i,cb); add(i,k,cb); add(i,j,cg); add(j,i,cg)
    A[A<1e-8]=1e-8; return A,W

def lap(f,A,W):
    out=np.zeros(len(f))
    for i in range(len(f)): out[i]=sum(w*(f[j]-f[i]) for j,w in W[i].items())/(2*A[i])
    return out

def grad(f,A,W):
    out=np.zeros(len(f))
    for i in range(len(f)): out[i]=np.sqrt(max(sum(w*(f[j]-f[i])**2 for j,w in W[i].items())/(2*A[i]),0.))
    return out

def sphere(V):
    X=V-V.mean(0); n=np.linalg.norm(X,axis=1); n[n<1e-8]=1e-8; return X/n[:,None]

def smooth(S,f,R):
    tree=cKDTree(S); ids=tree.query_ball_point(S,r=R); return np.array([np.mean(f[j]) if j else f[i] for i,j in enumerate(ids)])

def surface_fields(V,F) -> Dict[str,np.ndarray]:
    Ao,_=areas_weights(V,F); S=sphere(V); As,W=areas_weights(S,F)
    lam2=np.clip(Ao/As,1e-8,None); phi=.5*np.log(lam2); Keff=-lap(phi,As,W)
    gp=grad(phi,As,W); gk=grad(Keff,As,W); lk=lap(Keff,As,W)
    d={"Keff":Keff,"Keff_neg":np.where(Keff<0,Keff,np.nan),"Keff_pos":np.where(Keff>0,Keff,np.nan),
       "Keff_r050":smooth(S,Keff,.05),"Keff_r100":smooth(S,Keff,.10),"grad_Keff":gk,"grad_phi":gp,
       "grad_phi_r050":smooth(S,gp,.05),"grad_phi_r100":smooth(S,gp,.10),"lap_Keff":lk,
       "phi":phi,"phi_abs":np.abs(phi),"phi_r050":smooth(S,phi,.05),"phi_r100":smooth(S,phi,.10)}
    return {k:robust_clip(d[k]) for k in FIELDS}

def _atoms_and_owners(path,chain="ALL"):
    st=parser_for(path).get_structure(os.path.basename(path),path); model=next(st.get_models())
    residues=load_rna(path,chain); key={(r.chain,r.resseq,r.icode):i for i,r in enumerate(residues)}
    coords=[]; elems=[]; owners=[]
    for ch in model:
        if chain!="ALL" and ch.id!=chain: continue
        for res in ch:
            rk=(ch.id,int(res.id[1]),str(res.id[2]).strip())
            if rk not in key: continue
            for a in res.get_atoms():
                e=(a.element or a.get_name()[0]).strip().upper()
                if e=="H": continue
                coords.append(np.asarray(a.coord,float)); elems.append(e); owners.append(key[rk])
    return residues,np.array(coords),elems,np.array(owners,int)

def run_msms(path,chain,msms_bin,probe=1.5,density=3.0):
    residues,C,E,O=_atoms_and_owners(path,chain)
    if not os.path.isfile(msms_bin): raise FileNotFoundError(msms_bin)
    with tempfile.TemporaryDirectory(prefix='rnanet_msms_') as td:
        xyzr=os.path.join(td,'input.xyzr'); pref=os.path.join(td,'surf')
        with open(xyzr,'w') as f:
            for c,e in zip(C,E): f.write(f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {VDW.get(e,1.7):.3f}\n")
        cmd=[msms_bin,'-if',xyzr,'-of',pref,'-probe_radius',str(probe),'-density',str(density)]
        subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        V=[]
        with open(pref+'.vert') as f:
            for line in list(f)[3:]:
                p=line.split();
                if len(p)>=3: V.append(tuple(map(float,p[:3])))
        F=[]
        with open(pref+'.face') as f:
            for line in list(f)[3:]:
                p=line.split();
                if len(p)>=3: F.append(tuple(int(x)-1 for x in p[:3]))
        V=np.asarray(V,float); F=np.asarray(F,int)
    # Final RNA ownership is nearest heavy atom, then atom -> nucleotide.
    _,nn=cKDTree(C).query(V,k=1); vertex_owner=O[nn]
    return residues,V,F,vertex_owner

def aggregate(residues,V,F,owner):
    fields=surface_fields(V,F); rows=[]
    for i,r in enumerate(residues):
        ids=np.where(owner==i)[0]; row={"ResidueIndex":r.index,"Chain":r.chain,"ResSeq":r.resseq,"ICode":r.icode,"ResName":r.resname,"n_verts":int(len(ids))}
        for k in FIELDS:
            x=fields[k][ids]; x=x[np.isfinite(x)]
            for stat in ('mean','std','min','max'):
                row[f'surf_{k}_{stat}']=float(getattr(np,stat)(x)) if len(x) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def extract(path,chain,msms_bin):
    residues,V,F,owner=run_msms(path,chain,msms_bin,1.5,3.0)
    df=aggregate(residues,V,F,owner)
    # 14*4 + n_verts = 57 surface variables; ID columns are metadata.
    return df
