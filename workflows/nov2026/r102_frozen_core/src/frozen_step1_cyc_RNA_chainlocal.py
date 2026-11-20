#!/usr/bin/env python3
"""Chain-local RNA CY-C multiscale coordinate-space descriptors.

The coordinate representation uses P, C4', C3' and O3' atoms (12 Cartesian
coordinates per nucleotide). All neighbor geometry and contact-order calculations
are restricted to one RNA chain at a time. This prevents unrelated chains in the
same deposited assembly from changing an intrinsic chain descriptor.
"""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser, PDBParser
from sklearn.neighbors import NearestNeighbors

ATOM_ORDER = ["P", "C4'", "C3'", "O3'"]
CORE = ["integrability_err", "kahler_err", "omega_norm", "logdetG", "ricci_proxy"]


def canonical_chain_key(value):
    """Upper-case the PDB accession only; preserve chain/asym-id case exactly."""
    text = str(value).strip()
    if "_" not in text:
        return text
    pdb_id, chain_id = text.split("_", 1)
    return f"{pdb_id.upper()}_{chain_id}"


def pick_atom(res, name):
    hits = [a for a in res.get_atoms() if a.get_id().strip() == name]
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]
    return max(hits, key=lambda a: 0.0 if a.get_occupancy() is None else float(a.get_occupancy()))


def parse_structure(path):
    ext = os.path.splitext(path)[1].lower()
    parser = MMCIFParser(QUIET=True) if ext in (".cif", ".mmcif") else PDBParser(QUIET=True)
    return parser.get_structure(os.path.basename(path), path)


def extract_chains(st, chain_mode="ALL") -> List[List[Dict]]:
    model = next(st.get_models())
    mode = str(chain_mode)
    mode_up = mode.upper()
    out = []
    for ch in model:
        if mode_up not in ("ALL", "LONGEST") and ch.id != mode:
            continue
        rr = []
        for res in ch:
            c4 = pick_atom(res, "C4'")
            if c4 is None:
                continue
            coords = np.full((4,3), np.nan, float)
            bf = []
            for i, an in enumerate(ATOM_ORDER):
                a = pick_atom(res, an)
                if a is None:
                    continue
                coords[i] = np.asarray(a.get_coord(), float)
                try:
                    bf.append(float(a.get_bfactor()))
                except Exception:
                    pass
            rr.append({
                "chain": str(ch.id), "resseq": int(res.id[1]),
                "resname": str(res.get_resname()).strip(),
                "coords": coords, "c4": np.asarray(c4.get_coord(), float),
                "b_avg": float(np.mean(bf)) if bf else np.nan,
            })
        if rr:
            out.append(rr)
    if mode_up == "LONGEST" and out:
        out = [max(out, key=len)]
    return out


def embedding(rr):
    X = np.asarray([r["coords"].reshape(-1) for r in rr], float)
    B = np.asarray([r["b_avg"] for r in rr], float)
    if X.size:
        cm = np.nanmean(X, axis=0)
        cm = np.where(np.isfinite(cm), cm, 0.0)
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(cm, np.where(bad)[1])
    bm = float(np.nanmean(B)) if np.isfinite(B).any() else 0.0
    B = np.where(np.isfinite(B), B, bm)
    return X, B


def contacts(rr, cutoff):
    n = len(rr)
    if n == 0:
        return np.array([]), np.array([])
    pts = np.asarray([r["c4"] for r in rr], float)
    nbr = NearestNeighbors(radius=float(cutoff), algorithm="ball_tree").fit(pts)
    inds = nbr.radius_neighbors(pts, return_distance=False)
    cc = np.zeros(n, float); co = np.zeros(n, float)
    for i, js in enumerate(inds):
        js = [int(j) for j in js if int(j) != i]
        cc[i] = len(js)
        co[i] = float(np.mean([abs(j-i) for j in js])) if js else 0.0
    return cc, co


def standard_J(dim):
    if dim % 2:
        raise ValueError("CY-C embedding dimension must be even")
    J = np.zeros((dim,dim), float)
    for k in range(dim//2):
        a,b=2*k,2*k+1
        J[a,b] = -1.0; J[b,a] = 1.0
    return J


def local_pca_tangent(Y, tan_dim):
    if len(Y) == 0:
        return None
    Yc = Y - Y.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    rank = int(np.sum(S > 1e-12))
    t = min(int(tan_dim), rank, Y.shape[1])
    return None if t == 0 else Vt[:t].T


def local_features(X, ks, tan_dim=4, eps=1e-6):
    n,d = X.shape
    ks = sorted(set(int(k) for k in ks if int(k)>0))
    out = {k:{f:np.zeros(n,float) for f in CORE} for k in ks}
    if n <= 1:
        return out
    K = min(max(ks), n-1)
    nn = NearestNeighbors(n_neighbors=K+1, algorithm="auto").fit(X)
    _, all_idx = nn.kneighbors(X)
    all_idx = all_idx[:,1:]
    J = standard_J(d)
    eye = np.eye(d)
    for k in ks:
        ke = min(k, n-1)
        nb = all_idx[:,:ke]
        integ=np.full(n,np.nan); kah=np.full(n,np.nan); omg=np.full(n,np.nan); logd=np.full(n,np.nan)
        for i in range(n):
            Y=X[nb[i]]
            Yc=Y-Y.mean(axis=0,keepdims=True)
            m=len(Yc)
            if m < 2:
                G=eye.copy()
            else:
                G=(Yc.T@Yc)/float(m-1)+float(eps)*eye
            T=local_pca_tangent(Y,int(tan_dim))
            if T is not None:
                JT=J@T
                M=T.T@T
                P=T@np.linalg.pinv(M)@T.T
                R=(eye-P)@JT
                den=np.linalg.norm(JT,"fro")
                integ[i]=np.linalg.norm(R,"fro")/den if den>0 else np.nan
            JGJ=J.T@G@J
            den=np.linalg.norm(G,"fro")
            kah[i]=np.linalg.norm(JGJ-G,"fro")/den if den>0 else np.nan
            omg[i]=np.linalg.norm(G@J,"fro")
            sign,v=np.linalg.slogdet(G)
            if sign <= 0:
                sign,v=np.linalg.slogdet(G+float(eps)*eye)
            logd[i]=v
        ric=np.full(n,np.nan)
        for i in range(n):
            vals=logd[nb[i]]
            vals=vals[np.isfinite(vals)]
            if np.isfinite(logd[i]) and vals.size:
                ric[i]=abs(logd[i]-vals.mean())
        vals = {"integrability_err":integ,"kahler_err":kah,"omega_norm":omg,"logdetG":logd,"ricci_proxy":ric}
        for name,a in vals.items():
            a=np.asarray(a,float)
            good=np.isfinite(a)
            fill=float(np.nanmedian(a[good])) if good.any() else 0.0
            out[k][name]=np.where(good,a,fill)
    return out


def load_allowed(path: Optional[str]) -> Optional[Set[str]]:
    if not path:
        return None
    d=pd.read_csv(path)
    if "is_strict_rna" in d.columns:
        d=d[pd.to_numeric(d["is_strict_rna"],errors="coerce").fillna(0).astype(int)==1]
    return set(d["chain_key"].map(canonical_chain_key))


def process_one(path, out_dir, ks, tan_dim, eps, cutoff, chain_mode, allowed):
    sid=os.path.splitext(os.path.basename(path))[0]
    out_csv=os.path.join(out_dir,f"{sid}.csv")
    try:
        st=parse_structure(path)
        chains=extract_chains(st,chain_mode)
        frames=[]; stats=[]
        for rr in chains:
            ch=rr[0]["chain"]; ck=canonical_chain_key(f"{sid}_{ch}")
            if allowed is not None and ck not in allowed:
                continue
            X,B=embedding(rr); cc,co=contacts(rr,cutoff); ff=local_features(X,ks,tan_dim,eps)
            data={
                "ResidueIndex":np.arange(1,len(rr)+1), "Chain":[ch]*len(rr),
                "ResSeq":[r["resseq"] for r in rr], "ResName":[r["resname"] for r in rr],
                "B_avg":B, "ContactCount":cc, "ContactOrder":co,
            }
            for k in ks:
                for name in CORE:
                    data[f"{name}_k{k}"]=ff[k][name]
            frames.append(pd.DataFrame(data)); stats.append({"chain_key":ck,"n_res":len(rr)})
        os.makedirs(out_dir,exist_ok=True)
        if frames:
            out=pd.concat(frames,ignore_index=True)
        else:
            cols=["ResidueIndex","Chain","ResSeq","ResName","B_avg","ContactCount","ContactOrder"]
            for k in ks:
                cols.extend([f"{name}_k{k}" for name in CORE])
            out=pd.DataFrame(columns=cols)
        out.to_csv(out_csv,index=False)
        return path,True,("empty" if out.empty else f"OK n_res={len(out)}"),stats
    except Exception as e:
        return path,False,repr(e),[]


def main():
    ap=argparse.ArgumentParser(description="Chain-local RNA CY-C")
    ap.add_argument("--input",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--ks",default="4,12,20,28,36,44"); ap.add_argument("--chain",default="ALL")
    ap.add_argument("--tan-dim",type=int,default=4); ap.add_argument("--eps",type=float,default=1e-6)
    ap.add_argument("--contacts-cutoff",type=float,default=10.0); ap.add_argument("--jobs",type=int,default=1)
    ap.add_argument("--strict-manifest",default=None); ap.add_argument("--stats-out",default=None)
    args=ap.parse_args()
    ks=sorted(set(int(x) for x in args.ks.split(",") if x.strip()))
    paths=[]
    for ext in ("*.cif","*.mmcif","*.pdb"):
        paths.extend(glob.glob(os.path.join(args.input,"**",ext),recursive=True))
    paths=sorted(set(paths)); allowed=load_allowed(args.strict_manifest)
    if not paths: raise SystemExit(f"[FATAL] no structures in {args.input}")
    os.makedirs(args.output,exist_ok=True)
    kw=dict(out_dir=args.output,ks=ks,tan_dim=args.tan_dim,eps=args.eps,cutoff=args.contacts_cutoff,chain_mode=args.chain,allowed=allowed)
    res=[]
    if args.jobs<=1:
        res=[process_one(p,**kw) for p in paths]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs={ex.submit(process_one,p,**kw):p for p in paths}
            for f in as_completed(futs): res.append(f.result())
    res.sort(key=lambda x:x[0]); stats=[]; ok=0
    for p,s,msg,st in res:
        print(f"[{'OK' if s else 'ERR'}] {os.path.basename(p)}: {msg}"); ok+=int(s); stats.extend(st)
    so=args.stats_out or os.path.join(args.output,"chain_stats_cyc.csv")
    pd.DataFrame(stats).to_csv(so,index=False)
    print(f"[DONE] files={len(paths)} ok={ok} chains={len(stats)} out={args.output} stats={so}")
    if ok!=len(paths): raise SystemExit(2)

if __name__=="__main__": main()
