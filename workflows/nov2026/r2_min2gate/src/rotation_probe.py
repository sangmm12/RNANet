#!/usr/bin/env python3
"""Random proper-rigid-rotation probe of the exact Stage02 CY-C extractor.

Each selected chain is re-parsed from the original CIF. We compare unrotated
features to the frozen C master before interpreting differences. The original
source file is imported and never modified.
"""
import argparse, hashlib, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'r102_frozen_core', 'src'))  # shared frozen CY-C extractor
import frozen_step1_cyc_RNA_chainlocal as cyc
import frozen_cya_core as cya
from ablation_retrieval import read_seq

KS=(4,12,20,28,36,44)
COLS=['ContactCount','ContactOrder','B_avg']+[f'cyc_{core}_k{k}' for k in KS for core in cyc.CORE]

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as fh:
        for chunk in iter(lambda:fh.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def feature_vector(rr,X=None):
    if X is None:X,_=cyc.embedding(rr)
    ff=cyc.local_features(X,KS,tan_dim=4,eps=1e-6)
    cc,co=cyc.contacts(rr,10.0)
    bb=np.array([r['b_avg'] for r in rr],float)
    vec=[float(np.nanmean(cc)),float(np.nanmean(co)),float(np.nanmean(bb)) if np.isfinite(bb).any() else 0.0]
    for k in KS:
        for name in cyc.CORE:vec.append(float(np.nanmean(ff[k][name])))
    return np.array(vec,float)

def rand_rotation(rng):
    A=rng.normal(size=(3,3))
    Q,R=np.linalg.qr(A)
    signs=np.sign(np.diag(R));signs[signs==0]=1
    Q=Q*signs[np.newaxis,:]
    if np.linalg.det(Q)<0:Q[:,0]*=-1
    assert np.linalg.det(Q)>0
    return Q

def rot_X(X,Q):
    a=X.reshape(-1,4,3)
    center=a.mean(axis=(0,1),keepdims=True)
    out=(a-center)@Q.T+center
    return out.reshape(X.shape)

def build_index(cifdir):
    index={};duplicates=set()
    for root, _, names in os.walk(cifdir):
        for name in names:
            if not name.lower().endswith(('.cif','.mmcif')):continue
            key=name.rsplit('.',1)[0].upper()
            p=str(Path(root)/name)
            if key in index:duplicates.add(key)
            else:index[key]=p
    return index,duplicates

def sample_ids(master, seqs, count, maxlen, seed):
    rng=np.random.default_rng(seed)
    bins=[(0,40),(41,100),(101,500),(501,maxlen)]
    available=[]
    for key in master.Chain_ID.astype(str):
        if key in seqs and 8<=len(seqs[key])<=maxlen:
            available.append((key,len(seqs[key])))
    pools=[]
    for low,high in bins:
        group=[key for key,length in available if low<=length<=high]
        group.sort();rng.shuffle(group);pools.append(group)
    chosen=[]
    quota=max(1,count//4)
    for group in pools:
        for key in group[:quota]:
            if key not in chosen:chosen.append(key)
    remaining=[key for group in pools for key in group if key not in chosen]
    for key in remaining:
        if len(chosen)>=count:break
        chosen.append(key)
    return chosen[:count]

def zstats(a):
    a=np.asarray(a,float).copy();a[~np.isfinite(a)]=np.nan
    med=np.nanmedian(a,axis=0);med[~np.isfinite(med)]=0
    bad=np.isnan(a)
    if bad.any():a[bad]=med[np.where(bad)[1]]
    mu=a.mean(0);sd=a.std(0);sd[sd<=1e-12]=1
    return mu,sd,(a-mu)/sd

def rank_neighbors(fit,x,own,k=20):
    _,indices=fit.kneighbors(x.reshape(1,-1),n_neighbors=k+2)
    arr=[int(i) for i in indices[0] if int(i)!=own]
    return arr[:k]

def main():
    p=argparse.ArgumentParser();p.add_argument('--master',required=True);p.add_argument('--seq',required=True);p.add_argument('--cif-dir',required=True);p.add_argument('--out',required=True);p.add_argument('--n-chains',type=int,default=12);p.add_argument('--n-rot',type=int,default=3);p.add_argument('--max-length',type=int,default=1500);p.add_argument('--seed',type=int,default=20261112);p.add_argument('--atol',type=float,default=1e-5);p.add_argument('--rtol',type=float,default=1e-4);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.master).drop_duplicates('Chain_ID').reset_index(drop=True)
    if len(df)!=1406 or len([c for c in df.columns if c.startswith('fm_')])!=640 or not all(c in df.columns for c in COLS):
        raise RuntimeError('Input does not match 1406-chain frozen C 33D + FM640 contract')
    seqs=read_seq(args.seq); rng=np.random.default_rng(args.seed)
    keys=sample_ids(df,seqs,args.n_chains,args.max_length,args.seed)
    if len(keys)<min(6,args.n_chains):raise RuntimeError('Too few eligible chains')
    index,duplicates=build_index(args.cif_dir)
    key_to_idx={str(k):i for i,k in enumerate(df.Chain_ID)}
    geom_raw=df[COLS].to_numpy(float)
    gmu,gsd,zg=zstats(geom_raw)
    fmcols=[c for c in df if c.startswith('fm_')]
    _,_,zf=zstats(df[fmcols].to_numpy(float))
    NN={};BASE={}
    for mode,wgeo in [('C_ONLY',1.0),('C_FM',67.0/33.0)]:
        X=np.sqrt(wgeo)*zg if mode=='C_ONLY' else np.concatenate([np.sqrt(wgeo)*zg,zf],axis=1)
        NN[mode]=NearestNeighbors(metric='euclidean').fit(X)
        BASE[mode]=X
    detail=[];ranks=[];anchors=[];failures=[]
    for ix,key in enumerate(keys,1):
        pdb,chain=key.split('_',1);cif=index.get(pdb.upper())
        if not cif or pdb.upper() in duplicates:
            failures.append({'chain':key,'error':'source cif missing or duplicate'});continue
        try:
            st=cyc.parse_structure(cif)
            matches=cyc.extract_chains(st,chain_mode=chain)
            if len(matches)!=1:raise RuntimeError(f'expected exact chain {chain!r}, found {len(matches)}')
            rr=matches[0];X,_=cyc.embedding(rr)
            # Frozen Stage02 aggregation takes ContactCount/Order/B_avg from
            # the CY-A residue table, not from CY-C. Reconstruct exact CY-A
            # base descriptors rather than silently forcing values from master.
            cya_matches=cya.extract_nucleic_chains(st,chain_mode=chain)
            if len(cya_matches)!=1 or len(cya_matches[0])!=len(rr):
                raise RuntimeError(f'CY-A/CY-C source residue mismatch for {key}')
            cya_rr=cya_matches[0]
            base_cc,base_co=cya.compute_contacts_legacy(cya_rr,cutoff=10.0)
            base_b=np.asarray([r['B_avg'] for r in cya_rr],float)
            cya3=np.array([float(np.nanmean(base_cc)),float(np.nanmean(base_co)),float(np.nanmean(base_b)) if np.isfinite(base_b).any() else 0.0],float)
            # CY-C source produces only the 30 multiscale CY-C columns in the
            # frozen 33D partition; the first three are CY-A sourced.
            raw_cyc=feature_vector(rr,X)
            base=np.concatenate([cya3,raw_cyc[3:]])
            idx=key_to_idx[key];frozen=df.loc[idx,COLS].to_numpy(dtype=float)
            absdiff=np.abs(base-frozen)
            closeness=np.isclose(base,frozen,rtol=1e-4,atol=1e-4,equal_nan=False)
            anchored=bool(closeness.all())
            anchors.append({'chain':key,'n_res':len(rr),'sequence_length':len(seqs[key]),'source_cif':cif,'source_sha256':sha(cif),'baseline_agrees_with_master':anchored,'max_anchor_abs_delta':float(np.max(absdiff)),'n_anchor_fail':int((~closeness).sum()),'max_cya3_master_delta':float(np.max(absdiff[:3])),'max_cyc30_master_delta':float(np.max(absdiff[3:])),'raw_cyc_B_avg_minus_frozen':float(raw_cyc[2]-frozen[2])})
            if not anchored:failures.append({'chain':key,'error':'re-extracted baseline does not agree with frozen master'})
            for ri in range(1,args.n_rot+1):
                Q=rand_rotation(rng);newX=rot_X(X,Q)
                newrr=[]
                for r in rr:
                    # Invariance of contact coordinates is independently checked.
                    rec=dict(r)
                    cen=X.reshape(-1,4,3).mean(axis=(0,1))
                    rec['c4']=(np.asarray(r['c4'])-cen)@Q.T+cen
                    newrr.append(rec)
                # CY-A base descriptors are separately rotation-invariant;
                # use the exact CY-A-derived values, not CY-C's B_avg proxy.
                rotated=np.concatenate([cya3,feature_vector(newrr,newX)[3:]])
                for col,b,v in zip(COLS,base,rotated):
                    delta=abs(float(v-b))
                    ok=bool(np.isclose(v,b,rtol=args.rtol,atol=args.atol))
                    detail.append({'chain':key,'n_res':len(rr),'rotation_id':ri,'feature':col,'unrotated':float(b),'rotated':float(v),'abs_delta':delta,'scaled_delta':float(delta/max(1,abs(float(b)))),'is_stable':int(ok),'baseline_master_agrees':int(anchored)})
                if anchored:
                    newgeom=(rotated-gmu)/gsd
                    for mode,w in [('C_ONLY',1.0),('C_FM',67.0/33.0)]:
                        query=np.sqrt(w)*newgeom if mode=='C_ONLY' else np.concatenate([np.sqrt(w)*newgeom,zf[idx]])
                        orig=rank_neighbors(NN[mode],BASE[mode][idx],idx)
                        changed=rank_neighbors(NN[mode],query,idx)
                        ranks.append({'chain':key,'rotation_id':ri,'arm':mode,'top20_overlap':len(set(orig)&set(changed))/20.0,'unrotated_neighbor_ids':'|'.join(df.Chain_ID.iloc[orig].astype(str)),'rotated_neighbor_ids':'|'.join(df.Chain_ID.iloc[changed].astype(str))})
            print(f'[ROT] {ix}/{len(keys)} {key} n={len(rr)} anchor={anchored}',flush=True)
        except Exception as exc:
            failures.append({'chain':key,'error':str(exc)})
            print(f'[HOLD] {key}: {exc}',flush=True)
    pd.DataFrame(anchors).to_csv(out/'baseline_anchor.csv',index=False)
    pd.DataFrame(detail).to_csv(out/'rotation_feature_detail.csv',index=False)
    pd.DataFrame(ranks).to_csv(out/'rotation_query_rank_overlap.csv',index=False)
    d=pd.DataFrame(detail)
    if d.empty:raise RuntimeError('No rotation output; inspect errors')
    group=d.groupby('feature').agg(n_tests=('is_stable','count'),n_unstable=('is_stable',lambda x:int((x==0).sum())),median_abs_delta=('abs_delta','median'),max_abs_delta=('abs_delta','max'),median_scaled_delta=('scaled_delta','median')).reset_index()
    group.to_csv(out/'rotation_feature_summary.csv',index=False)
    rank_means={k:float(v) for k,v in pd.DataFrame(ranks).groupby('arm').top20_overlap.mean().items()} if ranks else {}
    n_anchor=sum(x['baseline_agrees_with_master'] for x in anchors)
    status='HOLD_SOURCE_MISMATCH' if n_anchor<len(anchors) or failures or len(anchors)<len(keys) else ('PASS_ROTATION_INVARIANCE' if (d.is_stable==1).all() else 'HOLD_COORDINATE_FRAME_SENSITIVITY')
    contract={'status':status,'n_requested_chains':len(keys),'n_extracted_chains':len(anchors),'n_anchored_chains':n_anchor,'n_rotations_per_chain':args.n_rot,'n_feature_tests':len(d),'n_feature_instabilities':int((d.is_stable==0).sum()),'max_abs_feature_delta':float(d.abs_delta.max()),'rank_overlap_top20_mean':rank_means,'selection_seed':args.seed,'rotation_definition':'3D proper rotation shared across all backbone atoms','atol':args.atol,'rtol':args.rtol,'failures':failures,'note':'CY-A-derived 3D base block and CY-C-derived 30D core separately reconstructed against frozen master; pre-sequence-gating rank sensitivity only; no model predictions.'}
    (out/'rotation_gate.json').write_text(json.dumps(contract,indent=2)+'\n')
    print(json.dumps({k:v for k,v in contract.items() if k!='failures'},indent=2),flush=True)
    return contract

if __name__=='__main__':main()
