#!/usr/bin/env python3
"""Fair three-arm C/FM/C+FM challenge with a common kNN, p20 and top-K rule.

CHALLENGER experiment, not an exact replay of Stage04B's separate FM/geometry gates.
Read-only toward source data. The fused arm is newly generated for fair comparison.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

MODES = {'C_ONLY': (1.0, 0.0), 'FM_ONLY': (0.0, 1.0), 'C_FM': (67.0/33.0, 1.0)}

def hash_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def zscore(df):
    a=df.to_numpy(dtype=np.float64)
    a[~np.isfinite(a)]=np.nan
    med=np.nanmedian(a,axis=0)
    med[~np.isfinite(med)]=0
    bad=np.isnan(a)
    if np.any(bad): a[bad]=med[np.where(bad)[1]]
    mu=a.mean(axis=0); std=a.std(axis=0)
    std=np.where(std>1e-12,std,1.0)
    return (a-mu)/std

def read_seq(path):
    df=pd.read_csv(path)
    cm={c.lower():c for c in df.columns}
    ident=next((cm[c] for c in ('chain_id','rnaid','id') if c in cm),None)
    seq=next((cm[c] for c in ('sequence','seq') if c in cm),None)
    if not ident or not seq:raise RuntimeError(f'Sequence ID/sequence columns missing: {list(df.columns)}')
    return {str(k).strip():str(s).strip().upper().replace('T','U') for k,s in df[[ident,seq]].dropna().itertuples(index=False,name=None)}

def kset(s,k=3):
    if len(s)>=k:return frozenset(s[i:i+k] for i in range(len(s)-k+1))
    return frozenset([s]) if s else frozenset()

def kmer_jaccard(s,t,sa,sb):
    if not s and not t:return 1.0
    if not s or not t:return 0.0
    if len(s)<3 or len(t)<3:return float(s==t)
    den=len(sa|sb)
    return float(len(sa&sb)/den) if den else 0.0

def calculate(master, seqs, output, knn=100, percent=20, top=100, modes=None, min_length=20):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    full=pd.read_csv(master).drop_duplicates('Chain_ID').reset_index(drop=True)
    geom=[c for c in full.columns if not c.startswith('fm_') and c!='Chain_ID']
    fm=[c for c in full.columns if c.startswith('fm_')]
    if len(full)!=1406 or len(geom)!=33 or len(fm)!=640:
        raise RuntimeError(f'FROZEN DOMAIN MISMATCH: chains={len(full)}, C={len(geom)}, FM={len(fm)}; expected 1406/33/640')
    allkeys=full.Chain_ID.astype(str).to_list()
    if len(set(allkeys))!=len(allkeys):raise RuntimeError('Nonunique chain IDs')
    if any(x not in seqs for x in allkeys):raise RuntimeError('Missing sequence for a master chain')
    if min_length < 20:raise ValueError('R2 frozen minimum sequence length is 20 nt')
    mask=np.array([len(seqs[x])>=min_length for x in allkeys],dtype=bool)
    df=full.loc[mask].reset_index(drop=True)
    keys=df.Chain_ID.astype(str).to_list()
    if len(df)<101:raise RuntimeError(f'Only {len(df)} chains >= {min_length} nt; cannot run common kNN100')
    # Normalize on the original common 1,406-chain master to avoid renormalizing
    # each arm when the eligibility domain changes. Filter rows before kNN.
    c=zscore(full[geom])[mask];f=zscore(full[fm])[mask];ks=[kset(seqs[z]) for z in keys]
    use=modes or list(MODES)
    info={'experiment':'fixed-rank-budget three-arm retrieval challenge','master_sha256':hash_file(master),'mode_summaries':{},'n_chains':len(df),'full_master_chains':len(full),'min_sequence_length_nt':min_length,'normalization_fit':'common 1406-chain master','geometry_columns':geom,'fm_columns':len(fm),'kmer':3,'fixed_kNN':knn,'first_distance_percentile':percent,'fixed_candidate_cap':top,'extra_geo_fm_gates':False,'sequence_ceiling':0.35}
    for name in use:
        if name not in MODES:raise ValueError(name)
        wg,wf=MODES[name]
        blocks=[]
        if wg:blocks.append(np.sqrt(wg)*c)
        if wf:blocks.append(np.sqrt(wf)*f)
        x=np.concatenate(blocks,axis=1)
        k=min(int(knn),len(df)-1)
        nn=NearestNeighbors(n_neighbors=k+1,metric='euclidean').fit(x)
        distances, indices=nn.kneighbors(x,return_distance=True)
        threshold=float(np.percentile(distances[:,1:][np.isfinite(distances[:,1:])],percent))
        rows=[]
        for i in range(len(df)):
            key_i=keys[i];pdb_i=key_i.split('_',1)[0].upper();si=seqs[key_i]
            for r in range(1,indices.shape[1]):
                j=int(indices[i,r]); dst=float(distances[i,r])
                if j<=i or not np.isfinite(dst) or dst>threshold:continue
                key_j=keys[j];pdb_j=key_j.split('_',1)[0].upper()
                if pdb_i==pdb_j:continue
                sj=seqs[key_j]
                sim=kmer_jaccard(si,sj,ks[i],ks[j])
                if sim>0.35:continue
                distg=float(np.linalg.norm(c[i]-c[j]));distf=float(np.linalg.norm(f[i]-f[j]))
                rows.append({'RNA_A':key_i,'RNA_B':key_j,'PDB_A':pdb_i,'PDB_B':pdb_j,'Feature_Dist':dst,'Geo_Dist':distg,'FM_Dist':distf,'SeqSim':sim,'SeqDiff':1-sim,'Length_A':len(si),'Length_B':len(sj),'Score':(1-sim)/(dst+1e-9),'seq_metric':'kmer3_jaccard','arm':name})
        if len(rows)==0:raise RuntimeError(f'NO ELIGIBLE CANDIDATES in {name}; do not relax filters silently')
        cand=pd.DataFrame(rows)
        cand['_pdbpair']=['|'.join(sorted((a,b))) for a,b in zip(cand.PDB_A,cand.PDB_B)]
        # Tertiary deterministic key added only to eliminate unstable exact ties.
        cand=cand.sort_values(['Score','SeqDiff','RNA_A','RNA_B'],ascending=[False,False,True,True],kind='mergesort')
        cand=cand.drop_duplicates('_pdbpair',keep='first').drop(columns=['_pdbpair']).head(top).reset_index(drop=True)
        cand.insert(0,'candidate_rank',np.arange(1,len(cand)+1))
        cand.to_csv(output/f'candidates_{name}.csv',index=False)
        info['mode_summaries'][name]={'n_scanned_sequence_qualified':len(rows),'n_ranked_after_pdb_dedup_and_cap':len(cand),'median_min_candidate_length_nt':float(cand[['Length_A','Length_B']].min(axis=1).median()),'p20_distance_threshold':threshold,'w_geo':wg,'w_fm':wf}
        print(f'[ARM] {name}: scanned={len(rows)} ranked={len(cand)} p20={threshold:.5f}',flush=True)
    allpairs=[]
    for name in use:
        d=pd.read_csv(output/f'candidates_{name}.csv')
        allpairs.append(d[['RNA_A','RNA_B']])
    union=pd.concat(allpairs,ignore_index=True)
    union['_pair']=['|'.join(sorted((a,b))) for a,b in zip(union.RNA_A,union.RNA_B)]
    union=union.drop_duplicates('_pair').drop(columns=['_pair']).reset_index(drop=True)
    union.to_csv(output/'union_candidates.csv',index=False)
    info['n_structural_alignments_required']=len(union)
    (output/'retrieval_contract.json').write_text(json.dumps(info,indent=2,allow_nan=False)+'\n')
    print('[PASS] retrieval candidate challenge completed; structural evaluation still required',flush=True)
    return info

def main():
    a=argparse.ArgumentParser();a.add_argument('--master',required=True);a.add_argument('--seq',required=True);a.add_argument('--out',required=True);a.add_argument('--kNN',type=int,default=100);a.add_argument('--percent',type=float,default=20);a.add_argument('--topN',type=int,default=100);a.add_argument('--min-length',type=int,default=20);a.add_argument('--modes',default='C_ONLY,FM_ONLY,C_FM');args=a.parse_args()
    seqs=read_seq(args.seq)
    calculate(args.master,seqs,args.out,knn=args.kNN,percent=args.percent,top=args.topN,modes=args.modes.split(','),min_length=args.min_length)

if __name__=='__main__':main()
