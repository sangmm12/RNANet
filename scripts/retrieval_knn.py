#!/usr/bin/env python3
"""Generic kNN candidate generator for an already-frozen feature matrix.
This utility intentionally does not invent the study-specific post-kNN fusion filters.
"""
import argparse,sys
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rnanet.retrieval import knn_euclidean,kmer_jaccard

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('features',help='CSV: id, sequence, numeric feature columns')
    ap.add_argument('--id-col',default='chain_key'); ap.add_argument('--seq-col',default='sequence'); ap.add_argument('-k',type=int,default=100); ap.add_argument('-o','--output',required=True); a=ap.parse_args()
    d=pd.read_csv(a.features); feat=[c for c in d.columns if c not in {a.id_col,a.seq_col,'pdb_id'} and pd.api.types.is_numeric_dtype(d[c])]
    X=d[feat].to_numpy(float); mu=np.nanmean(X,0); sd=np.nanstd(X,0); sd[sd==0]=1.; X=(np.where(np.isfinite(X),X,mu)-mu)/sd
    ids,dist=knn_euclidean(X,a.k); rows=[]
    for i in range(len(d)):
        for rank,(j,dd) in enumerate(zip(ids[i],dist[i]),1):
            rows.append({'query':d.iloc[i][a.id_col],'candidate':d.iloc[j][a.id_col],'rank':rank,'feature_distance':dd,'seqsim_k3':kmer_jaccard(d.iloc[i][a.seq_col],d.iloc[j][a.seq_col],3)})
    pd.DataFrame(rows).to_csv(a.output,index=False); print(f'[OK] candidates={len(rows)} output={a.output}')
if __name__=='__main__': main()
