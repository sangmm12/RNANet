#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rnanet.io import load_rna
from rnanet.cya import compute_cya,SCALES

def main():
    ap=argparse.ArgumentParser(description='Extract frozen RNA CY-A residue features')
    ap.add_argument('structure'); ap.add_argument('-o','--output',required=True); ap.add_argument('--chain',default='ALL')
    a=ap.parse_args(); res=load_rna(a.structure,a.chain); Q,cc,co,A=compute_cya(res)
    rows=[]
    for i,r in enumerate(res):
        z={'ResidueIndex':r.index,'Chain':r.chain,'ResSeq':r.resseq,'ICode':r.icode,'ResName':r.resname,
           'eta':Q[i,0],'theta':Q[i,1],'chi':Q[i,2],'puckerP':Q[i,3],'ContactCount':cc[i],'ContactOrder':co[i]}
        for k in SCALES:
            for n,v in A[k].items(): z[f'{n}_k{k}']=v[i]
        rows.append(z)
    pd.DataFrame(rows).to_csv(a.output,index=False)
    print(f'[OK] CY-A rows={len(rows)} output={a.output}')
if __name__=='__main__': main()
