#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rnanet.io import load_rna
from rnanet.cyc import embedding,compute_cyc,contact_features,SCALES

def main():
    ap=argparse.ArgumentParser(description='Extract frozen RNA CY-C residue features')
    ap.add_argument('structure'); ap.add_argument('-o','--output',required=True); ap.add_argument('--chain',default='ALL')
    ap.add_argument('--strict-coordinates',action='store_true',help='disable column-mean coordinate imputation')
    a=ap.parse_args(); res=load_rna(a.structure,a.chain); X=embedding(res,impute=not a.strict_coordinates); F=compute_cyc(X); cc,co=contact_features(res)
    rows=[]
    for i,r in enumerate(res):
        z={'ResidueIndex':r.index,'Chain':r.chain,'ResSeq':r.resseq,'ICode':r.icode,'ResName':r.resname,'ContactCount':cc[i],'ContactOrder':co[i]}
        for k in SCALES:
            for n,v in F[k].items(): z[f'{n}_k{k}']=v[i]
        rows.append(z)
    pd.DataFrame(rows).to_csv(a.output,index=False)
    print(f'[OK] CY-C rows={len(rows)} output={a.output}')
if __name__=='__main__': main()
