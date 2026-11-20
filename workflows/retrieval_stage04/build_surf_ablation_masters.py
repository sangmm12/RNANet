#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--master',required=True); ap.add_argument('--surf',required=True); ap.add_argument('--outdir',required=True); ap.add_argument('--contract',required=True)
    args=ap.parse_args(); base=pd.read_csv(args.master); surf=pd.read_csv(args.surf)
    if base.Chain_ID.duplicated().any() or surf.Chain_ID.duplicated().any(): raise RuntimeError('duplicate exact Chain_ID')
    m=base.merge(surf,on='Chain_ID',how='inner',validate='one_to_one')
    fm=[c for c in base.columns if c.startswith('fm_')]
    angles=[c for c in ['eta','theta','chi','puckerP'] if c in base.columns]
    aux=[c for c in ['ContactCount','ContactOrder','B_avg'] if c in base.columns]
    a=angles+[c for c in base.columns if c.startswith('cya_')]
    c=aux+[c for c in base.columns if c.startswith('cyc_')]
    ac=aux+a+[x for x in c if x not in aux]
    s=[c for c in surf.columns if c!='Chain_ID']
    # exact expected final dimensions for the current frozen RNA geometry.
    exp={'A':34,'C':33,'AC':67,'S':57,'CS':90,'ACS':124,'FM':640}
    got={'A':len(a),'C':len(c),'AC':len(ac),'S':len(s),'CS':len(c)+len(s),'ACS':len(ac)+len(s),'FM':len(fm)}
    if got!=exp: raise RuntimeError(f'ablation dimension mismatch got={got} expected={exp}')
    defs={'C':c,'AC':ac,'S':s,'CS':c+s,'ACS':ac+s}
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    for name,geom in defs.items():
        d=m[['Chain_ID']+geom+fm].copy(); d.to_csv(outdir/f'master_{name}.csv',index=False)
    contract={'base_chains':len(base),'surf_chains':len(surf),'common_chains':len(m),'coverage':len(m)/len(base),'dimensions':got,
              'variants':{k:len(v) for k,v in defs.items()},'fm_dim':len(fm),'same_common_universe_for_all_variants':True,'gate_pass':len(m)>0}
    Path(args.contract).write_text(json.dumps(contract,indent=2)); print(json.dumps(contract,indent=2))
if __name__=='__main__': main()
