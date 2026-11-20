#!/usr/bin/env python3
"""Synthetic check for US-align output parser and Top-K summary."""
import sys,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from align_resume import parse_output
from summarize_challenge import main as summary_main,pairkey

def run():
    test='Aligned length= 35, RMSD= 2.50, Seq_ID=n_identical/n_aligned= 0.500\nTM-score= 0.6300\nTM-score= 0.5700\n'
    p=parse_output(test)
    assert p['aligned_length']==35 and p['rmsd']==2.5 and len([p['tm1'],p['tm2']])==2
    with tempfile.TemporaryDirectory() as t:
        root=Path(t); cand=root/'c'; cand.mkdir()
        align=[]
        for offset,mode in enumerate(['C_ONLY','FM_ONLY','C_FM']):
            rows=[]
            for i in range(100):
                a=f'{offset:02X}{i:02X}_A';b=f'{offset+1:02X}{i:02X}_B'
                rows.append({'RNA_A':a,'RNA_B':b})
                align.append({'pair_key':pairkey(a,b),'tm_min':float(.15+.004*i),'coverage_min':float(.7),'rmsd':float(4.0),'aligned_length':25,'usalign_ok':1})
            pd.DataFrame(rows).to_csv(cand/f'candidates_{mode}.csv',index=False)
        pd.DataFrame(align).to_csv(root/'align.csv',index=False)
        sys.argv=['summarize_challenge.py','--candidate-dir',str(cand),'--align',str(root/'align.csv'),'--out',str(root/'sum')]
        summary_main()
        d=pd.read_csv(root/'sum/TopK_comparison.csv')
        assert len(d)==9
    print('[SELFTEST PASS] US-align output parsing and TopK summary')
if __name__=='__main__':run()
