#!/usr/bin/env python3
"""Resumable US-align validation of the fair challenge union.

Uses identical US-align -mol RNA invocation and parse contract as frozen Stage04B.
"""
import argparse,csv,re,subprocess
from pathlib import Path
import numpy as np
import pandas as pd

def pairkey(a,b):return '|'.join(sorted((str(a),str(b))))

def parse_output(txt):
    tms=[float(x) for x in re.findall(r'TM-score=\s*([0-9.]+)',txt)]
    m=re.search(r'Aligned length=\s*(\d+),\s*RMSD=\s*([0-9.]+),\s*Seq_ID=n_identical/n_aligned=\s*([0-9.]+)',txt)
    if not m:m=re.search(r'Aligned length\s*=\s*(\d+).*?RMSD\s*=\s*([0-9.]+).*?Seq_ID[^=]*=\s*([0-9.]+)',txt,re.S)
    if len(tms)<2 or not m:raise RuntimeError('US-align output parser failed')
    return dict(tm1=tms[0],tm2=tms[1],aligned_length=int(m.group(1)),rmsd=float(m.group(2)),aligned_seqid=float(m.group(3)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidates',required=True);ap.add_argument('--chain-manifest',required=True);ap.add_argument('--usalign',required=True);ap.add_argument('--out',required=True);ap.add_argument('--raw-dir',required=True);args=ap.parse_args()
    candidates=pd.read_csv(args.candidates); man=pd.read_csv(args.chain_manifest)
    lookup={str(r.Chain_ID):(str(r.export_pdb),int(r.n_rna_res)) for r in man.itertuples(index=False)}
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    raw_dir=Path(args.raw_dir);raw_dir.mkdir(parents=True,exist_ok=True)
    done={}
    if path.exists():
        for r in pd.read_csv(path).to_dict('records'):done[str(r['pair_key'])]=r
    for i,r in enumerate(candidates.itertuples(index=False),1):
        a=str(r.RNA_A);b=str(r.RNA_B);pair=pairkey(a,b)
        if pair in done and int(done[pair].get('usalign_ok',0))==1:continue
        if a not in lookup or b not in lookup:raise RuntimeError(f'missing exported chain {pair}')
        pa,la=lookup[a];pb,lb=lookup[b]
        cp=subprocess.run([args.usalign,pa,pb,'-mol','RNA'],capture_output=True,text=True)
        txt=(cp.stdout or '')+'\n'+(cp.stderr or '')
        raw_file=raw_dir/f'{i:03d}__{a.replace("/","-")}__{b.replace("/","-")}.txt'
        raw_file.write_text(txt)
        row={'pair_key':pair,'RNA_A':a,'RNA_B':b,'length_A':la,'length_B':lb,'usalign_exit':cp.returncode,'usalign_ok':0,'usalign_raw':str(raw_file)}
        if cp.returncode==0:
            try:
                d=parse_output(txt);row.update(d)
                row.update({'tm_min':min(d['tm1'],d['tm2']),'tm_mean':(d['tm1']+d['tm2'])/2,'coverage_min':min(d['aligned_length']/la,d['aligned_length']/lb),'usalign_ok':1})
            except Exception as exc:row['parse_error']=str(exc)
        done[pair]=row
        # Atomic replace at every candidate for crash-safe resume.
        temp=path.with_name(path.name+'.tmp')
        pd.DataFrame(list(done.values())).to_csv(temp,index=False)
        temp.replace(path)
        print(f'[ALIGN] {i}/{len(candidates)} {pair} ok={row["usalign_ok"]} tm={row.get("tm_min")}',flush=True)
    output=pd.read_csv(path)
    if len(output)!=len(candidates):raise RuntimeError(f'Partial US-align result: {len(output)}/{len(candidates)}')
    if (output.usalign_ok!=1).any():
        raise RuntimeError(f'{int((output.usalign_ok!=1).sum())} US-align failures; do not publish incomplete comparison')
    print(f'[PASS] aligned unique pairs={len(output)}',flush=True)

if __name__=='__main__':main()
