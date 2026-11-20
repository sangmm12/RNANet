#!/usr/bin/env python3
import argparse, re, subprocess
from pathlib import Path
import pandas as pd

def parse_output(txt):
    tms=[float(x) for x in re.findall(r'TM-score=\s*([0-9.]+)',txt)]
    m=re.search(r'Aligned length=\s*(\d+),\s*RMSD=\s*([0-9.]+),\s*Seq_ID=n_identical/n_aligned=\s*([0-9.]+)',txt)
    if not m:
        # tolerate spacing or slightly different versions
        m=re.search(r'Aligned length\s*=\s*(\d+).*?RMSD\s*=\s*([0-9.]+).*?Seq_ID[^=]*=\s*([0-9.]+)',txt,re.S)
    if len(tms)<2 or not m:
        raise RuntimeError('Could not parse US-align output')
    return dict(tm1=tms[0],tm2=tms[1],aligned_length=int(m.group(1)),rmsd=float(m.group(2)),aligned_seqid=float(m.group(3)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--candidates',required=True)
    ap.add_argument('--chain-manifest',required=True)
    ap.add_argument('--usalign',required=True)
    ap.add_argument('--out-csv',required=True)
    ap.add_argument('--raw-dir',required=True)
    args=ap.parse_args()
    cand=pd.read_csv(args.candidates)
    man=pd.read_csv(args.chain_manifest)
    mp={r.Chain_ID:(r.export_pdb,int(r.n_rna_res)) for r in man.itertuples(index=False)}
    rawdir=Path(args.raw_dir); rawdir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for idx,r in cand.iterrows():
        a=str(r['RNA_A']); b=str(r['RNA_B'])
        if a not in mp or b not in mp: raise RuntimeError(f'missing exported chain {a} or {b}')
        pa,la=mp[a]; pb,lb=mp[b]
        cmd=[args.usalign,pa,pb,'-mol','RNA']
        cp=subprocess.run(cmd,text=True,capture_output=True)
        raw=(cp.stdout or '')+'\n'+(cp.stderr or '')
        raw_path=rawdir/f'{idx+1:03d}__{a.replace("/","-")}__{b.replace("/","-")}.txt'
        raw_path.write_text(raw)
        rec={k:r[k] for k in cand.columns}
        rec.update({'length_A':la,'length_B':lb,'length_ratio':min(la,lb)/max(la,lb),'usalign_exit':cp.returncode,'usalign_raw':str(raw_path)})
        if cp.returncode==0:
            try:
                p=parse_output(raw)
                # US-align score order corresponds to structure1/structure2 normalization.
                rec.update(p)
                rec['tm_min']=min(p['tm1'],p['tm2']); rec['tm_mean']=(p['tm1']+p['tm2'])/2
                rec['coverage_A']=p['aligned_length']/la if la else 0
                rec['coverage_B']=p['aligned_length']/lb if lb else 0
                rec['coverage_min']=min(rec['coverage_A'],rec['coverage_B'])
                rec['usalign_ok']=1
            except Exception as e:
                rec['usalign_ok']=0; rec['usalign_parse_error']=str(e)
        else:
            rec['usalign_ok']=0
        rows.append(rec)
        print(f"[{idx+1}/{len(cand)}] {a} {b} exit={cp.returncode} tm_min={rec.get('tm_min','NA')}")
    out=pd.DataFrame(rows)
    out.to_csv(args.out_csv,index=False)
    print(f'[DONE] {args.out_csv} rows={len(out)} ok={int(out.usalign_ok.sum())}')
if __name__=='__main__': main()
