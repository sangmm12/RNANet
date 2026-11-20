#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rnanet.surf import extract

def main():
    ap=argparse.ArgumentParser(description='Extract frozen RNA SURF57 features (MSMS only; no hull fallback)')
    ap.add_argument('structure'); ap.add_argument('-o','--output',required=True); ap.add_argument('--chain',default='ALL'); ap.add_argument('--msms-bin',required=True)
    a=ap.parse_args(); df=extract(a.structure,a.chain,a.msms_bin); df.to_csv(a.output,index=False)
    print(f'[OK] SURF57 rows={len(df)} output={a.output}')
if __name__=='__main__': main()
