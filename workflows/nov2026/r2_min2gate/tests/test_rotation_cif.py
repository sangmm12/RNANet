#!/usr/bin/env python3
"""Test real mmCIF parsing and anchor checks using a temporary synthetic 1406-row master."""
import tempfile,sys
from pathlib import Path
import numpy as np
import pandas as pd
from Bio.PDB import Structure,Model,Chain,Residue,Atom,MMCIFIO
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'r102_frozen_core'/'src'))
import frozen_step1_cyc_RNA_chainlocal as cyc
from rotation_probe import feature_vector,COLS,main as probe_main

def run():
    rng=np.random.default_rng(12)
    with tempfile.TemporaryDirectory() as t:
        root=Path(t);cifdir=root/'cif';cifdir.mkdir()
        m={'Chain_ID':[f'{i:04X}_A' for i in range(1406)]}
        for col in COLS:m[col]=rng.normal(size=1406)
        for i in range(640):m[f'fm_{i}']=rng.normal(size=1406)
        seqs=[]
        for j in range(12):
            pdb=f'{j:04X}'; s=Structure.Structure('s');mod=Model.Model(0);s.add(mod);ch=Chain.Chain('A');mod.add(ch)
            for i in range(25+j):
                res=Residue.Residue((' ',i+1,' '),'A',' ');ch.add(res)
                xyz=np.array([i*0.8,np.sin(i/2),np.cos(i/4)])+rng.normal(scale=.25,size=3)
                for a_idx,an in enumerate(cyc.ATOM_ORDER):
                    a=Atom.Atom(an,xyz+rng.normal(scale=.15,size=3),25,1,' ',f'{an:>4}',i*4+a_idx+1,element=an[0])
                    res.add(a)
            writer=MMCIFIO();writer.set_structure(s);writer.save(str(cifdir/(pdb+'.cif')))
            st=cyc.parse_structure(str(cifdir/(pdb+'.cif')))
            rr=cyc.extract_chains(st,chain_mode='A')[0]
            vals=feature_vector(rr)
            for col,v in zip(COLS,vals):m[col][j]=v
            seqs.append({'Chain_ID':f'{pdb}_A','Sequence':'AUGC'*(15+j)})
        master=root/'master.csv';pd.DataFrame(m).to_csv(master,index=False)
        seq=root/'seq.csv';pd.DataFrame(seqs).to_csv(seq,index=False)
        argv=['rotation_probe.py','--master',str(master),'--seq',str(seq),'--cif-dir',str(cifdir),'--out',str(root/'rotation'),'--n-chains','12','--n-rot','2','--max-length','500']
        sys.argv=argv;probe_main()
        result=pd.read_csv(root/'rotation/baseline_anchor.csv')
        assert len(result)==12 and result.baseline_agrees_with_master.all(),result.to_string()
        assert len(pd.read_csv(root/'rotation/rotation_feature_detail.csv'))==12*2*33
        print('[SELFTEST PASS] CIF parse, anchor 12/12, rotation feature + rank tables')

if __name__=='__main__':run()
