#!/usr/bin/env python3
import argparse, csv, os
from pathlib import Path
import pandas as pd
from Bio.PDB import MMCIFParser

RNA_ATOM="C4'"

def find_cif(cif_dir: Path, pdb: str):
    target=pdb.lower()+'.cif'
    hits=[p for p in cif_dir.rglob('*') if p.is_file() and p.name.lower()==target]
    if len(hits)!=1:
        raise RuntimeError(f'{pdb}: expected exactly one {target}, found {len(hits)}: {hits[:5]}')
    return hits[0]

def residue_has_rna_anchor(res):
    return any(a.get_id().strip()==RNA_ATOM for a in res.get_atoms())

def pdb_atom_line(serial, atom, resname, chain, resseq, icode=' '):
    name=atom.get_name().strip()
    alt=(atom.get_altloc() or ' ').strip() or ' '
    x,y,z=map(float,atom.get_coord())
    occ=atom.get_occupancy(); occ=1.0 if occ is None else float(occ)
    b=atom.get_bfactor(); b=0.0 if b is None else float(b)
    element=(getattr(atom,'element','') or '').strip().upper()[:2]
    # Right/left atom-name alignment simplified but valid for common RNA atoms.
    atom_field=f'{name:>4s}'[:4]
    return f"ATOM  {serial:5d} {atom_field}{alt:1s}{resname:>3s} {chain:1s}{resseq:4d}{icode:1s}   {x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{b:6.2f}          {element:>2s}\n"

def export_chain(cif_path: Path, chain_id: str, out_path: Path):
    st=MMCIFParser(QUIET=True, auth_chains=True, auth_residues=True).get_structure(cif_path.stem, str(cif_path))
    model=next(st.get_models())
    matches=[ch for ch in model if str(ch.id)==str(chain_id)]
    if len(matches)!=1:
        avail=[str(ch.id) for ch in model]
        raise RuntimeError(f'{cif_path.name}: exact auth chain {chain_id!r} not uniquely found; available={avail}')
    ch=matches[0]
    rna_res=[r for r in ch if residue_has_rna_anchor(r)]
    if not rna_res:
        raise RuntimeError(f'{cif_path.name}:{chain_id} has no C4-prime RNA residues')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serial=1; new_resseq=1
    with open(out_path,'w') as f:
        for res in rna_res:
            resname=str(res.get_resname()).strip()[:3] or 'N'
            # choose highest-occupancy atom per atom name to avoid altloc duplication
            atoms={}
            for atom in res.get_atoms():
                nm=atom.get_name().strip()
                occ=atom.get_occupancy(); occ=0.0 if occ is None else float(occ)
                if nm not in atoms or occ > atoms[nm][0]:
                    atoms[nm]=(occ,atom)
            for nm in sorted(atoms):
                atom=atoms[nm][1]
                f.write(pdb_atom_line(serial, atom, resname, 'A', new_resseq))
                serial+=1
            new_resseq+=1
        f.write('TER\nEND\n')
    return len(rna_res)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--candidates',required=True)
    ap.add_argument('--cif-dir',required=True)
    ap.add_argument('--outdir',required=True)
    ap.add_argument('--top-n',type=int,default=-1)
    args=ap.parse_args()
    df=pd.read_csv(args.candidates)
    if args.top_n>0: df=df.head(args.top_n).copy()
    need=[]
    for _,r in df.iterrows():
        for side in ('A','B'):
            key=str(r[f'RNA_{side}'])
            if '_' not in key: raise RuntimeError(f'bad chain key {key}')
            pdb,chain=key.split('_',1)
            need.append((pdb.upper(),chain,key))
    # preserve exact chain case; only PDB accession normalized
    seen={}
    rows=[]
    cif_dir=Path(args.cif_dir); outdir=Path(args.outdir)
    for pdb,chain,key in need:
        if key in seen: continue
        cif=find_cif(cif_dir,pdb)
        out=outdir/'chains'/f'{pdb}__{chain}.pdb'
        n=export_chain(cif,chain,out)
        seen[key]=out
        rows.append({'Chain_ID':key,'PDB':pdb,'Chain':chain,'n_rna_res':n,'source_cif':str(cif),'export_pdb':str(out)})
        print(f'[OK] {key} n_res={n} -> {out}')
    pd.DataFrame(rows).sort_values('Chain_ID').to_csv(outdir/'chain_export_manifest.csv',index=False)
    print(f'[DONE] unique candidate chains={len(rows)}')
if __name__=='__main__': main()
