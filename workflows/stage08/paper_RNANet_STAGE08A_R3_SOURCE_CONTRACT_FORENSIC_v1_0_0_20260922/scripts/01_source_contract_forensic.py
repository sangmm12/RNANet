#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, gzip, hashlib, json, os, re, tarfile
from pathlib import Path

BIN_SUFFIXES=(".npy",".npz",".dat",".bin",".mmap",".memmap",".pkl",".pickle",".joblib",".parquet",".feather",".csv",".csv.gz",".tsv",".tsv.gz",".json",".txt")
SURF57_RE=re.compile(r"_SURF57\.csv$",re.I)
IO_RE=re.compile(r"memmap|open_memmap|np\.save|np\.load|savez|to_csv|read_csv|to_parquet|read_parquet|to_pickle|read_pickle|matrix|chain_row_ranges|feature_names|columns",re.I)


def rel(root,p):
    try:return str(p.resolve().relative_to(root.resolve()))
    except Exception:return str(p)

def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def suffix(p):
    n=p.name.lower()
    for s in BIN_SUFFIXES:
        if n.endswith(s):return s
    return p.suffix.lower()

def head_cols(p):
    n=p.name.lower()
    try:
        if n.endswith('.csv.gz') or n.endswith('.tsv.gz'):
            with gzip.open(p,'rt',encoding='utf-8',errors='replace') as f: line=f.readline().rstrip('\r\n')
        elif n.endswith('.csv') or n.endswith('.tsv'):
            with p.open('rt',encoding='utf-8',errors='replace') as f: line=f.readline().rstrip('\r\n')
        else:return []
        d='\t' if '.tsv' in n else ','
        return next(csv.reader([line],delimiter=d))
    except Exception:return []

def binary_meta(p):
    out={"shape":"","dtype":"","arrays":"","rows":""}
    n=p.name.lower()
    try:
        if n.endswith('.npy'):
            import numpy as np
            a=np.load(p,mmap_mode='r',allow_pickle=False)
            out['shape']='x'.join(map(str,a.shape)); out['dtype']=str(a.dtype)
            if a.ndim>=1:out['rows']=int(a.shape[0])
        elif n.endswith('.npz'):
            import numpy as np
            z=np.load(p,allow_pickle=False)
            desc=[]
            for k in z.files:
                a=z[k]; desc.append(f"{k}:{'x'.join(map(str,a.shape))}:{a.dtype}")
            out['arrays']=';'.join(desc[:30])
        elif n.endswith('.parquet'):
            import pyarrow.parquet as pq
            pf=pq.ParquetFile(p); out['rows']=int(pf.metadata.num_rows); out['shape']=f"{pf.metadata.num_rows}x{len(pf.schema_arrow.names)}"
    except Exception as e:
        out['arrays']=f"ERROR:{type(e).__name__}:{e}"[:500]
    return out

def inventory_files(project, root, exclude_residue=False):
    rows=[]
    if not root.exists():return rows
    for p in root.rglob('*'):
        if not p.is_file():continue
        rp=rel(root,p)
        if exclude_residue and ('/residue/' in f'/{rp}/' or SURF57_RE.search(p.name)):
            continue
        s=suffix(p)
        if s not in BIN_SUFFIXES:continue
        meta=binary_meta(p)
        cols=head_cols(p)
        rows.append({"path":rel(project,p),"bytes":p.stat().st_size,"suffix":s,"shape":meta['shape'],"dtype":meta['dtype'],"rows":meta['rows'],"arrays":meta['arrays'],"columns_n":len(cols),"header":";".join(cols[:60])})
    rows.sort(key=lambda x:(-x['bytes'],x['path']))
    return rows

def write_tsv(p,rows,fields):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',extrasaction='ignore');w.writeheader();w.writerows(rows)

def source_evidence(project):
    roots=[]
    tool=project/'stage05a_tools'
    if tool.exists():
        for p in tool.iterdir():
            if p.is_dir() and any(k in p.name.lower() for k in ['full_core','surf_ad_sensitivity','chain_sensitivity']):roots.append(p)
    rows=[]
    for root in roots:
        for p in root.rglob('*.py'):
            try:lines=p.read_text(errors='replace').splitlines()
            except Exception:continue
            for i,line in enumerate(lines,1):
                if IO_RE.search(line):
                    rows.append({"script":rel(project,p),"line":i,"content":line.strip()[:1200]})
    return rows

def tar_inventory(project):
    rows=[]
    pats=list((project/'05_ligand_full_core').glob('*.tar.gz')) if (project/'05_ligand_full_core').exists() else []
    pats+=list((project/'05_ligand_surf_closeout').glob('*.tar.gz')) if (project/'05_ligand_surf_closeout').exists() else []
    for p in sorted(pats):
        try:
            with tarfile.open(p,'r:gz') as tf:
                names=tf.getnames()
            interesting=[n for n in names if re.search(r'matrix|feature|surf|manifest|split|\.npy$|\.npz$|\.parquet$',n,re.I)]
            rows.append({"archive":rel(project,p),"bytes":p.stat().st_size,"members_n":len(names),"interesting_members_n":len(interesting),"interesting_members":";".join(interesting[:120])})
        except Exception as e:
            rows.append({"archive":rel(project,p),"bytes":p.stat().st_size,"members_n":"","interesting_members_n":"","interesting_members":f"ERROR:{e}"})
    return rows

def count_surf_shards(project):
    d=project/'05_ligand_surf_ad_sensitivity/03_surf_extraction/residue'
    if not d.exists():return 0,0
    fs=list(d.glob('*_SURF57.csv'))
    total=sum(p.stat().st_size for p in fs)
    return len(fs),total

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project-root',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    project=Path(a.project_root).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    r2=project/'08_v2_systematic_selection/source_binding_r2_v1_0_0/STAGE08A_R2_EXACT_SOURCE_BINDING_GATE.json'
    r2j=json.loads(r2.read_text()) if r2.exists() else {}

    full_matrix=inventory_files(project,project/'05_ligand_full_core/matrix')
    full_root=inventory_files(project,project/'05_ligand_full_core')
    surf_agg=inventory_files(project,project/'05_ligand_surf_ad_sensitivity',exclude_residue=True)
    src=source_evidence(project)
    tins=tar_inventory(project)
    surf_shards,surf_shard_bytes=count_surf_shards(project)

    fields=['path','bytes','suffix','shape','dtype','rows','arrays','columns_n','header']
    write_tsv(out/'FULL_CORE_MATRIX_DIR_FILES.tsv',full_matrix,fields)
    write_tsv(out/'FULL_CORE_ALL_NONSHARD_ASSETS.tsv',full_root,fields)
    write_tsv(out/'SURF_AGGREGATE_CANDIDATES.tsv',surf_agg,fields)
    write_tsv(out/'SOURCE_IO_EVIDENCE.tsv',src,['script','line','content'])
    write_tsv(out/'ARCHIVE_CONTENT_EVIDENCE.tsv',tins,['archive','bytes','members_n','interesting_members_n','interesting_members'])

    # Existing aggregate arrays/tables with expected first dimension.
    full_expected=[r for r in full_root if str(r.get('rows')) in {'6252187','6250559'}]
    surf_expected=[r for r in surf_agg if str(r.get('rows'))=='431226']

    required={
      'full_chain_manifest':project/'05_ligand_full_core/full_chain_manifest.csv',
      'chain_row_ranges':project/'05_ligand_full_core/matrix/chain_row_ranges.csv',
      'cya_dir':project/'05_ligand_full_core/geometry/cya',
      'cyc_dir':project/'05_ligand_full_core/geometry/cyc',
      'surf_residue_dir':project/'05_ligand_surf_ad_sensitivity/03_surf_extraction/residue',
      'surf_builder_v102':project/'stage05a_tools/surf_ad_sensitivity_v1_0_2/src/build_surf_ad_matrix.py',
    }
    present={k:p.exists() for k,p in required.items()}
    reconstructible=all(present.values()) and surf_shards>7000

    if full_expected and surf_expected:
        mode='EXISTING_AGGREGATES_DISCOVERED'
        gate='PASS_STAGE08A_R3_EXISTING_AGGREGATES_DISCOVERED'
        next_action='bind_existing_aggregates'
        gate_pass=True
    elif reconstructible:
        mode='DETERMINISTIC_REBUILD_REQUIRED'
        gate='PASS_STAGE08A_R3_RECONSTRUCTIBLE_SOURCE_CONTRACT'
        next_action='build_canonical_v2_matrices_from_frozen_stage05a_sources'
        gate_pass=True
    else:
        mode='SOURCE_CONTRACT_INCOMPLETE'
        gate='BLOCK_STAGE08A_R3_SOURCE_CONTRACT_INCOMPLETE'
        next_action='inspect_missing_contract_components'
        gate_pass=False

    g={
      'gate':gate,'gate_pass':gate_pass,'training_performed':False,'geometry_recomputed':False,'test17_used_for_v2_selection':False,
      'r2_gate':r2j.get('gate'),'r2_gate_pass':bool(r2j.get('gate_pass',False)),
      'source_mode':mode,'next_action':next_action,
      'full_existing_expected_row_assets_n':len(full_expected),'full_existing_expected_row_assets':[r['path'] for r in full_expected[:20]],
      'surf_existing_431226_assets_n':len(surf_expected),'surf_existing_431226_assets':[r['path'] for r in surf_expected[:20]],
      'surf57_residue_shards_n':surf_shards,'surf57_residue_shards_bytes':surf_shard_bytes,
      'required_source_components':{k:{'path':rel(project,p),'present':present[k]} for k,p in required.items()},
      'reconstructible_from_frozen_stage05a_sources':reconstructible,
      'interpretation':'R2 searched tabular aggregate files and correctly blocked. R3 determines whether Stage05A retained a binary aggregate or whether the v2 canonical matrix must be deterministically rebuilt from frozen per-chain geometry/SURF assets and frozen row/label contracts. No model is selected here.'
    }
    (out/'STAGE08A_R3_SOURCE_CONTRACT_GATE.json').write_text(json.dumps(g,indent=2,sort_keys=True)+'\n')
    print(json.dumps(g,indent=2,sort_keys=True))
    print('[OUT]',out)

if __name__=='__main__':main()
