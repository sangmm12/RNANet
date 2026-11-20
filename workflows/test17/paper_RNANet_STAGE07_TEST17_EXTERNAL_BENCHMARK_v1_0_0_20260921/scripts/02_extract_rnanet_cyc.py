#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stage07_common import load_label_pickle, normalize_label_collection, read_fasta, json_dump, sha256_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-root', required=True)
    ap.add_argument('--package-root', required=True)
    a = ap.parse_args()

    project = Path(a.project_root).resolve()
    package = Path(a.package_root).resolve()
    out = project / '07_test17_external_benchmark' / 'v1_0_0'
    manifest_path = out / 'DATASET_MANIFEST.csv'
    preflight = out / 'STAGE07A_TEST17_PREFLIGHT_GATE.json'
    if not preflight.exists() or not json.loads(preflight.read_text()).get('gate_pass'):
        raise RuntimeError('Stage07A preflight gate missing or false')

    # Verify the bundled public implementation is byte-exact to RNANet GitHub v2.0.0.
    contract = json.loads((package / 'config' / 'stage07_test17_contract.json').read_text())
    cyc_path = package.parents[2] / 'src' / 'rnanet' / 'cyc.py'
    io_path = package.parents[2] / 'src' / 'rnanet' / 'io.py'
    if sha256_file(cyc_path) != contract['rnanet']['cyc_source_sha256']:
        raise RuntimeError('bundled CY-C implementation hash mismatch')
    if sha256_file(io_path) != contract['rnanet']['io_source_sha256']:
        raise RuntimeError('bundled RNA IO implementation hash mismatch')

    sys.path.insert(0, str(package.parents[2] / 'src'))
    from rnanet.io import load_rna
    from rnanet.cyc import embedding, compute_cyc, contact_features, SCALES

    manifest = pd.read_csv(manifest_path)
    vendor = out / 'vendor' / 'rna-small-molecule-review'
    bs = vendor / 'BindingSite'
    common = bs / 'data' / 'common'
    train_fastas = sorted((common / 'train60_fastas').glob('*.fasta'))
    test_fastas = sorted((common / 'test17_fastas').glob('*.fasta'))
    train_labels = normalize_label_collection(
        load_label_pickle(common / 'all_label' / 'label' / 'train_label.pkl'), train_fastas
    )
    test_labels = normalize_label_collection(
        load_label_pickle(common / 'all_label' / 'label' / 'test17_labels.pkl'), test_fastas
    )
    label_maps = {'Train60': train_labels, 'Test17': test_labels}

    rows = []
    chain_audit = []
    for j, rec in manifest.iterrows():
        set_name = str(rec['set'])
        chain_key = str(rec['chain_key'])
        chain_id = str(rec['chain_id'])
        fasta = Path(rec['fasta'])
        structure = Path(rec['structure'])
        _, seq = read_fasta(fasta)
        labels = np.asarray(label_maps[set_name][chain_key], dtype=int)

        residues = load_rna(str(structure), chain_id)
        if len(residues) != len(seq):
            raise RuntimeError(
                f'RNA structure/FASTA mismatch for {chain_key}: '
                f'residues={len(residues)} sequence={len(seq)} structure={structure}'
            )
        if len(labels) != len(seq):
            raise RuntimeError(f'label length mismatch for {chain_key}')

        X = embedding(residues, impute=True)
        F = compute_cyc(X)
        cc, co = contact_features(residues)
        for i, residue in enumerate(residues):
            base = seq[i]
            z = {
                'set': set_name,
                'chain_key': chain_key,
                'pdb_id': str(rec['pdb_id']),
                'chain_id': chain_id,
                'ResidueIndex': i + 1,
                'ResSeq': residue.resseq,
                'ICode': residue.icode,
                'ResName': residue.resname,
                'base': base,
                'label': int(labels[i]),
                'ContactCount': float(cc[i]),
                'ContactOrder': float(co[i]),
                'nt_A': 1.0 if base == 'A' else 0.0,
                'nt_C': 1.0 if base == 'C' else 0.0,
                'nt_G': 1.0 if base == 'G' else 0.0,
                'nt_U': 1.0 if base == 'U' else 0.0,
            }
            for k in SCALES:
                for name, arr in F[k].items():
                    z[f'{name}_k{k}'] = float(arr[i])
            rows.append(z)
        chain_audit.append({
            'set': set_name,
            'chain_key': chain_key,
            'n_rows': len(residues),
            'n_positive': int(labels.sum()),
            'feature_extraction_ok': True,
        })
        print(f'[CY-C] {j+1:02d}/{len(manifest)} {set_name} {chain_key} rows={len(residues)} pos={int(labels.sum())}')

    d = pd.DataFrame(rows)
    feature_cols = ['ContactCount', 'ContactOrder']
    for k in SCALES:
        for name in ('integrability_err', 'kahler_err', 'omega_norm', 'logdetG', 'ricci_proxy'):
            feature_cols.append(f'{name}_k{k}')
    if len(feature_cols) != 42:
        raise AssertionError(len(feature_cols))
    model_features = feature_cols + ['nt_A', 'nt_C', 'nt_G', 'nt_U']
    if len(model_features) != 46:
        raise AssertionError(len(model_features))
    forbidden = [c for c in model_features if any(x in c.lower() for x in ('b_factor', 'bfactor', 'identifier', 'pdb_id', 'resseq'))]
    if forbidden:
        raise RuntimeError(f'forbidden predictors: {forbidden}')

    matrix = out / 'rnanet_test17_cyc_matrix.csv.gz'
    d.to_csv(matrix, index=False, compression='gzip')
    (out / 'CY-C_FEATURES_42.txt').write_text('\n'.join(feature_cols) + '\n')
    (out / 'CY-C_MODEL_FEATURES_46.txt').write_text('\n'.join(model_features) + '\n')
    pd.DataFrame(chain_audit).to_csv(out / 'CY-C_CHAIN_EXTRACTION_AUDIT.csv', index=False)

    n_test = int((d['set'] == 'Test17').sum())
    gate = {
        'gate': 'PASS_STAGE07B_CYC_EXTRACT',
        'gate_pass': bool(n_test == 583 and len(feature_cols) == 42 and len(model_features) == 46),
        'rows_total': int(len(d)),
        'rows_train60': int((d['set'] == 'Train60').sum()),
        'rows_test17': n_test,
        'test17_positive': int(d.loc[d['set'].eq('Test17'), 'label'].sum()),
        'cyc_intrinsic_dim': len(feature_cols),
        'model_dim_with_nt_onehot': len(model_features),
        'B_factor_included': False,
        'SURF_included': False,
        'matrix_sha256': sha256_file(matrix),
    }
    json_dump(out / 'STAGE07B_CYC_EXTRACT_GATE.json', gate)
    print(json.dumps(gate, indent=2))
    if not gate['gate_pass']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
