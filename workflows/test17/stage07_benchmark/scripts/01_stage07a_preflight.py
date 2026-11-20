#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stage07_common import (
    find_structure, json_dump, load_label_pickle, normalize_label_collection,
    parse_chain_from_stem, read_fasta, sha256_file,
)


def git(*args: str, cwd: Path | None = None) -> str:
    p = subprocess.run(['git', *args], cwd=cwd, text=True, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{p.stdout}\n{p.stderr}")
    return p.stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-root', required=True)
    ap.add_argument('--package-root', required=True)
    a = ap.parse_args()

    project = Path(a.project_root).resolve()
    package = Path(a.package_root).resolve()
    out = project / '07_test17_external_benchmark' / 'v1_0_0'
    vendor = out / 'vendor' / 'rna-small-molecule-review'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'logs').mkdir(exist_ok=True)

    contract = json.loads((package / 'config' / 'stage07_test17_contract.json').read_text())
    repo = contract['benchmark']['repository']
    branch = contract['benchmark']['branch']

    if not vendor.exists():
        vendor.parent.mkdir(parents=True, exist_ok=True)
        print(f'[CLONE] {repo} -> {vendor}')
        git('clone', '--depth', '1', '--branch', branch, repo, str(vendor))
    if not (vendor / '.git').exists():
        raise RuntimeError(f'vendor path exists but is not a git clone: {vendor}')

    commit = git('rev-parse', 'HEAD', cwd=vendor)
    lock = out / 'BENCHMARK_COMMIT.lock'
    if lock.exists():
        frozen = lock.read_text().strip()
        if frozen != commit:
            raise RuntimeError(f'benchmark commit drift: frozen={frozen} current={commit}')
    else:
        lock.write_text(commit + '\n')
    print(f'[FREEZE] benchmark_commit={commit}')

    bs = vendor / contract['benchmark']['subdir']
    if not bs.exists():
        raise FileNotFoundError(f'BindingSite directory missing at {bs}; repository layout changed')

    validator = bs / 'validate_review.py'
    if not validator.exists():
        raise FileNotFoundError(validator)
    p = subprocess.run([sys.executable, str(validator)], cwd=bs, text=True, capture_output=True)
    (out / 'logs' / 'benchmark_validate_review.log').write_text(p.stdout + p.stderr)
    if p.returncode != 0:
        raise RuntimeError('external benchmark validate_review.py failed; see log')
    print(p.stdout.strip())

    common = bs / 'data' / 'common'
    train_fastas = sorted((common / 'train60_fastas').glob('*.fasta'))
    test_fastas = sorted((common / 'test17_fastas').glob('*.fasta'))
    train_obj = load_label_pickle(common / 'all_label' / 'label' / 'train_label.pkl')
    test_obj = load_label_pickle(common / 'all_label' / 'label' / 'test17_labels.pkl')
    train_labels = normalize_label_collection(train_obj, train_fastas)
    test_labels = normalize_label_collection(test_obj, test_fastas)

    exp_train = int(contract['benchmark']['expected_train_chains'])
    exp_test = int(contract['benchmark']['expected_test_chains'])
    exp_nt = int(contract['benchmark']['expected_test_nucleotides'])
    if len(train_fastas) != exp_train or len(test_fastas) != exp_test:
        raise RuntimeError(f'chain count mismatch train={len(train_fastas)} test={len(test_fastas)}')

    rows = []
    for set_name, fastas, labels in [('Train60', train_fastas, train_labels), ('Test17', test_fastas, test_labels)]:
        for fp in fastas:
            header, seq = read_fasta(fp)
            lab = labels[fp.stem]
            if len(seq) != len(lab):
                raise RuntimeError(f'FASTA/label length mismatch {fp.stem}: seq={len(seq)} labels={len(lab)}')
            if set(lab.tolist()) - {0, 1}:
                raise RuntimeError(f'non-binary labels for {fp.stem}')
            pdb_id, chain = parse_chain_from_stem(fp.stem)
            structure = find_structure(bs, pdb_id, chain)
            rows.append({
                'set': set_name,
                'chain_key': fp.stem,
                'pdb_id': pdb_id,
                'chain_id': chain,
                'n_nt': len(seq),
                'n_positive': int(lab.sum()),
                'positive_fraction': float(lab.mean()),
                'fasta': str(fp.resolve()),
                'structure': str(structure),
                'fasta_sha256': sha256_file(fp),
                'structure_sha256': sha256_file(structure),
                'header': header,
            })

    df = pd.DataFrame(rows)
    if int(df.loc[df['set'].eq('Test17'), 'n_nt'].sum()) != exp_nt:
        raise RuntimeError('Test17 nucleotide count != 583')
    if set(df.loc[df['set'].eq('Train60'), 'chain_key']) & set(df.loc[df['set'].eq('Test17'), 'chain_key']):
        raise RuntimeError('exact chain overlap between Train60 and Test17')

    manifest = out / 'DATASET_MANIFEST.csv'
    df.to_csv(manifest, index=False)

    critical = [
        common / 'test17.fasta',
        common / 'all_label' / 'label' / 'train_label.pkl',
        common / 'all_label' / 'label' / 'test17_labels.pkl',
    ]
    hashes = {str(p.relative_to(vendor)): sha256_file(p) for p in critical if p.exists()}
    json_dump(out / 'BENCHMARK_CRITICAL_SHA256.json', hashes)

    gate = {
        'gate': 'PASS_STAGE07A_TEST17_PREFLIGHT',
        'gate_pass': True,
        'benchmark_commit': commit,
        'train_chains': int((df['set'] == 'Train60').sum()),
        'test_chains': int((df['set'] == 'Test17').sum()),
        'test_nucleotides': int(df.loc[df['set'].eq('Test17'), 'n_nt'].sum()),
        'train_positive_nucleotides': int(df.loc[df['set'].eq('Train60'), 'n_positive'].sum()),
        'test_positive_nucleotides': int(df.loc[df['set'].eq('Test17'), 'n_positive'].sum()),
        'exact_chain_overlap_train_test': 0,
        'primary_RNANet_reopened': False,
        'external_validation_only': True,
    }
    json_dump(out / 'STAGE07A_TEST17_PREFLIGHT_GATE.json', gate)
    print(json.dumps(gate, indent=2))
    print(f'[DONE] {manifest}')


if __name__ == '__main__':
    main()
