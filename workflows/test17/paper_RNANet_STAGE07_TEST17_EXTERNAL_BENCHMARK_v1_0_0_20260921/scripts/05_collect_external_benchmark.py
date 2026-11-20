#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stage07_common import json_dump, sha256_file

METHODS = ['RLBind','RNet','RBind','Rsite','Rsite2','RNAsite','RNABind','MultiModRLBP','MVRBind']
RNANET = [('RNANet CY-C','RNANet_CY-C'), ('RNANet CY-C+HNM','RNANet_CY-C+HNM')]
METRICS = ['Accuracy','Precision','Recall','F1','MCC','AUC','AUPR','BACC']


def extract_summary(payload: dict, metric: str):
    src = payload.get('summary', payload.get('metrics', {}))
    v = src.get(metric)
    if isinstance(v, dict):
        return v.get('mean'), v.get('std')
    return v, None


def protocol(payload: dict, method: str) -> str:
    if method.startswith('RNANet'):
        return 'Train60; 10% chain-level validation; 5 frozen RNANet seeds; fixed Test17'
    epochs = payload.get('epochs')
    seeds = payload.get('seeds')
    if epochs is not None and isinstance(seeds, list):
        return f'benchmark full configured run; epochs={epochs}; repeats={len(seeds)}'
    if epochs is not None:
        return f'benchmark full configured run; epochs={epochs}'
    ps = payload.get('per_seed')
    if isinstance(ps, dict):
        lens = [len(v) for v in ps.values() if isinstance(v, list)]
        if lens:
            return f'benchmark full configured run; repeats={min(lens)}'
    return 'benchmark deterministic/pretrained full evaluation'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-root', required=True)
    a = ap.parse_args()
    project = Path(a.project_root).resolve()
    out = project / '07_test17_external_benchmark' / 'v1_0_0'

    for gate_name in ('STAGE07A_TEST17_PREFLIGHT_GATE.json','STAGE07B_RNANET_TEST17_GATE.json','STAGE07C_COMPARATOR_AUDIT.json'):
        gp = out / gate_name
        if not gp.exists() or not json.loads(gp.read_text()).get('gate_pass'):
            raise RuntimeError(f'required PASS gate missing/false: {gp}')

    bs = out / 'vendor' / 'rna-small-molecule-review' / 'BindingSite'
    rows = []
    for method in METHODS:
        p = bs / 'results' / method / 'test17_results.json'
        payload = json.loads(p.read_text(encoding='utf-8'))
        row = {
            'method': payload.get('method', method),
            'method_key': method,
            'source': 'external_benchmark_full_run',
            'protocol': protocol(payload, method),
            'rna_count': payload.get('rna_count', 17),
            'nucleotide_count': payload.get('evaluated_nucleotide_count', payload.get('nucleotide_count', payload.get('canonical_nucleotide_count', 583))),
        }
        for m in METRICS:
            mean, std = extract_summary(payload, m)
            row[f'{m}_mean'] = mean
            row[f'{m}_std'] = std
        rows.append(row)

    for display, key in RNANET:
        p = out / 'results' / key / 'test17_results.json'
        payload = json.loads(p.read_text(encoding='utf-8'))
        row = {
            'method': display,
            'method_key': key,
            'source': 'RNANet_external_validation',
            'protocol': protocol(payload, display),
            'rna_count': payload.get('rna_count', 17),
            'nucleotide_count': payload.get('nucleotide_count', 583),
        }
        for m in METRICS:
            mean, std = extract_summary(payload, m)
            row[f'{m}_mean'] = mean
            row[f'{m}_std'] = std
        rows.append(row)

    df = pd.DataFrame(rows)
    csv = out / 'TEST17_11_METHOD_COMPARISON.csv'
    df.to_csv(csv, index=False)

    formatted = df[['method','source','AUPR_mean','AUPR_std','AUC_mean','AUC_std','MCC_mean','F1_mean','Precision_mean','Recall_mean','protocol']].copy()
    for metric in ('AUPR','AUC'):
        formatted[f'{metric}_report'] = formatted.apply(
            lambda r: (f"{r[f'{metric}_mean']:.4f} ± {r[f'{metric}_std']:.4f}" if pd.notna(r[f'{metric}_std']) else f"{r[f'{metric}_mean']:.4f}"), axis=1
        )
    formatted.to_csv(out / 'TEST17_11_METHOD_MANUSCRIPT_TABLE.csv', index=False)

    rn = df[df['source'].eq('RNANet_external_validation')].set_index('method')
    cy = rn.loc['RNANet CY-C']
    hn = rn.loc['RNANet CY-C+HNM']
    md = f'''# Stage07 Test17 external benchmark — manuscript-ready facts\n\nExternal validation was performed on the annotation-corrected Test17 benchmark (17 RNA chains; 583 nucleotides) without reopening the frozen RNANet primary analysis. The frozen RNANet CY-C representation was evaluated using Train60 and five prespecified RNANet seeds, with Test17 excluded from training, validation, hard-negative selection and threshold tuning.\n\n- RNANet CY-C: AUPR {cy.AUPR_mean:.4f} ± {cy.AUPR_std:.4f}; AUC {cy.AUC_mean:.4f} ± {cy.AUC_std:.4f}.\n- RNANet CY-C+HNM: AUPR {hn.AUPR_mean:.4f} ± {hn.AUPR_std:.4f}; AUC {hn.AUC_mean:.4f} ± {hn.AUC_std:.4f}.\n- HNM external AUPR increment: {hn.AUPR_mean-cy.AUPR_mean:+.4f}.\n\nAUC is used as the main cross-method headline because it is the most consistently defined threshold-independent metric across the benchmark runners. The benchmark-native AUPR field is retained as supporting evidence and its implementation provenance is audited separately; precision/recall/F1/MCC remain supplementary because the legacy methods use heterogeneous thresholding/model-selection rules.\n'''
    (out / 'MANUSCRIPT_READY_EXTERNAL_BENCHMARK.md').write_text(md, encoding='utf-8')

    metric_audit_path = out / 'TEST17_METRIC_IMPLEMENTATION_AUDIT.json'
    if not metric_audit_path.exists():
        raise RuntimeError(f'metric implementation audit missing: {metric_audit_path}')
    metric_audit = json.loads(metric_audit_path.read_text(encoding='utf-8'))

    gate = {
        'gate': 'PASS_STAGE07D_TEST17_11_METHOD_EXTERNAL_BENCHMARK',
        'gate_pass': True,
        'test17_chains': 17,
        'test17_nucleotides': 583,
        'comparator_methods': 9,
        'rnanet_arms': 2,
        'total_rows_in_table': int(len(df)),
        'RNANet_primary_external_metric': 'AUPR',
        'cross_method_headline_metric': 'AUC',
        'benchmark_native_AUPR_supporting_only': True,
        'AUPR_uniform_definition_confirmed': bool(metric_audit.get('AUPR_uniform_definition', False)),
        'metric_implementation_audit': str(metric_audit_path),
        'primary_RNANet_reopened': False,
        'external_validation_only': True,
        'comparison_csv_sha256': sha256_file(csv),
    }
    json_dump(out / 'STAGE07D_FINAL_EXTERNAL_BENCHMARK_GATE.json', gate)
    print(json.dumps(gate, indent=2))
    print('\n=== RNANet external results ===')
    print(df[df['source'].eq('RNANet_external_validation')][['method','AUPR_mean','AUPR_std','AUC_mean','AUC_std']].to_string(index=False))


if __name__ == '__main__':
    main()
