#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stage07_common import json_dump, finite_float

METHODS = ['RLBind','RNet','RBind','Rsite','Rsite2','RNAsite','RNABind','MultiModRLBP','MVRBind']
EXPECTED_EPOCHS = {'RLBind': 60, 'RNABind': 30, 'MultiModRLBP': 100}
EXPECTED_REPEATS = {'RLBind': 5, 'RNet': 5, 'RNAsite': 5, 'RNABind': 5, 'MVRBind': 5}
METRICS = ['Accuracy','Precision','Recall','F1','MCC','AUC','AUPR','BACC']


def repeat_count(payload: dict) -> int | None:
    ps = payload.get('per_seed')
    if isinstance(ps, dict):
        lens = [len(v) for v in ps.values() if isinstance(v, list)]
        if lens:
            return min(lens)
    seeds = payload.get('seeds')
    if isinstance(seeds, list):
        return len(seeds)
    cps = payload.get('checkpoints')
    if isinstance(cps, list):
        return len(cps)
    return None


def metric_value(payload: dict, metric: str):
    src = payload.get('summary', payload.get('metrics', {}))
    v = src.get(metric)
    if isinstance(v, dict):
        return v.get('mean')
    return v


def _method_dir(bs: Path, method: str) -> Path | None:
    direct = bs / method
    if direct.is_dir():
        return direct
    for p in bs.iterdir():
        if p.is_dir() and p.name.casefold() == method.casefold():
            return p
    return None


def _metric_evidence_for_method(bs: Path, method: str) -> dict:
    """Best-effort source audit only; never rewrites third-party metric code."""
    root = _method_dir(bs, method)
    if root is None:
        return {
            'method': method,
            'source_root': '',
            'AUPR_definition': 'source_not_located',
            'AUC_definition': 'source_not_located',
            'evidence_files': '',
            'evidence': '',
        }

    hits = []
    aupr_defs = set()
    auc_defs = set()
    for p in sorted(root.rglob('*.py')):
        try:
            text = p.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        low = text.lower()
        rel = p.relative_to(bs).as_posix()
        local = []
        if 'average_precision_score' in low:
            aupr_defs.add('sklearn_average_precision')
            local.append('average_precision_score')
        if 'precision_recall_curve' in low and re.search(r'\bauc\s*\(', low):
            aupr_defs.add('trapezoidal_PR_auc')
            local.append('precision_recall_curve+auc')
        if ('np.trapz' in low or 'numpy.trapz' in low or 'trapezoid' in low) and ('precision' in low and 'recall' in low):
            aupr_defs.add('trapezoidal_PR_auc')
            local.append('PR_trapezoid')
        if 'roc_auc_score' in low:
            auc_defs.add('sklearn_roc_auc')
            local.append('roc_auc_score')
        if local:
            # Preserve only compact line evidence around metric calls.
            ev_lines = []
            for i, line in enumerate(text.splitlines(), start=1):
                ll = line.lower()
                if any(tok in ll for tok in ('average_precision_score','precision_recall_curve','roc_auc_score','trapz','trapezoid')):
                    ev_lines.append(f'{i}:{line.strip()}')
                    if len(ev_lines) >= 8:
                        break
            hits.append((rel, ','.join(local), ' || '.join(ev_lines)))

    if not aupr_defs:
        aupr = 'not_resolved_from_method_source'
    elif len(aupr_defs) == 1:
        aupr = next(iter(aupr_defs))
    else:
        aupr = 'mixed:' + '+'.join(sorted(aupr_defs))
    if not auc_defs:
        auc = 'not_resolved_from_method_source'
    elif len(auc_defs) == 1:
        auc = next(iter(auc_defs))
    else:
        auc = 'mixed:' + '+'.join(sorted(auc_defs))

    return {
        'method': method,
        'source_root': root.relative_to(bs).as_posix(),
        'AUPR_definition': aupr,
        'AUC_definition': auc,
        'evidence_files': ';'.join(x[0] for x in hits),
        'evidence': ' /// '.join(f'{a} [{b}] {c}' for a,b,c in hits),
    }


def audit_metric_implementations(bs: Path, out: Path) -> dict:
    rows = [_metric_evidence_for_method(bs, m) for m in METHODS]
    csv_path = out / 'TEST17_METRIC_IMPLEMENTATION_AUDIT.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    known = [r['AUPR_definition'] for r in rows if not r['AUPR_definition'].startswith(('not_resolved','source_not'))]
    all_known = len(known) == len(METHODS)
    aupr_uniform = bool(all_known and len(set(known)) == 1 and not any(x.startswith('mixed:') for x in known))
    auc_known = [r['AUC_definition'] for r in rows if not r['AUC_definition'].startswith(('not_resolved','source_not'))]
    auc_uniform = bool(len(auc_known) == len(METHODS) and len(set(auc_known)) == 1 and not any(x.startswith('mixed:') for x in auc_known))
    rec = {
        'audit_type': 'runtime_source_metric_implementation_provenance',
        'methods': len(rows),
        'AUPR_all_definitions_resolved': all_known,
        'AUPR_uniform_definition': aupr_uniform,
        'AUPR_definitions_observed': sorted(set(known)),
        'AUC_all_definitions_resolved': len(auc_known) == len(METHODS),
        'AUC_uniform_definition': auc_uniform,
        'AUC_definitions_observed': sorted(set(auc_known)),
        'cross_method_headline_metric': 'AUC',
        'AUPR_reporting_rule': 'supporting_only_unless_uniform_definition_is_confirmed',
        'thresholded_metrics_rule': 'supplementary_only_due_to_method_specific_thresholding',
        'csv': str(csv_path),
    }
    json_dump(out / 'TEST17_METRIC_IMPLEMENTATION_AUDIT.json', rec)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-root', required=True)
    a = ap.parse_args()
    project = Path(a.project_root).resolve()
    out = project / '07_test17_external_benchmark' / 'v1_0_0'
    bs = out / 'vendor' / 'rna-small-molecule-review' / 'BindingSite'
    rows = []
    all_pass = True
    for method in METHODS:
        path = bs / 'results' / method / 'test17_results.json'
        rec = {'method': method, 'result_path': str(path), 'exists': path.exists()}
        ok = path.exists()
        payload = {}
        if ok:
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
            except Exception as e:
                rec['json_error'] = repr(e)
                ok = False
        if ok:
            rec['epochs'] = payload.get('epochs')
            rec['repeat_count'] = repeat_count(payload)
            rec['dataset'] = payload.get('dataset')
            rec['rna_count'] = payload.get('rna_count')
            rec['nucleotide_count'] = payload.get('nucleotide_count', payload.get('canonical_nucleotide_count'))
            rec['evaluated_nucleotide_count'] = payload.get('evaluated_nucleotide_count', rec['nucleotide_count'])
            if payload.get('dataset') not in (None, 'Test17'):
                ok = False; rec['dataset_fail'] = True
            if rec['rna_count'] not in (None, 17):
                ok = False; rec['rna_count_fail'] = True
            if rec['nucleotide_count'] not in (None, 583):
                ok = False; rec['canonical_nt_fail'] = True
            if rec['evaluated_nucleotide_count'] not in (None, 583):
                ok = False; rec['evaluated_nt_fail'] = True
            if method in EXPECTED_EPOCHS:
                ep = payload.get('epochs')
                rec['expected_epochs'] = EXPECTED_EPOCHS[method]
                if ep is None or int(ep) < EXPECTED_EPOCHS[method]:
                    ok = False; rec['full_epoch_fail'] = True
            if method in EXPECTED_REPEATS:
                rc = repeat_count(payload)
                rec['expected_repeats'] = EXPECTED_REPEATS[method]
                if rc is None or int(rc) < EXPECTED_REPEATS[method]:
                    ok = False; rec['repeat_count_fail'] = True
            missing_metrics = [m for m in ('AUC','AUPR') if not finite_float(metric_value(payload,m))]
            if missing_metrics:
                ok = False; rec['missing_primary_metrics'] = missing_metrics
        rec['full_run_pass'] = bool(ok)
        rows.append(rec)
        all_pass = all_pass and ok
        print(f"[{'PASS' if ok else 'FAIL'}] {method} {rec}")

    metric_audit = audit_metric_implementations(bs, out)
    print('[METRIC AUDIT]', json.dumps(metric_audit, sort_keys=True))

    gate = {
        'gate': 'PASS_STAGE07C_COMPARATOR_FULL_RUNS' if all_pass else 'FAIL_STAGE07C_COMPARATOR_FULL_RUNS',
        'gate_pass': bool(all_pass),
        'methods_expected': len(METHODS),
        'methods_full_pass': int(sum(r['full_run_pass'] for r in rows)),
        'methods': rows,
        'smoke_runs_allowed_in_final_table': False,
        'metric_implementation_audit': metric_audit,
        'cross_method_headline_metric': 'AUC',
        'benchmark_native_AUPR_is_supporting': True,
    }
    json_dump(out / 'STAGE07C_COMPARATOR_AUDIT.json', gate)
    print(json.dumps(gate, indent=2))
    if not all_pass:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
