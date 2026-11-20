#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    f1_score, matthews_corrcoef, precision_score, recall_score,
    roc_auc_score,
)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stage07_common import json_dump, sha256_file

METRICS = ['Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'AUC', 'AUPR', 'BACC']


def evaluate(y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (score >= threshold).astype(int)
    return {
        'Accuracy': float(accuracy_score(y, pred)),
        'Precision': float(precision_score(y, pred, zero_division=0)),
        'Recall': float(recall_score(y, pred, zero_division=0)),
        'F1': float(f1_score(y, pred, zero_division=0)),
        'MCC': float(matthews_corrcoef(y, pred)),
        'AUC': float(roc_auc_score(y, score)),
        'AUPR': float(average_precision_score(y, score)),
        'BACC': float(balanced_accuracy_score(y, pred)),
    }


def split_train60_chains(d: pd.DataFrame, seed: int, val_fraction: float) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    groups = np.array(sorted(d['chain_key'].astype(str).unique()))
    rng = np.random.default_rng(int(seed))
    groups = groups[rng.permutation(len(groups))]
    n_val = max(1, int(round(val_fraction * len(groups))))
    val_groups = set(groups[:n_val].tolist())
    train_groups = set(groups[n_val:].tolist())
    tr = d['chain_key'].isin(train_groups).to_numpy()
    va = d['chain_key'].isin(val_groups).to_numpy()
    return tr, va, sorted(train_groups), sorted(val_groups)


def make_model(seed: int, device: str, cfg: dict):
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=int(cfg['n_estimators']),
        learning_rate=float(cfg['learning_rate']),
        max_depth=int(cfg['max_depth']),
        subsample=float(cfg['subsample']),
        colsample_bytree=float(cfg['colsample_bytree']),
        reg_lambda=float(cfg['reg_lambda']),
        min_child_weight=float(cfg['min_child_weight']),
        tree_method=str(cfg['tree_method']),
        device=device,
        eval_metric=str(cfg['eval_metric']),
        random_state=int(seed),
        early_stopping_rounds=int(cfg['early_stopping_rounds']),
        n_jobs=max(1, int(os.environ.get('THREADS', '16'))),
    )


def select_hard_negatives(train_df: pd.DataFrame, train_score: np.ndarray, topk: int) -> np.ndarray:
    selected = np.zeros(len(train_df), dtype=bool)
    work = train_df[['chain_key', 'label', 'ResidueIndex']].copy().reset_index(drop=True)
    work['score'] = train_score
    for chain, g in work.groupby('chain_key', sort=True):
        neg = g[g['label'].eq(0)].sort_values(['score', 'ResidueIndex'], ascending=[False, True], kind='mergesort')
        chosen = neg.head(topk).index.to_numpy()
        selected[chosen] = True
    return selected


def summarize(per_seed: dict) -> dict:
    out = {}
    for metric in METRICS:
        vals = np.asarray(per_seed[metric], dtype=float)
        out[metric] = {'mean': float(vals.mean()), 'std': float(vals.std(ddof=0))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-root', required=True)
    ap.add_argument('--package-root', required=True)
    a = ap.parse_args()

    project = Path(a.project_root).resolve()
    package = Path(a.package_root).resolve()
    out = project / '07_test17_external_benchmark' / 'v1_0_0'
    gate_path = out / 'STAGE07B_CYC_EXTRACT_GATE.json'
    if not gate_path.exists() or not json.loads(gate_path.read_text()).get('gate_pass'):
        raise RuntimeError('CY-C extraction gate missing or false')

    contract = json.loads((package / 'config' / 'stage07_test17_contract.json').read_text())
    rncfg = contract['rnanet']
    seeds = [int(x) for x in rncfg['seeds']]
    val_fraction = float(rncfg['validation_fraction_of_Train60_chains'])
    hcfg = rncfg['hnm']
    xcfg = rncfg['xgboost']
    topk = int(hcfg['top_hard_negatives_per_training_chain'])
    hweight = float(hcfg['hard_negative_weight'])
    threshold = 0.5
    device = os.environ.get('XGB_DEVICE', 'cuda')

    matrix_path = out / 'rnanet_test17_cyc_matrix.csv.gz'
    d = pd.read_csv(matrix_path)
    features = [x.strip() for x in (out / 'CY-C_MODEL_FEATURES_46.txt').read_text().splitlines() if x.strip()]
    if len(features) != 46:
        raise RuntimeError(f'expected 46 model features, got {len(features)}')

    train60 = d[d['set'].eq('Train60')].copy().reset_index(drop=True)
    test17 = d[d['set'].eq('Test17')].copy().reset_index(drop=True)
    if len(test17) != 583:
        raise RuntimeError(f'Test17 rows={len(test17)} != 583')
    if set(train60['chain_key']) & set(test17['chain_key']):
        raise RuntimeError('Train60/Test17 exact chain overlap')

    model_dirs = {
        'CY-C': out / 'results' / 'RNANet_CY-C',
        'CY-C+HNM': out / 'results' / 'RNANet_CY-C+HNM',
    }
    for p in model_dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    per = {arm: {m: [] for m in METRICS} for arm in model_dirs}
    pred_rows = []
    hnm_rows = []
    split_rows = []
    best_iters = {arm: [] for arm in model_dirs}

    for seed in seeds:
        tr_mask, va_mask, tr_chains, va_chains = split_train60_chains(train60, seed, val_fraction)
        if train60.loc[tr_mask, 'label'].nunique() != 2 or train60.loc[va_mask, 'label'].nunique() != 2:
            raise RuntimeError(f'seed {seed}: train/validation class collapse')
        split_rows.extend([{'seed': seed, 'chain_key': c, 'split': 'train'} for c in tr_chains])
        split_rows.extend([{'seed': seed, 'chain_key': c, 'split': 'validation'} for c in va_chains])

        Xtr = train60.loc[tr_mask, features]
        ytr = train60.loc[tr_mask, 'label'].to_numpy(int)
        Xva = train60.loc[va_mask, features]
        yva = train60.loc[va_mask, 'label'].to_numpy(int)
        Xte = test17[features]
        yte = test17['label'].to_numpy(int)

        # Frozen CY-C arm / HNM stage 1. Test17 is never in fit/eval_set.
        base = make_model(seed, device, xcfg)
        base.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        base_score = base.predict_proba(Xte)[:, 1]
        base_metrics = evaluate(yte, base_score, threshold)
        for k, v in base_metrics.items():
            per['CY-C'][k].append(v)
        best_iters['CY-C'].append(int(getattr(base, 'best_iteration', xcfg['n_estimators'] - 1)))
        base.save_model(model_dirs['CY-C'] / f'model_seed{seed}.json')

        train_subset = train60.loc[tr_mask].copy().reset_index(drop=True)
        train_stage1_score = base.predict_proba(train_subset[features])[:, 1]
        hard = select_hard_negatives(train_subset, train_stage1_score, topk)
        weights = np.ones(len(train_subset), dtype=float)
        weights[hard] = hweight
        for idx in np.where(hard)[0]:
            rr = train_subset.iloc[idx]
            hnm_rows.append({
                'seed': seed,
                'chain_key': rr['chain_key'],
                'ResidueIndex': int(rr['ResidueIndex']),
                'label': int(rr['label']),
                'stage1_score': float(train_stage1_score[idx]),
                'weight': hweight,
            })
        if np.any(train_subset.loc[hard, 'label'].to_numpy(int) != 0):
            raise AssertionError('positive residue selected as hard negative')

        hnm = make_model(seed, device, xcfg)
        hnm.fit(
            train_subset[features], train_subset['label'].to_numpy(int),
            sample_weight=weights,
            eval_set=[(Xva, yva)],
            verbose=False,
        )
        hnm_score = hnm.predict_proba(Xte)[:, 1]
        hnm_metrics = evaluate(yte, hnm_score, threshold)
        for k, v in hnm_metrics.items():
            per['CY-C+HNM'][k].append(v)
        best_iters['CY-C+HNM'].append(int(getattr(hnm, 'best_iteration', xcfg['n_estimators'] - 1)))
        hnm.save_model(model_dirs['CY-C+HNM'] / f'model_seed{seed}.json')

        for arm, score in [('CY-C', base_score), ('CY-C+HNM', hnm_score)]:
            for i, s in enumerate(score):
                pred_rows.append({
                    'arm': arm,
                    'seed': seed,
                    'chain_key': test17.iloc[i]['chain_key'],
                    'ResidueIndex': int(test17.iloc[i]['ResidueIndex']),
                    'y_true': int(yte[i]),
                    'score': float(s),
                })

        print('[SEED]', seed,
              'CY-C AUPR=%.6f AUC=%.6f' % (base_metrics['AUPR'], base_metrics['AUC']),
              'CY-C+HNM AUPR=%.6f AUC=%.6f' % (hnm_metrics['AUPR'], hnm_metrics['AUC']),
              f'hard_negatives={int(hard.sum())}')

    pd.DataFrame(pred_rows).to_csv(out / 'RNANET_TEST17_PREDICTIONS.csv.gz', index=False, compression='gzip')
    pd.DataFrame(hnm_rows).to_csv(out / 'RNANET_HNM_SELECTED_TRAINING_NEGATIVES.csv.gz', index=False, compression='gzip')
    pd.DataFrame(split_rows).to_csv(out / 'RNANET_TRAIN60_SPLIT_AUDIT.csv', index=False)

    for arm in ('CY-C', 'CY-C+HNM'):
        payload = {
            'method': f'RNANet {arm}',
            'dataset': 'Test17',
            'rna_count': 17,
            'nucleotide_count': 583,
            'training_set': 'Train60',
            'seeds': seeds,
            'validation_fraction_of_train60_chains': val_fraction,
            'test_tuning': False,
            'threshold_rule': 'fixed_0.5',
            'feature_contract': '42D CY-C + 4 nucleotide one-hot; no B-factor; no SURF',
            'HNM': None if arm == 'CY-C' else {
                'top_negative_per_training_chain': topk,
                'hard_negative_weight': hweight,
                'selection_uses_validation': False,
                'selection_uses_test17': False,
            },
            'xgboost': xcfg,
            'best_iteration': best_iters[arm],
            'per_seed': {k: [float(x) for x in v] for k, v in per[arm].items()},
            'summary': summarize(per[arm]),
        }
        json_dump(model_dirs[arm] / 'test17_results.json', payload)

    gate = {
        'gate': 'PASS_STAGE07B_RNANET_TEST17',
        'gate_pass': True,
        'arms': ['CY-C', 'CY-C+HNM'],
        'seeds': seeds,
        'test17_rows_each_seed': 583,
        'test17_never_used_for_training_validation_hnm_or_threshold_tuning': True,
        'fixed_threshold_for_thresholded_metrics': threshold,
        'primary_external_metric': 'AUPR',
        'secondary_external_metric': 'AUC',
        'CY-C_AUPR_mean': summarize(per['CY-C'])['AUPR']['mean'],
        'CY-C_HNM_AUPR_mean': summarize(per['CY-C+HNM'])['AUPR']['mean'],
        'CY-C_AUC_mean': summarize(per['CY-C'])['AUC']['mean'],
        'CY-C_HNM_AUC_mean': summarize(per['CY-C+HNM'])['AUC']['mean'],
        'primary_RNANet_reopened': False,
        'external_validation_only': True,
        'matrix_sha256': sha256_file(matrix_path),
    }
    json_dump(out / 'STAGE07B_RNANET_TEST17_GATE.json', gate)
    print(json.dumps(gate, indent=2))


if __name__ == '__main__':
    main()
