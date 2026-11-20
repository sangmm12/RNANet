#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

METRICS=["Accuracy","Precision","Recall","F1","MCC","AUC","AUPR","BACC"]

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def dump(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True)+'\n')

def evaluate(y, score, threshold=.5):
    pred=(score>=threshold).astype(int)
    return {
        "Accuracy":float(accuracy_score(y,pred)),
        "Precision":float(precision_score(y,pred,zero_division=0)),
        "Recall":float(recall_score(y,pred,zero_division=0)),
        "F1":float(f1_score(y,pred,zero_division=0)),
        "MCC":float(matthews_corrcoef(y,pred)),
        "AUC":float(roc_auc_score(y,score)),
        "AUPR":float(average_precision_score(y,score)),
        "BACC":float(balanced_accuracy_score(y,pred)),
    }

def make_model(seed, device, cfg):
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
        n_jobs=max(1,int(os.environ.get('THREADS','16'))),
    )

def select_hard_negatives(train_df: pd.DataFrame, train_score: np.ndarray, topk: int) -> np.ndarray:
    selected=np.zeros(len(train_df), dtype=bool)
    work=train_df[['chain_key','label','ResidueIndex']].copy().reset_index(drop=True)
    work['score']=np.asarray(train_score,float)
    for _,g in work.groupby('chain_key',sort=True):
        neg=g[g['label'].eq(0)].sort_values(['score','ResidueIndex'],ascending=[False,True],kind='mergesort')
        selected[neg.head(topk).index.to_numpy()]=True
    return selected

def summary(per):
    return {m:{'mean':float(np.mean(per[m])),'std':float(np.std(per[m],ddof=0))} for m in METRICS}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project-root',required=True); a=ap.parse_args()
    project=Path(a.project_root).resolve()
    out=project/'07_test17_external_benchmark'/'v1_0_0'
    e=out/'posthoc_surf_v1_0_0'
    f=out/'posthoc_cyc_surf_hnm_v1_0_0'; f.mkdir(parents=True,exist_ok=True)
    pkg=project/'stage07_benchmark'

    eg=json.loads((e/'STAGE07E_SURF_EXTRACT_GATE.json').read_text())
    bg=json.loads((e/'STAGE07E_POSTHOC_CYC_SURF_TEST17_GATE.json').read_text())
    if not eg.get('gate_pass') or not bg.get('gate_pass'):
        raise RuntimeError('Stage07E extraction/training gate missing or false')
    matrix=e/'rnanet_test17_cyc_surf_matrix.csv.gz'
    if sha256_file(matrix)!=eg['cyc_surf_matrix_sha256']:
        raise RuntimeError('Stage07E CY-C+SURF matrix hash drift')
    features=[x.strip() for x in (e/'CY-C_SURF_MODEL_FEATURES_103.txt').read_text().splitlines() if x.strip()]
    if len(features)!=103: raise RuntimeError(f'expected 103 features, got {len(features)}')

    contract=json.loads((pkg/'config'/'stage07_test17_contract.json').read_text())
    cfg=contract['rnanet']['xgboost']; hcfg=contract['rnanet']['hnm']
    seeds=[int(x) for x in contract['rnanet']['seeds']]
    topk=int(hcfg['top_hard_negatives_per_training_chain']); hweight=float(hcfg['hard_negative_weight'])
    device=os.environ.get('XGB_DEVICE','cuda')

    d=pd.read_csv(matrix)
    train=d[d['set'].eq('Train60')].copy().reset_index(drop=True)
    test=d[d['set'].eq('Test17')].copy().reset_index(drop=True)
    if len(train)!=3054 or len(test)!=583: raise RuntimeError(f'row drift train={len(train)} test={len(test)}')
    if int(train['label'].sum())!=950 or int(test['label'].sum())!=191: raise RuntimeError('label-count drift')

    splits=pd.read_csv(out/'RNANET_TRAIN60_SPLIT_AUDIT.csv')
    base=json.loads((e/'results'/'RNANet_CY-C+SURF_POSTHOC'/'test17_results.json').read_text())
    if [int(x) for x in base['seeds']]!=seeds: raise RuntimeError('Stage07E base seed drift')
    if base.get('model_dim')!=103: raise RuntimeError('Stage07E base dimension drift')

    per={m:[] for m in METRICS}; best=[]; pred_rows=[]; hnm_rows=[]; hard_counts=[]; val_aupr=[]
    result=f/'results'/'RNANet_CY-C+SURF+HNM_POSTHOC'; result.mkdir(parents=True,exist_ok=True)

    import xgboost as xgb
    for seed in seeds:
        ss=splits[splits['seed'].eq(seed)]
        tr=set(ss.loc[ss['split'].eq('train'),'chain_key'].astype(str)); va=set(ss.loc[ss['split'].eq('validation'),'chain_key'].astype(str))
        if not tr or not va or tr & va: raise RuntimeError(f'seed {seed}: invalid frozen split')
        trm=train['chain_key'].astype(str).isin(tr).to_numpy(); vam=train['chain_key'].astype(str).isin(va).to_numpy()
        if int(trm.sum()+vam.sum())!=len(train): raise RuntimeError(f'seed {seed}: frozen split does not cover Train60')

        train_sub=train.loc[trm].copy().reset_index(drop=True)
        val=train.loc[vam].copy().reset_index(drop=True)

        # Stage-1 is the exact already-saved Stage07E CY-C+SURF model for this seed.
        mpath=e/'results'/'RNANet_CY-C+SURF_POSTHOC'/f'model_seed{seed}.json'
        if not mpath.exists(): raise RuntimeError(f'missing Stage07E base model {mpath}')
        stage1=xgb.XGBClassifier(); stage1.load_model(mpath)
        trscore=stage1.predict_proba(train_sub[features])[:,1]
        hard=select_hard_negatives(train_sub,trscore,topk)
        if np.any(train_sub.loc[hard,'label'].to_numpy(int)!=0): raise RuntimeError('positive selected as hard negative')
        weights=np.ones(len(train_sub),float); weights[hard]=hweight
        hard_counts.append(int(hard.sum()))
        for idx in np.where(hard)[0]:
            r=train_sub.iloc[idx]
            hnm_rows.append({'seed':seed,'chain_key':str(r['chain_key']),'ResidueIndex':int(r['ResidueIndex']),'label':int(r['label']),'stage1_score':float(trscore[idx]),'weight':hweight})

        model=make_model(seed,device,cfg)
        model.fit(train_sub[features],train_sub['label'].to_numpy(int),sample_weight=weights,
                  eval_set=[(val[features],val['label'].to_numpy(int))],verbose=False)
        vscore=model.predict_proba(val[features])[:,1]
        val_aupr.append(float(average_precision_score(val['label'].to_numpy(int),vscore)))
        score=model.predict_proba(test[features])[:,1]
        met=evaluate(test['label'].to_numpy(int),score,.5)
        for k,v in met.items(): per[k].append(v)
        best.append(int(getattr(model,'best_iteration',cfg['n_estimators']-1)))
        model.save_model(result/f'model_seed{seed}.json')
        for i,sc in enumerate(score):
            pred_rows.append({'seed':seed,'chain_key':str(test.iloc[i]['chain_key']),'ResidueIndex':int(test.iloc[i]['ResidueIndex']),'y_true':int(test.iloc[i]['label']),'score':float(sc)})
        print(f"[SEED] {seed} CY-C+SURF+HNM AUPR={met['AUPR']:.6f} AUC={met['AUC']:.6f} val_AUPR={val_aupr[-1]:.6f} best_iteration={best[-1]} hard_negatives={hard_counts[-1]}")

    pd.DataFrame(pred_rows).to_csv(f/'RNANET_CYC_SURF_HNM_TEST17_PREDICTIONS.csv.gz',index=False,compression='gzip')
    pd.DataFrame(hnm_rows).to_csv(f/'RNANET_CYC_SURF_HNM_SELECTED_TRAINING_NEGATIVES.csv.gz',index=False,compression='gzip')
    summ=summary(per)
    b_auc=np.asarray(base['per_seed']['AUC'],float); b_aupr=np.asarray(base['per_seed']['AUPR'],float)
    n_auc=np.asarray(per['AUC'],float); n_aupr=np.asarray(per['AUPR'],float)
    payload={
        'method':'RNANet CY-C+SURF+HNM (post-hoc exploratory)','dataset':'Test17','training_set':'Train60','rna_count':17,'nucleotide_count':583,
        'feature_contract':'42D CY-C + 4 nucleotide one-hot + frozen SURF57; no B-factor; HNM top20 negatives/training chain x5 weight',
        'model_dim':103,'seeds':seeds,'train_validation_splits':'exact reuse of frozen Stage07B RNANET_TRAIN60_SPLIT_AUDIT.csv',
        'test_tuning':False,'post_hoc_exploratory':True,'test17_seen_before_arm_definition':True,'eligible_for_confirmatory_external_claim':False,
        'HNM':{'top_negative_per_training_chain':topk,'hard_negative_weight':hweight,'selection_uses_validation':False,'selection_uses_test17':False,'stage1_model':'exact saved Stage07E CY-C+SURF model per seed'},
        'xgboost':cfg,'best_iteration':best,'hard_negative_count':hard_counts,'validation_AUPR':val_aupr,
        'per_seed':{k:[float(x) for x in v] for k,v in per.items()},'summary':summ,
        'paired_vs_CY-C+SURF':{
            'AUC_delta_per_seed':[float(x) for x in n_auc-b_auc],'AUC_delta_mean':float(np.mean(n_auc-b_auc)),'AUC_positive_seeds':int(np.sum(n_auc>b_auc)),
            'AUPR_delta_per_seed':[float(x) for x in n_aupr-b_aupr],'AUPR_delta_mean':float(np.mean(n_aupr-b_aupr)),'AUPR_positive_seeds':int(np.sum(n_aupr>b_aupr)),
        },
        'matrix_sha256':sha256_file(matrix),
    }
    dump(result/'test17_results.json',payload)
    gate={
        'gate':'PASS_STAGE07F_POSTHOC_CYC_SURF_HNM_TEST17','gate_pass':True,'post_hoc_exploratory':True,
        'eligible_for_confirmatory_external_claim':False,'no_test_tuning_within_arm':True,'same_frozen_Train60_splits_as_Stage07B':True,
        'same_frozen_HNM_rule_as_Stage07B':True,'test17_rows_each_seed':583,'model_dim':103,
        'CY-C_SURF_HNM_AUC_mean':summ['AUC']['mean'],'CY-C_SURF_HNM_AUPR_mean':summ['AUPR']['mean'],
        'delta_AUC_vs_CY-C_SURF':payload['paired_vs_CY-C+SURF']['AUC_delta_mean'],'delta_AUPR_vs_CY-C_SURF':payload['paired_vs_CY-C+SURF']['AUPR_delta_mean'],
        'AUC_positive_seeds':payload['paired_vs_CY-C+SURF']['AUC_positive_seeds'],'AUPR_positive_seeds':payload['paired_vs_CY-C+SURF']['AUPR_positive_seeds'],
        'matrix_sha256':sha256_file(matrix),
    }
    dump(f/'STAGE07F_POSTHOC_CYC_SURF_HNM_TEST17_GATE.json',gate)
    print(json.dumps(gate,indent=2))

if __name__=='__main__': main()
