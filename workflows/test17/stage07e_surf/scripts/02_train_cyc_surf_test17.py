#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, balanced_accuracy_score

METRICS = ["Accuracy","Precision","Recall","F1","MCC","AUC","AUPR","BACC"]

def sha256_file(path: Path):
    h=hashlib.sha256();
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def evaluate(y, score, threshold=.5):
    pred=(score>=threshold).astype(int)
    return {
        "Accuracy":float(accuracy_score(y,pred)), "Precision":float(precision_score(y,pred,zero_division=0)),
        "Recall":float(recall_score(y,pred,zero_division=0)), "F1":float(f1_score(y,pred,zero_division=0)),
        "MCC":float(matthews_corrcoef(y,pred)), "AUC":float(roc_auc_score(y,score)),
        "AUPR":float(average_precision_score(y,score)), "BACC":float(balanced_accuracy_score(y,pred)),
    }

def make_model(seed, device, cfg):
    import xgboost as xgb
    return xgb.XGBClassifier(n_estimators=int(cfg['n_estimators']), learning_rate=float(cfg['learning_rate']), max_depth=int(cfg['max_depth']),
        subsample=float(cfg['subsample']), colsample_bytree=float(cfg['colsample_bytree']), reg_lambda=float(cfg['reg_lambda']),
        min_child_weight=float(cfg['min_child_weight']), tree_method=str(cfg['tree_method']), device=device, eval_metric=str(cfg['eval_metric']),
        random_state=int(seed), early_stopping_rounds=int(cfg['early_stopping_rounds']), n_jobs=max(1,int(os.environ.get('THREADS','16'))))

def summary(per):
    return {m:{"mean":float(np.mean(per[m])),"std":float(np.std(per[m],ddof=0))} for m in METRICS}

def dump(path,obj): path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project-root',required=True); ap.add_argument('--tool-root',required=True); a=ap.parse_args()
    project=Path(a.project_root).resolve(); out=project/'07_test17_external_benchmark'/'v1_0_0'; edir=out/'posthoc_surf_v1_0_0'
    eg=json.loads((edir/'STAGE07E_SURF_EXTRACT_GATE.json').read_text())
    if not eg.get('gate_pass'): raise RuntimeError('Stage07E SURF extraction gate missing/false')
    matrix=edir/'rnanet_test17_cyc_surf_matrix.csv.gz'
    if sha256_file(matrix)!=eg['cyc_surf_matrix_sha256']: raise RuntimeError('CY-C+SURF matrix hash drift')
    contract=json.loads((project/'stage07_benchmark'/'config'/'stage07_test17_contract.json').read_text())
    seeds=[int(x) for x in contract['rnanet']['seeds']]; cfg=contract['rnanet']['xgboost']; device=os.environ.get('XGB_DEVICE','cuda')
    d=pd.read_csv(matrix); features=[x.strip() for x in (edir/'CY-C_SURF_MODEL_FEATURES_103.txt').read_text().splitlines() if x.strip()]
    if len(features)!=103: raise RuntimeError(f'expected 103 features, got {len(features)}')
    train=d[d['set'].eq('Train60')].copy().reset_index(drop=True); test=d[d['set'].eq('Test17')].copy().reset_index(drop=True)
    if len(test)!=583: raise RuntimeError('Test17 row drift')
    splits=pd.read_csv(out/'RNANET_TRAIN60_SPLIT_AUDIT.csv')
    base=json.loads((out/'results'/'RNANet_CY-C'/'test17_results.json').read_text())
    if [int(x) for x in base['seeds']]!=seeds: raise RuntimeError('base CY-C seed drift')
    per={m:[] for m in METRICS}; best=[]; pred_rows=[]; val_aupr=[]
    result_dir=edir/'results'/'RNANet_CY-C+SURF_POSTHOC'; result_dir.mkdir(parents=True,exist_ok=True)
    for seed in seeds:
        s=splits[splits['seed'].eq(seed)]; tr=set(s.loc[s['split'].eq('train'),'chain_key'].astype(str)); va=set(s.loc[s['split'].eq('validation'),'chain_key'].astype(str))
        if not tr or not va or tr & va: raise RuntimeError(f'seed {seed}: invalid frozen split')
        trm=train['chain_key'].astype(str).isin(tr).to_numpy(); vam=train['chain_key'].astype(str).isin(va).to_numpy()
        if int(trm.sum()+vam.sum())!=len(train): raise RuntimeError(f'seed {seed}: split does not cover Train60')
        model=make_model(seed,device,cfg)
        model.fit(train.loc[trm,features],train.loc[trm,'label'].to_numpy(int),eval_set=[(train.loc[vam,features],train.loc[vam,'label'].to_numpy(int))],verbose=False)
        vscore=model.predict_proba(train.loc[vam,features])[:,1]; val_aupr.append(float(average_precision_score(train.loc[vam,'label'].to_numpy(int),vscore)))
        score=model.predict_proba(test[features])[:,1]; met=evaluate(test['label'].to_numpy(int),score,.5)
        for k,v in met.items(): per[k].append(v)
        best.append(int(getattr(model,'best_iteration',cfg['n_estimators']-1))); model.save_model(result_dir/f'model_seed{seed}.json')
        for i,sc in enumerate(score): pred_rows.append({'seed':seed,'chain_key':test.iloc[i]['chain_key'],'ResidueIndex':int(test.iloc[i]['ResidueIndex']),'y_true':int(test.iloc[i]['label']),'score':float(sc)})
        print(f"[SEED] {seed} CY-C+SURF AUPR={met['AUPR']:.6f} AUC={met['AUC']:.6f} val_AUPR={val_aupr[-1]:.6f} best_iteration={best[-1]}")
    pd.DataFrame(pred_rows).to_csv(edir/'RNANET_CYC_SURF_TEST17_PREDICTIONS.csv.gz',index=False,compression='gzip')
    summ=summary(per)
    base_auc=np.array(base['per_seed']['AUC'],float); base_aupr=np.array(base['per_seed']['AUPR'],float)
    new_auc=np.array(per['AUC'],float); new_aupr=np.array(per['AUPR'],float)
    payload={
        'method':'RNANet CY-C+SURF (post-hoc exploratory)','dataset':'Test17','training_set':'Train60','rna_count':17,'nucleotide_count':583,
        'feature_contract':'42D CY-C + 4 nucleotide one-hot + frozen SURF57; no B-factor','model_dim':103,'seeds':seeds,
        'train_validation_splits':'exact reuse of frozen Stage07B RNANET_TRAIN60_SPLIT_AUDIT.csv','test_tuning':False,
        'post_hoc_exploratory':True,'test17_seen_before_arm_definition':True,'eligible_for_primary_external_claim':False,
        'xgboost':cfg,'best_iteration':best,'validation_AUPR':val_aupr,'per_seed':{k:[float(x) for x in v] for k,v in per.items()},'summary':summ,
        'paired_vs_frozen_CY-C':{
            'AUC_delta_per_seed':[float(x) for x in (new_auc-base_auc)],'AUC_delta_mean':float(np.mean(new_auc-base_auc)),'AUC_positive_seeds':int(np.sum(new_auc>base_auc)),
            'AUPR_delta_per_seed':[float(x) for x in (new_aupr-base_aupr)],'AUPR_delta_mean':float(np.mean(new_aupr-base_aupr)),'AUPR_positive_seeds':int(np.sum(new_aupr>base_aupr)),
        }
    }
    dump(result_dir/'test17_results.json',payload)
    gate={
        'gate':'PASS_STAGE07E_POSTHOC_CYC_SURF_TEST17','gate_pass':True,'post_hoc_exploratory':True,'eligible_for_primary_external_claim':False,
        'no_test_tuning_within_arm':True,'same_frozen_Train60_splits_as_Stage07B':True,'test17_rows_each_seed':583,'model_dim':103,
        'CY-C_SURF_AUC_mean':summ['AUC']['mean'],'CY-C_SURF_AUPR_mean':summ['AUPR']['mean'],
        'delta_AUC_vs_frozen_CY-C':payload['paired_vs_frozen_CY-C']['AUC_delta_mean'],'delta_AUPR_vs_frozen_CY-C':payload['paired_vs_frozen_CY-C']['AUPR_delta_mean'],
        'AUC_positive_seeds':payload['paired_vs_frozen_CY-C']['AUC_positive_seeds'],'AUPR_positive_seeds':payload['paired_vs_frozen_CY-C']['AUPR_positive_seeds'],
        'matrix_sha256':sha256_file(matrix),
    }
    dump(edir/'STAGE07E_POSTHOC_CYC_SURF_TEST17_GATE.json',gate); print(json.dumps(gate,indent=2))

if __name__=='__main__': main()
