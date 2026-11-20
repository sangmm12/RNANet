#!/usr/bin/env python3
"""Train/evaluate the frozen XGBoost protocol when the full residue matrix is supplied.

The public release does not redistribute the multi-million-row derived matrices or trained
model binaries. Input must contain label, grouping columns and feature columns. For the
frozen HNM replay, supply the already-frozen training weight column with --weight-col;
this script never derives weights using held-out outcomes.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rnanet.splits import grouped_split,audit_disjoint
from rnanet.metrics import binary_metrics

def read(path): return pd.read_parquet(path) if str(path).endswith('.parquet') else pd.read_csv(path)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('matrix'); ap.add_argument('--features',required=True,help='text file: one feature column per line')
    ap.add_argument('--label',default='label'); ap.add_argument('--group-col',default='pdb_id'); ap.add_argument('--seed',type=int,default=3407)
    ap.add_argument('--weight-col'); ap.add_argument('--device',default='cuda'); ap.add_argument('-o','--output',required=True); a=ap.parse_args()
    import xgboost as xgb
    d=read(a.matrix); feats=[x.strip() for x in Path(a.features).read_text().splitlines() if x.strip()]
    d=d.copy(); d['split']=grouped_split(d,a.group_col,a.seed); aud=audit_disjoint(d,a.group_col)
    tr=d.split.eq('train'); va=d.split.eq('validation'); te=d.split.eq('test')
    m=xgb.XGBClassifier(n_estimators=6000,learning_rate=.03,max_depth=6,subsample=.8,colsample_bytree=.8,reg_lambda=1,
        min_child_weight=1,tree_method='hist',device=a.device,eval_metric='aucpr',random_state=a.seed,early_stopping_rounds=400)
    kw={}
    if a.weight_col: kw['sample_weight']=d.loc[tr,a.weight_col].to_numpy(float)
    m.fit(d.loc[tr,feats],d.loc[tr,a.label],eval_set=[(d.loc[va,feats],d.loc[va,a.label])],verbose=False,**kw)
    s=m.predict_proba(d.loc[te,feats])[:,1]; met=binary_metrics(d.loc[te,a.label],s)
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); pd.DataFrame({'index':d.index[te],'y_true':d.loc[te,a.label],'score':s}).to_csv(out/'test_predictions.csv',index=False)
    m.save_model(out/'model.json'); (out/'metrics.json').write_text(json.dumps({'seed':a.seed,'group_col':a.group_col,'split_audit':aud,**met},indent=2)+'\n')
    print(json.dumps(met,indent=2))
if __name__=='__main__': main()
