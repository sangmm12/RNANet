"""Frozen grouping contracts for PDB-disjoint primary and chain-disjoint sensitivity."""
from __future__ import annotations
import numpy as np, pandas as pd

def grouped_split(df: pd.DataFrame, group_col: str, seed: int, test_frac=.20, val_frac_of_remainder=.10):
    groups=np.array(sorted(df[group_col].astype(str).unique()))
    rng=np.random.default_rng(int(seed)); groups=groups[rng.permutation(len(groups))]
    ntest=max(1,int(round(test_frac*len(groups)))); test=set(groups[:ntest])
    rem=groups[ntest:]; nval=max(1,int(round(val_frac_of_remainder*len(rem)))) if len(rem)>1 else 0
    val=set(rem[:nval]); train=set(rem[nval:])
    g=df[group_col].astype(str)
    split=np.where(g.isin(test),'test',np.where(g.isin(val),'validation','train'))
    return pd.Series(split,index=df.index,name='split')

def audit_disjoint(df: pd.DataFrame, group_col: str, split_col='split'):
    c=df.groupby(group_col)[split_col].nunique()
    bad=c[c>1]
    return {"group_col":group_col,"groups":int(len(c)),"violating_groups":int(len(bad)),"pass":bool(len(bad)==0)}
