#!/usr/bin/env python3
"""Descriptive fixed-rank-budget structural retrieval comparison. No P-value fishing."""
import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np

def pairkey(a,b):return '|'.join(sorted((str(a),str(b))))

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate-dir',required=True);p.add_argument('--align',required=True);p.add_argument('--out',required=True);p.add_argument('--frozen-c',default=None);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    aln=pd.read_csv(args.align)
    if not {'pair_key','tm_min','coverage_min','usalign_ok'}.issubset(aln.columns):raise RuntimeError('Missing alignment columns')
    if (aln.usalign_ok!=1).any():raise RuntimeError('Incomplete US-align results')
    aln=aln.drop_duplicates('pair_key').set_index('pair_key')
    modes=['C_ONLY','FM_ONLY','C_FM'];rows=[];details=[];pairsets={}
    min_available=min(len(pd.read_csv(Path(args.candidate_dir)/f'candidates_{m}.csv')) for m in modes)
    evaluated_k=[k for k in (20,50,100) if min_available>=k]
    if not evaluated_k:raise RuntimeError(f'No common Top20 budget (minimum arm size {min_available}); report HOLD')
    for mode in modes:
        cand=pd.read_csv(Path(args.candidate_dir)/f'candidates_{mode}.csv')
        cand['pair_key']=[pairkey(a,b) for a,b in zip(cand.RNA_A,cand.RNA_B)]
        merged=cand.join(aln[['tm_min','coverage_min','rmsd','aligned_length']],on='pair_key',how='left',validate='one_to_one')
        if not merged.tm_min.notna().all():raise RuntimeError(f'Incomplete alignment mapping: {mode}')
        pairsets[mode]=set(merged.pair_key[:50])
        details.append(merged)
        for k in evaluated_k:
            x=merged.head(k)
            if len(x)<k:raise RuntimeError(f'{mode}: fewer than {k} eligible candidates; cannot compare budget without reporting it')
            rec={'arm':mode,'K':k,'n':len(x),'median_tm_min':float(x.tm_min.median()),'mean_tm_min':float(x.tm_min.mean()),'median_coverage_min':float(x.coverage_min.median())}
            for t in (0.35,0.40,0.42,0.45):
                good=(x.tm_min>=t)&(x.coverage_min>=0.5)
                rec[f'hits_TM_{str(t).replace(".","p")}_cov50']=int(good.sum())
                rec[f'precision_TM_{str(t).replace(".","p")}_cov50']=float(good.mean())
            rows.append(rec)
    summary=pd.DataFrame(rows)
    summary.to_csv(out/'TopK_comparison.csv',index=False)
    pd.concat(details,ignore_index=True).to_csv(out/'all_candidate_structural_detail.csv',index=False)
    overlaps={f'{a}__{b}':{'top50_shared':len(pairsets[a]&pairsets[b]),'top50_jaccard':len(pairsets[a]&pairsets[b])/len(pairsets[a]|pairsets[b])} for a,b in [('C_ONLY','FM_ONLY'),('C_ONLY','C_FM'),('FM_ONLY','C_FM')]}
    info={'status':'PASS_FIXED_BUDGET_STRUCTURAL_COMPARISON','common_evaluated_K':evaluated_k,'min_available_candidates_per_arm':min_available,'threshold_policy':'descriptive thresholds, not an independent labeled recall benchmark','matched_null_significance':'NOT_PERFORMED','overlaps':overlaps,'fixed_budget':100,'raw_result_rows':len(aln)}
    if args.frozen_c:
        fr=pd.read_csv(args.frozen_c)
        info['frozen_stage04b_C_n']=len(fr)
        info['frozen_stage04b_C_note']='These 90 pairs used additional geometry/FM gates and are NOT the controlled C_FM arm.'
    (out/'comparison_gate.json').write_text(json.dumps(info,indent=2)+'\n')
    print(summary.to_string(index=False));print('[PASS] fixed-rank-budget comparison',flush=True)

if __name__=='__main__':main()
