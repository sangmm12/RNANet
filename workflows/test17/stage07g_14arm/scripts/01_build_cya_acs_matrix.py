#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent
PKG=HERE.parent
sys.path.insert(0,str(PKG/'src'))
from stage07g_common import sha256_file,dumpj,DIMS

CYA_SHA='539f9921151c43e4664b49489a201d7a708559193318afc6c33d31b188cb1d5b'
IO_SHA='0f4554f67d2770598c89491ef1fa28197c57ec5db087700299d042a4096429dc'


def read_fasta(path: Path) -> str:
    return ''.join(x.strip() for x in path.read_text().splitlines() if x.strip() and not x.startswith('>')).upper().replace('T','U')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project-root',required=True); a=ap.parse_args()
    project=Path(a.project_root).resolve()
    base=project/'07_test17_external_benchmark'/'v1_0_0'
    out=base/'exhaustive_14arm_transfer_v1_0_2'; out.mkdir(parents=True,exist_ok=True)
    stage07pkg=project/'stage07_benchmark'

    req=[base/'STAGE07A_TEST17_PREFLIGHT_GATE.json',base/'STAGE07B_R5_ROW_CONTRACT_AUDIT.json',base/'STAGE07B_R5_ROW_CONTRACT_AUDIT.csv',base/'STAGE07B_CYC_EXTRACT_GATE.json',base/'DATASET_MANIFEST.csv',base/'RNANET_TRAIN60_SPLIT_AUDIT.csv']
    for p in req:
        if not p.exists(): raise RuntimeError(f'missing prerequisite: {p}')
    for p in [base/'STAGE07A_TEST17_PREFLIGHT_GATE.json',base/'STAGE07B_R5_ROW_CONTRACT_AUDIT.json',base/'STAGE07B_CYC_EXTRACT_GATE.json']:
        if not json.loads(p.read_text()).get('gate_pass'): raise RuntimeError(f'prerequisite gate false: {p}')

    surfdir=base/'posthoc_surf_v1_0_0'
    sg=surfdir/'STAGE07E_SURF_EXTRACT_GATE.json'; matrix_cs=surfdir/'rnanet_test17_cyc_surf_matrix.csv.gz'
    if not sg.exists() or not json.loads(sg.read_text()).get('gate_pass'): raise RuntimeError('Stage07E SURF extraction gate missing/false')
    sgj=json.loads(sg.read_text())
    if sha256_file(matrix_cs)!=sgj['cyc_surf_matrix_sha256']: raise RuntimeError('Stage07E C+S matrix hash drift')

    row_helper=stage07pkg/'scripts'/'stage07b_row_contract_r5.py'
    if not row_helper.exists(): raise RuntimeError(f'R5 helper missing: {row_helper}; re-apply Stage07B R5 hotfix')
    sys.path.insert(0,str(stage07pkg/'scripts'))
    from stage07b_row_contract_r5 import resolve_chain_contract

    vsrc=PKG.parents[2]/'src'
    if sha256_file(vsrc/'rnanet'/'cya.py')!=CYA_SHA: raise RuntimeError('bundled CY-A source hash mismatch')
    if sha256_file(vsrc/'rnanet'/'io.py')!=IO_SHA: raise RuntimeError('bundled IO source hash mismatch')
    sys.path.insert(0,str(vsrc))
    from rnanet.io import load_rna,parser_for
    from rnanet.cya import compute_cya,SCALES

    manifest=pd.read_csv(base/'DATASET_MANIFEST.csv')
    ra=pd.read_csv(base/'STAGE07B_R5_ROW_CONTRACT_AUDIT.csv')
    digest=dict(zip(ra.chain_key.astype(str),ra.row_contract_digest.astype(str)))
    rows=[]; audits=[]
    for j,rec in manifest.iterrows():
        set_name=str(rec['set']); key=str(rec['chain_key']); chain=str(rec['chain_id']); structure=Path(rec['structure']); seq=read_fasta(Path(rec['fasta']))
        raw=load_rna(str(structure),chain)
        chosen,metas,audit=resolve_chain_contract(structure,chain,seq,parser_for)
        if not audit.get('gate_pass'): raise RuntimeError(f'R5 row contract failed for {key}')
        if str(audit['row_contract_digest'])!=str(digest[key]): raise RuntimeError(f'R5 digest drift for {key}')
        if len(raw)!=len(metas): raise RuntimeError(f'load/parser row drift for {key}')
        res=[raw[i] for i in chosen]
        if len(res)!=len(seq): raise RuntimeError(f'selected row count != FASTA for {key}')
        Q,cc,co,A=compute_cya(res)
        for i,r in enumerate(res):
            z={'set':set_name,'chain_key':key,'ResidueIndex':i+1,
               'a_eta':float(Q[i,0]),'a_theta':float(Q[i,1]),'a_chi':float(Q[i,2]),'a_puckerP':float(Q[i,3]),
               'a_ContactCount':float(cc[i]),'a_ContactOrder':float(co[i])}
            for k in SCALES:
                for n,v in A[k].items(): z[f'a_{n}_k{k}']=float(v[i])
            rows.append(z)
        vals=np.array([[Q[i,0],Q[i,1],Q[i,2],Q[i,3],cc[i],co[i]] for i in range(len(res))],float)
        audits.append({'set':set_name,'chain_key':key,'rows':len(res),'raw6_finite_fraction':float(np.isfinite(vals).mean()),'row_contract_digest':audit['row_contract_digest']})
        print(f'[CY-A] {j+1:02d}/{len(manifest)} {set_name} {key} rows={len(res)}',flush=True)

    adf=pd.DataFrame(rows).sort_values(['set','chain_key','ResidueIndex'],kind='mergesort').reset_index(drop=True)
    pd.DataFrame(audits).to_csv(out/'CYA_CHAIN_EXTRACTION_AUDIT.csv',index=False)
    a_features=['a_ContactCount','a_ContactOrder','a_eta','a_theta','a_chi','a_puckerP']
    for k in SCALES:
        for n in ('lag_mean','lag_p90','sl_abs','phase_disp','pca_eff_dim'):
            a_features.append(f'a_{n}_k{k}')
    if len(a_features)!=46: raise RuntimeError(f'CY-A dim drift: {len(a_features)}')

    d=pd.read_csv(matrix_cs)
    keycols=['set','chain_key','ResidueIndex']
    # Distinguish a true key/merge failure from legitimate structure-derived NaN values.
    # Stage08B/C use the same XGBoost missing-value contract: NaN is allowed natively,
    # while +/-inf is forbidden. Test17 rows are therefore never dropped merely because
    # CY-A cannot be computed for a residue; this preserves the full 583-nt test universe.
    m=d.merge(adf,on=keycols,how='left',validate='one_to_one',indicator='_cya_merge')
    if len(m)!=3637 or int(m['set'].eq('Train60').sum())!=3054 or int(m['set'].eq('Test17').sum())!=583: raise RuntimeError('row-count drift')
    if int(m.loc[m.set.eq('Train60'),'label'].sum())!=950 or int(m.loc[m.set.eq('Test17'),'label'].sum())!=191: raise RuntimeError('label-count drift')
    merge_missing=int((m['_cya_merge']!='both').sum())
    if merge_missing:
        bad=m.loc[m['_cya_merge']!='both',keycols+['_cya_merge']].head(20).to_dict('records')
        raise RuntimeError(f'CY-A key merge failure rows={merge_missing} examples={bad}')
    m=m.drop(columns=['_cya_merge'])

    avec=m[a_features].to_numpy(float)
    if np.isinf(avec).any():
        raise RuntimeError('CY-A matrix contains +/-inf; NaN is allowed by the frozen XGBoost missing-value contract')
    a_specific=[x for x in a_features if x not in ('a_ContactCount','a_ContactOrder')]
    # Missingness is audited, not filtered or imputed. Shared ContactCount/ContactOrder
    # are taken from the already-frozen C matrix below, exactly as in the Stage08 arm contract.
    miss_rows=[]
    for split_name in ('Train60','Test17'):
        z=m['set'].eq(split_name).to_numpy()
        aa=m.loc[z,a_specific].to_numpy(float)
        raw=m.loc[z,a_features].to_numpy(float)
        miss_rows.append({
            'set':split_name,
            'rows':int(z.sum()),
            'all_raw_CYA_features_missing_rows':int(np.isnan(raw).all(axis=1).sum()),
            'all_CYA_specific_features_missing_rows':int(np.isnan(aa).all(axis=1).sum()),
            'CYA_specific_finite_fraction':float(np.isfinite(aa).mean()),
            'CYA_specific_any_missing_rows':int(np.isnan(aa).any(axis=1).sum()),
        })
    miss=pd.DataFrame(miss_rows)
    miss.to_csv(out/'CYA_MISSINGNESS_AUDIT.csv',index=False)

    chain_miss=[]
    for (split_name,key),g in m.groupby(['set','chain_key'],sort=True):
        aa=g[a_specific].to_numpy(float)
        chain_miss.append({
            'set':split_name,'chain_key':key,'rows':len(g),
            'all_CYA_specific_missing_rows':int(np.isnan(aa).all(axis=1).sum()),
            'any_CYA_specific_missing_rows':int(np.isnan(aa).any(axis=1).sum()),
            'CYA_specific_finite_fraction':float(np.isfinite(aa).mean()),
        })
    pd.DataFrame(chain_miss).to_csv(out/'CYA_MISSINGNESS_BY_CHAIN.csv',index=False)

    # Shared contacts are a cross-check only. Compare where both calculations are finite.
    acc=m['a_ContactCount'].to_numpy(float); ccc=m['ContactCount'].to_numpy(float)
    aco=m['a_ContactOrder'].to_numpy(float); cco=m['ContactOrder'].to_numpy(float)
    cc_joint=np.isfinite(acc)&np.isfinite(ccc); co_joint=np.isfinite(aco)&np.isfinite(cco)
    if not cc_joint.any() or not co_joint.any(): raise RuntimeError('no jointly finite shared-contact rows for source cross-check')
    cc_diff=float(np.max(np.abs(acc[cc_joint]-ccc[cc_joint])))
    co_diff=float(np.max(np.abs(aco[co_joint]-cco[co_joint])))
    if cc_diff>1e-12 or co_diff>1e-12: raise RuntimeError(f'shared contact drift cc={cc_diff} co={co_diff}')
    cc_mask_disagree=int(np.sum(np.isfinite(acc)!=np.isfinite(ccc)))
    co_mask_disagree=int(np.sum(np.isfinite(aco)!=np.isfinite(cco)))
    # Use the frozen shared C contact columns; do not duplicate A contact columns in the final matrix.
    m=m.drop(columns=['a_ContactCount','a_ContactOrder'])
    a_unique=['ContactCount','ContactOrder']+[x for x in a_features if x not in ('a_ContactCount','a_ContactOrder')]
    c_features=[x.strip() for x in (base/'CY-C_FEATURES_42.txt').read_text().splitlines() if x.strip()]
    s_features=[x.strip() for x in (surfdir/'SURF_FEATURES_57.txt').read_text().splitlines() if x.strip()]
    nt=['nt_A','nt_C','nt_G','nt_U']
    if len(c_features)!=42 or len(s_features)!=57: raise RuntimeError('C/S dimension drift')
    c_only=[x for x in c_features if x not in ('ContactCount','ContactOrder')]
    defs={
      'A':a_unique+nt,
      'C':c_features+nt,
      'S':s_features+nt,
      'AC':a_unique+c_only+nt,
      'AS':a_unique+s_features+nt,
      'CS':c_features+s_features+nt,
      'ACS':a_unique+c_only+s_features+nt,
    }
    got={k:len(v) for k,v in defs.items()}
    if got!=DIMS: raise RuntimeError(f'arm dimension mismatch: {got} expected={DIMS}')
    for arm,cols in defs.items():
        if len(cols)!=len(set(cols)): raise RuntimeError(f'duplicate feature name in {arm}')
        miss=[x for x in cols if x not in m.columns]
        if miss: raise RuntimeError(f'missing columns in {arm}: {miss[:10]}')
        (out/f'FEATURES_{arm}_{len(cols)}.txt').write_text('\n'.join(cols)+'\n')
    matrix=out/'rnanet_test17_acs_matrix.csv.gz'; m.to_csv(matrix,index=False,compression='gzip')
    gate={
      'gate':'PASS_STAGE07G_CYA_ACS_MATRIX_BUILD','gate_pass':True,'rows_total':len(m),'rows_train60':3054,'rows_test17':583,
      'train_positive':950,'test_positive':191,'cya_intrinsic_dim':46,'cyc_intrinsic_dim':42,'surf_dim':57,
      'arm_dimensions_with_nt_onehot':DIMS,'B_factor_included':False,'identifier_predictors_included':False,
      'missing_value_contract':'NaN preserved and handled natively by XGBoost; no Test17 row filtering or imputation; +/-inf forbidden',
      'cya_missingness_audit_file':'CYA_MISSINGNESS_AUDIT.csv','cya_missingness_by_chain_file':'CYA_MISSINGNESS_BY_CHAIN.csv',
      'shared_ContactCount_joint_finite_rows':int(cc_joint.sum()),'shared_ContactOrder_joint_finite_rows':int(co_joint.sum()),
      'shared_ContactCount_finite_mask_disagreement_rows':cc_mask_disagree,'shared_ContactOrder_finite_mask_disagreement_rows':co_mask_disagree,
      'shared_ContactCount_max_abs_diff':float(cc_diff),'shared_ContactOrder_max_abs_diff':float(co_diff),
      'cya_source_sha256':CYA_SHA,'io_source_sha256':IO_SHA,'source_cyc_surf_matrix_sha256':sha256_file(matrix_cs),'acs_matrix_sha256':sha256_file(matrix),
      'test17_used_for_feature_selection':False,
    }
    dumpj(out/'STAGE07G_CYA_MATRIX_GATE.json',gate); print(json.dumps(gate,indent=2))

if __name__=='__main__': main()
