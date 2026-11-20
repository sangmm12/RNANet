#!/usr/bin/env python3
"""Strict read-only Stage08D Core C+HNM frozen prediction rotation audit.

Uses actual Stage05A 96-column matrix and Stage08D saved TEST arrays. Never
trains, changes a split, writes a model, refits a preprocessor, or modifies
frozen source artifacts. Paired rotations are only scored after baseline replay.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

SEEDS = (3407, 3420, 3433, 3446, 3459)
KS = (4, 8, 12, 16, 20, 24, 28, 32)
FAMILIES = ('integrability_err', 'kahler_err', 'omega_norm', 'logdetG', 'ricci_proxy')
CYC_NAMES = [f'{family}_k{k}' for k in KS for family in FAMILIES]
EXPECTED_CYC_SRC = 'a4be2d6d040f43c3eb42d98af27ac201e37520bbe5a6d2156f0ed57167a18c29'
EXPECTED_STAGE08D_CORE_SHA256 = {
    3407: '8ea00292cdcb102b4297d5fdcf88df83da5e1e90f5fd965504712cf34487753d',
    3420: '62f442e2d3de4415f48fb611573894f112849542035564af785b5da1ed3d907e',
    3433: 'db2b2811ba0343eeb78aff8df992b78460afb1944a545674e1476710ee93476a',
    3446: '61c6a87fccad2e5c92dd78b2d1d1e5ac60ac71b45859589b722107049ece59c0',
    3459: '411a12e15ee1b5e7cf6970adc9531793810e2207e0ea461bcbce697d89f0ba6f',
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def dump_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False, default=str) + '\n')


def loaded_cyc():
    p = Path(__file__).resolve().parent / 'frozen_step1_cyc_RNA_chainlocal.py'
    if sha256(p) != EXPECTED_CYC_SRC:
        raise RuntimeError('HOLD_EXTRACTOR_HASH_MISMATCH')
    sp = importlib.util.spec_from_file_location('cyc_frozen05a', p)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def proper_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    sg = np.sign(np.diag(r))
    sg[sg == 0] = 1
    q = q * sg[np.newaxis, :]
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    assert np.linalg.det(q) > 0.99999999
    return q


def transform_coords(rr, Q=None, t=None, center=None):
    # Transform finite atomic coordinates first; retain missing atoms as NaN.
    # This replays the frozen embedding() missing-atom policy *after* rotation.
    result=[]
    for residue in rr:
        new=dict(residue)
        coords=np.asarray(residue['coords'],float).copy()
        valid=np.all(np.isfinite(coords),axis=1)
        if Q is not None:
            coords[valid]=(coords[valid]-center)@Q.T+center
        if t is not None:
            coords[valid]+=t
        new['coords']=coords
        new['c4']=coords[1].copy()
        result.append(new)
    return result


def cyc_feature_matrix(cyc, X):
    f = cyc.local_features(X, KS, tan_dim=4, eps=1e-6)
    return np.stack([f[k][family] for k in KS for family in FAMILIES], axis=1).astype(np.float32)


def source_path_for_chain(source_dir, pdb, chain):
    # Stage05A prepare_full_core_input.py explicitly constructs
    # input_pdb/{pdb.lower()}_{case_sensitive_chain}.pdb symlinks.
    exact=source_dir / f'{pdb.lower()}_{chain}.pdb'
    if not exact.exists():
        return None, 'EXACT_STAGE05A_CHAIN_SYMLINK_MISSING'
    if not exact.is_symlink():
        return None, 'STAGE05A_SOURCE_NOT_SYMLINK'
    return exact, 'OK'


def fixed_model_input(X, base, rows, core_cols):
    rows = np.asarray(rows, np.int64)
    values = np.asarray(X[rows[:, None], np.asarray(core_cols, np.int64)[None, :]], dtype=np.float32)
    b = np.asarray(base[rows], dtype=np.int64)
    if b.size and (b.min() < 0 or b.max() > 3):
        raise RuntimeError('HOLD_BASE_U8_OUTSIDE_ONE_HOT_RANGE')
    oh = np.eye(4, dtype=np.float32)[b]
    features = np.ascontiguousarray(np.column_stack((values, oh)), dtype=np.float32)
    if features.shape[1] != 46:
        raise RuntimeError('HOLD_MODEL_DIM_NOT_46')
    return features


def summary_metrics(y, p):
    if len(np.unique(y)) != 2:
        return {'auprc': None, 'auroc': None}
    return {'auprc': float(average_precision_score(y, p)), 'auroc': float(roc_auc_score(y, p))}


def rotation_stats(ref, curr, y, group, top_k=5):
    delta = np.abs(curr - ref)
    m = {'mean_abs_delta_p': float(np.mean(delta)),
         'median_abs_delta_p': float(np.median(delta)),
         'p95_abs_delta_p': float(np.quantile(delta, .95)),
         'max_abs_delta_p': float(np.max(delta)),
         'fraction_abs_delta_gt_0p01': float(np.mean(delta > .01)),
         'fraction_abs_delta_gt_0p05': float(np.mean(delta > .05))}
    scores0 = summary_metrics(y, ref)
    scores1 = summary_metrics(y, curr)
    for nm in ('auprc', 'auroc'):
        m[f'baseline_{nm}'] = scores0[nm]
        m[f'rotated_{nm}'] = scores1[nm]
        m[f'delta_{nm}'] = (scores1[nm] - scores0[nm]) if scores1[nm] is not None else None
    overlap=[];top1=[]
    for key in sorted(set(group)):
        ii=np.flatnonzero(group==key)
        k=min(top_k,len(ii))
        if k<1:continue
        a=np.argsort(-ref[ii],kind='stable')[:k]
        b=np.argsort(-curr[ii],kind='stable')[:k]
        overlap.append(len(set(a)&set(b))/k)
        top1.append(int(a[0]==b[0]))
    m['mean_within_chain_topk_overlap']=float(np.mean(overlap)) if overlap else None
    m['fraction_within_chain_top1_same']=float(np.mean(top1)) if top1 else None
    return m


def check_files(root, seed):
    project=root/'paper_RNANet'
    matrix=project/'05_ligand_full_core/matrix'
    routed=project/'08_v2_systematic_selection/routed_fixed_v1_0_0'
    rel=[matrix/'X_intrinsic.f32',matrix/'base.u8',matrix/'y.u8',
         matrix/'residue_index.i32',matrix/'row_chain_idx.i32',matrix/'row_pdb_idx.i32',
         matrix/'chain_row_ranges.csv',matrix/'matrix_contract.json',
         project/f'05_ligand_full_core/splits/pdb_split_seed{seed}.npy',
         routed/f'models/seed{seed}/core_C_HNM.json',
         routed/f'models/seed{seed}/calibration.json',
         routed/f'predictions/seed{seed}_routed_test.npz',
         routed/'STAGE08D_SOURCE_BINDING.json']
    missing=[str(x) for x in rel if not x.is_file()]
    if missing: raise RuntimeError('HOLD_REQUIRED_FROZEN_FILE_MISSING: '+repr(missing))
    mc=read_json(matrix/'matrix_contract.json')
    if not mc.get('gate_pass') or mc.get('matrix_shape') != [6252187, 96]:
        raise RuntimeError('HOLD_MATRIX_CONTRACT_NOT_FROZEN')
    if mc.get('B_factor_primary') is not False:
        raise RuntimeError('HOLD_B_FACTOR_CONTRACT')
    if mc.get('feature_sets', {}).get('CY-C') != [10,11]+list(range(56,96)):
        raise RuntimeError('HOLD_CY_C_FEATURE_ORDER_CONTRACT')
    if mc.get('feature_names', [])[56:96] != CYC_NAMES:
        raise RuntimeError('HOLD_CY_C_NAMES_CONTRACT')
    source=read_json(routed/'STAGE08D_SOURCE_BINDING.json')
    if source.get('gate_pass') is not True or source.get('full_core_contract_sha256') != sha256(matrix/'matrix_contract.json'):
        raise RuntimeError('HOLD_STAGE08_SOURCE_HASH_BINDING')
    rows,cols=mc['matrix_shape']
    for name,size in [('X_intrinsic.f32', rows*cols*4), ('base.u8', rows), ('y.u8', rows),
                      ('residue_index.i32', rows*4), ('row_pdb_idx.i32', rows*4),('row_chain_idx.i32',rows*4)]:
        if (matrix/name).stat().st_size!=size:
            raise RuntimeError('HOLD_MEMMAP_FILE_SIZE_'+name)
    return project,matrix,routed,mc


def select_candidates(chains, te, n_chains, max_len, seed, source_dir):
    # Chains selected by length and availability alone, never by labels or model scores.
    valid=[]
    te=np.asarray(te,np.int64)
    for row in chains.itertuples(index=False):
        start,end=int(row.row_start),int(row.row_end)
        n=int(end-start)
        if n < 20 or n > max_len:continue
        ia=np.searchsorted(te,start,'left')
        ib=np.searchsorted(te,end,'left')
        if ib - ia != n:continue
        if not (te[ia:ib] == np.arange(start,end,dtype=np.int64)).all():continue
        valid.append({'chain_key':str(row.chain_key),'pdb_id':str(row.pdb_id),
                      'chain_id':str(row.chain_id),'rows':np.arange(start,end,dtype=np.int64),'n_rows':n})
    rng=np.random.default_rng(seed)
    bounds=[(20,80),(81,200),(201,max_len)]
    # Allocate near-equal shares to available strata. A deterministic shuffled
    # backup is used for technical source failures; exclusions are reported.
    buckets=[]
    used=set()
    for lo,hi in bounds:
        group=[r for r in valid if lo<=r['n_rows']<=hi]
        rng.shuffle(group)
        buckets.append(group)
    order=[]
    cap=max(1,n_chains//3)
    for group in buckets:
        for r in group[:cap]:
            order.append(r)
            used.add(r['chain_key'])
    backups=[r for r in valid if r['chain_key'] not in used]
    rng.shuffle(backups)
    order+=backups
    return order, {f'{lo}-{hi}':sum(lo<=x['n_rows']<=hi for x in valid) for lo,hi in bounds},len(valid)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,default=3407)
    p.add_argument('--n-chains',type=int,default=12)
    p.add_argument('--n-rot',type=int,default=3)
    p.add_argument('--max-len',type=int,default=500)
    p.add_argument('--max-source-trials',type=int,default=140)
    p.add_argument('--baseline-atol',type=float,default=2e-6)
    p.add_argument('--feature-atol',type=float,default=1e-4)
    p.add_argument('--feature-rtol',type=float,default=1e-4)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    root=Path(args.root).resolve()
    out=Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    status={'protocol':'RNANET_STAGE08D_FROZEN_CORE_PAIRED_ROTATION_R102',
            'status':'RUNNING','seed':args.seed, 'n_requested_chains':args.n_chains,
            'n_rotations':args.n_rot, 'no_training':True,'no_test_fitting':True,
            'source_root':str(root)}
    def fail(reason):
        status['status']=reason
        dump_json(out/'GATE.json',status)
        raise RuntimeError(reason)
    try:
        project,matrix,routed,contract=check_files(root,args.seed)
        source_dir=project/'05_ligand_full_core/input_pdb'
        if not source_dir.is_dir(): fail('HOLD_STAGE05A_SOURCE_INPUT_VIEW_MISSING')
        status['source_dir']=str(source_dir)
        cyc=loaded_cyc()
        status['extractor_sha256']=EXPECTED_CYC_SRC
        model_file=routed/f'models/seed{args.seed}/core_C_HNM.json'
        if sha256(model_file) != EXPECTED_STAGE08D_CORE_SHA256[args.seed]:
            fail('HOLD_STAGE08D_MODEL_HASH_CHANGED_FROM_V101_INVENTORY')
        model=XGBClassifier()
        model.load_model(str(model_file))
        if model.n_features_in_!=46:fail('HOLD_MODEL_NOT_46_DIM')
        pred_path=routed/f'predictions/seed{args.seed}_routed_test.npz'
        with np.load(pred_path,allow_pickle=False) as z:
            all_test_rows=np.array(z['full_row_index'],dtype=np.int64)
            all_test_raw=np.array(z['core_raw'],dtype=np.float32)
            all_test_global=np.array(z['core_global_calibrated'],dtype=np.float32)
            all_test_label=np.array(z['label'],dtype=np.uint8)
            all_test_resid=np.array(z['residue_index'],dtype=np.int32)
        if len(np.unique(all_test_rows))!=len(all_test_rows) or not (np.diff(all_test_rows)>0).all():
            fail('HOLD_FROZEN_TEST_ROW_ORDER_NOT_UNIQUE_SORTED')
        nrows=contract['matrix_shape'][0]
        X=np.memmap(matrix/'X_intrinsic.f32',np.float32,'r',shape=(nrows,96))
        b=np.memmap(matrix/'base.u8',np.uint8,'r',shape=(nrows,))
        y=np.memmap(matrix/'y.u8',np.uint8,'r',shape=(nrows,))
        resid=np.memmap(matrix/'residue_index.i32',np.int32,'r',shape=(nrows,))
        chains=pd.read_csv(matrix/'chain_row_ranges.csv',dtype={'chain_key':str,'pdb_id':str,'chain_id':str})
        order,strata,neligible=select_candidates(chains,all_test_rows,args.n_chains,args.max_len,20261112,source_dir)
        status['eligible_test_chains']=neligible
        status['eligible_length_strata']=strata
        status['candidate_order_not_selected_by_labels']=True
        selected=[];skipped=[];all_rows=[];features_orig=[];byrot={i:[] for i in range(args.n_rot)}
        controls=[];feature_deltas=[];rot_meta=[]
        rng=np.random.default_rng(20261112)
        for item in order[:args.max_source_trials]:
            if len(selected)>=args.n_chains:break
            key=item['chain_key'];ridx=item['rows'];pdb=item['pdb_id'];chainid=item['chain_id']
            f,src_status=source_path_for_chain(source_dir,pdb,chainid)
            if src_status!='OK':
                skipped.append({'chain':key,'reason':src_status});continue
            try:
                st=cyc.parse_structure(str(f))
                rr=cyc.extract_chains(st,chain_mode=chainid)
                if len(rr)!=1:
                    skipped.append({'chain':key,'reason':'CHAIN_ID_NOT_EXACTLY_ONE'});continue
                rr=rr[0]
                # Missing terminal P/backbone atoms are allowed: frozen embedding()
                # imputes them. Rotate physical, finite atoms *before* embedding.
                XX,_=cyc.embedding(rr)
                ids=np.asarray(resid[ridx],np.int64)
                if (ids<1).any() or (ids>len(rr)).any() or len(np.unique(ids))!=len(ids):
                    skipped.append({'chain':key,'reason':'RESIDUE_INDEX_NOT_MAPPED'});continue
                original=cyc_feature_matrix(cyc,XX)[ids-1]
                frozen=np.asarray(X[ridx,56:96],np.float32)
                if not np.isclose(original,frozen,atol=args.feature_atol,rtol=args.feature_rtol,equal_nan=True).all():
                    diff=np.abs(original.astype(float)-frozen.astype(float))
                    skipped.append({'chain':key,'reason':'BASELINE_FEATURE_MISMATCH',
                                    'max_abs_diff':float(np.nanmax(diff))});continue
                ccount, corder=cyc.contacts(rr,cutoff=10.0)
                if not np.isclose(np.asarray(X[ridx,10:12]),np.column_stack((ccount[ids-1],corder[ids-1])),atol=1e-4,rtol=1e-4).all():
                    skipped.append({'chain':key,'reason':'CONTACT_FEATURE_MISMATCH'});continue
                center=XX.reshape(-1,4,3).mean(axis=(0,1))
                rrtrans=transform_coords(rr,t=np.array([11.25,-7.50,4.75]))
                translated,_=cyc.embedding(rrtrans)
                tt=cyc_feature_matrix(cyc,translated)[ids-1]
                if not np.isclose(original,tt,atol=1e-4,rtol=1e-4).all():
                    skipped.append({'chain':key,'reason':'TRANSLATION_CONTROL_FAILED'});continue
                this_rots=[];this_meta=[]
                finite_atoms=np.asarray([v['coords'] for v in rr],float).reshape(-1,3)
                finite_atoms=finite_atoms[np.isfinite(finite_atoms).all(axis=1)]
                for i in range(args.n_rot):
                    Q=proper_rotation(rng)
                    rrrot=transform_coords(rr,Q=Q,center=center)
                    rotated_atoms=np.asarray([v['coords'] for v in rrrot],float).reshape(-1,3)
                    rotated_atoms=rotated_atoms[np.isfinite(rotated_atoms).all(axis=1)]
                    if len(rotated_atoms)!=len(finite_atoms):
                        raise RuntimeError('HOLD_ROTATION_CHANGED_ATOM_AVAILABILITY')
                    if len(finite_atoms)>2:
                        j=len(finite_atoms)//2
                        if not np.isclose(np.linalg.norm(finite_atoms[0]-finite_atoms[j]),
                                          np.linalg.norm(rotated_atoms[0]-rotated_atoms[j]),rtol=1e-10,atol=1e-8):
                            raise RuntimeError('HOLD_RIGID_ROTATION_PAIRWISE_DISTANCE_CHANGED')
                    Xrot,_=cyc.embedding(rrrot)
                    vals=cyc_feature_matrix(cyc,Xrot)[ids-1]
                    cr,co=cyc.contacts(rrrot,cutoff=10.0)
                    if not np.isclose(np.column_stack((cr[ids-1],co[ids-1])),
                                      np.asarray(X[ridx,10:12]),rtol=1e-4,atol=1e-4).all():
                        raise RuntimeError('HOLD_ROTATION_CONTACTS_CHANGED')
                    this_rots.append(vals)
                    this_meta.append({'chain_key':key,'rotation':i+1,'det':float(np.linalg.det(Q)),
                                      'orthogonality_error':float(np.max(np.abs(Q.T@Q-np.eye(3)))),
                                      **{f'R{a}{c}':float(Q[a,c]) for a in range(3) for c in range(3)}})
                    eps=float(np.max(np.abs(original-vals)))
                    feature_deltas.append({'chain_key':key,'rotation':i+1,'max_abs_CYC40_change':eps})
                selected.append({'chain_key':key,'pdb_id':pdb,'chain_id':chainid,'n_rows':len(ridx),
                                 'source':str(f),'source_sha256':sha256(f),'max_anchor_abs_delta':float(np.max(np.abs(original-frozen)))})
                all_rows.append(ridx)
                features_orig.append(frozen)
                for i,vals in enumerate(this_rots):byrot[i].append(vals)
                rot_meta+=this_meta
                controls.append({'chain_key':key,'identity_control_max_delta':float(np.max(np.abs(original-frozen))),
                                 'translation_max_delta':float(np.max(np.abs(tt-original)))})
            except Exception as ex:
                skipped.append({'chain':key,'reason':'SOURCE_ERROR_'+type(ex).__name__, 'details':str(ex)[:280]})
        pd.DataFrame(selected).to_csv(out/'source_anchor.csv',index=False)
        pd.DataFrame(skipped).to_csv(out/'source_exclusions.csv',index=False)
        pd.DataFrame(controls).to_csv(out/'identity_translation_controls.csv',index=False)
        pd.DataFrame(rot_meta).to_csv(out/'rotation_matrices.csv',index=False)
        pd.DataFrame(feature_deltas).to_csv(out/'rotation_feature_deltas.csv',index=False)
        status.update({'n_selected_chains':len(selected),'n_skipped_source_chains':len(skipped),
                       'selected_chains':[x['chain_key'] for x in selected]})
        if len(selected)<4: fail('HOLD_INSUFFICIENT_VERIFIABLE_FROZEN_TEST_CHAINS')
        rows=np.concatenate(all_rows)
        ordidx=np.argsort(rows,kind='stable')
        rows=rows[ordidx]
        group=np.concatenate([np.full(len(rr),s['chain_key'],object) for rr,s in zip(all_rows,selected)])[ordidx]
        origcyc=np.concatenate(features_orig,axis=0)[ordidx]
        valsrots=[np.concatenate(byrot[i],axis=0)[ordidx] for i in range(args.n_rot)]
        loc=np.searchsorted(all_test_rows,rows)
        if not np.array_equal(all_test_rows[loc],rows):fail('HOLD_FROZEN_TEST_MEMBERSHIP_MISMATCH')
        if not np.array_equal(y[rows],all_test_label[loc]):fail('HOLD_TEST_LABEL_MISMATCH')
        if not np.array_equal(resid[rows],all_test_resid[loc]):fail('HOLD_TEST_RESIDUE_ID_MISMATCH')
        original_X=fixed_model_input(X,b,rows,contract['feature_sets']['CY-C'])
        if not np.isclose(original_X[:,2:42],origcyc,atol=args.feature_atol,rtol=args.feature_rtol,equal_nan=True).all():
            fail('HOLD_ORIGINAL_MODEL_INPUT_CYC_MISMATCH')
        proba0=np.asarray(model.predict_proba(original_X)[:,1],float)
        saved_raw=all_test_raw[loc].astype(float)
        replay_max=float(np.max(np.abs(proba0-saved_raw)))
        status.update({'n_test_rows':len(rows),'model_sha256':sha256(model_file),
                       'frozen_test_npz_sha256':sha256(pred_path),
                       'original_raw_replay_max_abs_error':replay_max,
                       'baseline_tolerance':args.baseline_atol})
        if replay_max>args.baseline_atol:fail('HOLD_BASELINE_FROZEN_PREDICTION_REPLAY_MISMATCH')
        platt=read_json(routed/f'models/seed{args.seed}/calibration.json')
        slope=platt.get('core_global_slope')
        intercept=platt.get('core_global_intercept')
        if slope is None or intercept is None:fail('HOLD_MISSING_FROZEN_CALIBRATION')
        def calibrate(p):
            q=np.clip(np.asarray(p,float),1e-6,1-1e-6)
            z=float(intercept)+float(slope)*np.log(q/(1-q))
            return np.clip(1/(1+np.exp(-np.clip(z,-50,50))),1e-12,1-1e-12)
        pp0=calibrate(proba0)
        cal_replay_max=float(np.max(np.abs(pp0-all_test_global[loc].astype(float))))
        status['original_calibrated_replay_max_abs_error']=cal_replay_max
        if cal_replay_max > args.baseline_atol:
            fail('HOLD_BASELINE_CALIBRATED_REPLAY_MISMATCH')
        scores=[];paired=[]
        sensitive=[j for j,n in enumerate(CYC_NAMES) if n.startswith(('integrability_err','kahler_err'))]
        other=[j for j in range(40) if j not in sensitive]
        for i,fr in enumerate(valsrots):
            if len(fr)!=len(rows):fail('HOLD_ROTATED_ROWS_ALIGNMENT')
            for condition,indices in [('full_rotation',list(range(40))),('two_sensitive_families_only',sensitive),('other_three_families_only',other)]:
                xx=original_X.copy()
                xx[:,2+np.array(indices)]=fr[:,indices]
                pr=np.asarray(model.predict_proba(xx)[:,1],float)
                pp=calibrate(pr)
                ms=rotation_stats(proba0,pr,np.asarray(y[rows]),group,top_k=5)
                cm=rotation_stats(pp0,pp,np.asarray(y[rows]),group,top_k=5)
                scores.append({'rotation':i+1,'condition':condition,'n_chains':len(selected),'n_rows':len(rows),
                               'raw':ms,'global_calibrated':cm})
                paired.append(pd.DataFrame({'full_row_index':rows,'chain_key':group,'residue_index':np.asarray(resid[rows]),
                                            'rotation':i+1,'condition':condition,'label':np.asarray(y[rows]),
                                            'original_raw_probability':proba0,'rotated_raw_probability':pr,
                                            'original_global_calibrated':pp0,'rotated_global_calibrated':pp}))
        pd.DataFrame([{'rotation':x['rotation'],'condition':x['condition'],
                       **{'raw_'+k:v for k,v in x['raw'].items()},
                       **{'cal_'+k:v for k,v in x['global_calibrated'].items()}} for x in scores]).to_csv(out/'prediction_rotation_summary.csv',index=False)
        pd.concat(paired,ignore_index=True).to_csv(out/'paired_prediction_rows.csv',index=False)
        status.update({'status':'PASS_FROZEN_CORE_BASELINE_REPLAY_AND_ROTATION_SCORED',
                       'model_input_dimension':46,
                       'cyc_feature_columns':40,
                       'frozen_model_best_iteration':int(getattr(model,'best_iteration',-1)),
                       'scientific_rotation_invariance':'NOT_ASSUMED',
                       'selection_note':'Technical availability/feature anchor selection only; report all skips; no label or model-score-based inclusion',
                       'prediction_note':'Raw core and frozen global-Platt calibrated probabilities; subset-only metrics, not manuscript full-universe AUPRC',
                       'not_a_retrained_24D_ablation':True})
        dump_json(out/'GATE.json',status)
        print(json.dumps(status,indent=2))
    except Exception as ex:
        if status.get('status')=='RUNNING':
            status['status']='HOLD_'+type(ex).__name__
        status['exception']=str(ex)
        dump_json(out/'GATE.json',status)
        print('[HOLD]',repr(ex),file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(2)

if __name__=='__main__':
    main()
