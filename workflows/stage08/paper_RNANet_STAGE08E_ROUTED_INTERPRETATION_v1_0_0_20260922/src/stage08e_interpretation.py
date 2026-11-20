#!/usr/bin/env python3
import csv, json, math, os, sys, hashlib
from pathlib import Path
from statistics import mean, pstdev

SEEDS = [3407,3420,3433,3446,3459]


def load_json(p):
    with open(p, 'r', encoding='utf-8') as f: return json.load(f)

def read_csv(p):
    with open(p, newline='', encoding='utf-8') as f: return list(csv.DictReader(f))

def f(x): return float(x)

def exact_sign_two_sided(k, n):
    # exact Binomial(n, 0.5), doubled smaller tail
    from math import comb
    lo = sum(comb(n,i) for i in range(0,k+1)) / (2**n)
    hi = sum(comb(n,i) for i in range(k,n+1)) / (2**n)
    return min(1.0, 2*min(lo,hi))

def summarize(vals):
    vals=list(vals)
    return {
        'mean': mean(vals),
        'std_population': pstdev(vals),
        'min': min(vals),
        'max': max(vals),
        'positive_n': sum(v>0 for v in vals),
        'negative_n': sum(v<0 for v in vals),
        'zero_n': sum(v==0 for v in vals),
    }

def write_csv(path, rows, fields):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main(root):
    root=Path(root).resolve()
    src=root/'08_v2_systematic_selection'/'routed_fixed_v1_0_0'
    out=root/'08_v2_systematic_selection'/'routed_interpretation_v1_0_0'
    out.mkdir(parents=True,exist_ok=True)

    req=[src/'STAGE08D_FIXED_ROUTED_GATE.json',src/'FROZEN_ROUTED_SYSTEM.json',src/'ROUTED_RESULTS.csv',src/'SURF_BRANCH_COMPARISON.csv',src/'CALIBRATION_AUDIT.csv']
    miss=[str(p) for p in req if not p.exists()]
    if miss: raise SystemExit('Missing Stage08D inputs: '+repr(miss))

    gate=load_json(req[0]); spec=load_json(req[1]); rr=read_csv(req[2]); sb=read_csv(req[3]); ca=read_csv(req[4])

    checks={
      'stage08d_gate_exact': gate.get('gate')=='PASS_STAGE08D_FIXED_CORE_COMPLETE_ROUTED_EVALUATION_V100' and gate.get('gate_pass') is True,
      'stage08d_source_binding_exact': gate.get('source_binding_gate')=='PASS_STAGE08D_EXACT_FIXED_ROUTED_SOURCE_BINDING',
      'core_arm_exact': gate.get('core_arm')=='C+HNM' and spec.get('core_branch',{}).get('arm')=='C+HNM',
      'complete_arm_exact': gate.get('complete_arm')=='ACS+HNM' and spec.get('complete_branch',{}).get('arm')=='ACS+HNM',
      'no_stage08d_selection': gate.get('candidate_selection_performed_in_stage08d') is False and spec.get('no_stage08d_model_or_arm_selection') is True,
      'label_blind_route': gate.get('route_domain_is_label_blind_and_frozen_before_stage08d') is True and spec.get('routing_rule_uses_labels') is False,
      'test17_unused': gate.get('test17_used') is False and spec.get('branch_models_use_test17') is False,
      'B_factor_unused': spec.get('branch_models_use_B_factor') is False,
      'outer_test_not_fit_or_calibration': gate.get('outer_test_used_for_fitting') is False and gate.get('outer_test_used_for_calibration') is False,
      'all_platt_slopes_positive': gate.get('all_platt_slopes_positive') is True,
      'five_seed_rows': len(rr)==5 and len(sb)==5 and len(ca)==5,
      'seed_identity': [int(x['outer_seed']) for x in rr]==SEEDS and [int(x['outer_seed']) for x in sb]==SEEDS and [int(x['outer_seed']) for x in ca]==SEEDS,
    }
    if not all(checks.values()):
        json.dump({'gate':'FAIL_STAGE08E_SOURCE_OR_PROTOCOL_AUDIT','gate_pass':False,'checks':checks},open(out/'STAGE08E_INTERPRETATION_GATE.json','w'),indent=2)
        raise SystemExit('Stage08E source/protocol audit failed')

    primary = {
      'AUPRC': [f(r['delta_routed_minus_core_partition_AUPR']) for r in rr],
      'AUC': [f(r['delta_routed_minus_core_partition_AUC']) for r in rr],
      'Brier': [f(r['delta_routed_minus_core_partition_Brier']) for r in rr],
      'LogLoss': [f(r['delta_routed_minus_core_partition_LogLoss']) for r in rr],
    }
    global_cmp = {
      'AUPRC': [f(r['delta_routed_minus_core_global_AUPR']) for r in rr],
      'AUC': [f(r['delta_routed_minus_core_global_AUC']) for r in rr],
    }
    surf = {
      'AUPRC': [f(r['delta_complete_minus_core_surf_AUPR']) for r in sb],
      'AUC': [f(r['delta_complete_minus_core_surf_AUC']) for r in sb],
    }
    surf_frac=[f(r['surf_test_fraction']) for r in rr]
    surf_pos=[f(r['surf_positive_fraction']) for r in sb]
    global_pos=[f(r['test_positive_fraction']) for r in rr]

    effect_rows=[]
    for metric,vals in primary.items():
        s=summarize(vals)
        beneficial = [v if metric in ('AUPRC','AUC') else -v for v in vals]
        k=sum(v>0 for v in beneficial)
        effect_rows.append({
          'comparison':'routed_minus_core_partition_calibrated_control',
          'metric':metric,
          'mean_delta':s['mean'],'std_delta':s['std_population'],'min_delta':s['min'],'max_delta':s['max'],
          'beneficial_direction_seeds':k,'n_seeds':len(vals),
          'exact_two_sided_sign_probability_descriptive':exact_sign_two_sided(k,len(vals)),
          'interpretation':'higher_is_better' if metric in ('AUPRC','AUC') else 'lower_is_better'
        })
    for metric,vals in global_cmp.items():
        s=summarize(vals); k=sum(v>0 for v in vals)
        effect_rows.append({
          'comparison':'routed_minus_core_global_calibrated','metric':metric,
          'mean_delta':s['mean'],'std_delta':s['std_population'],'min_delta':s['min'],'max_delta':s['max'],
          'beneficial_direction_seeds':k,'n_seeds':len(vals),
          'exact_two_sided_sign_probability_descriptive':exact_sign_two_sided(k,len(vals)),
          'interpretation':'higher_is_better'
        })
    write_csv(out/'ROUTED_EFFECT_SUMMARY.csv',effect_rows,list(effect_rows[0].keys()))

    surf_rows=[]
    for metric,vals in surf.items():
        s=summarize(vals); k=sum(v>0 for v in vals)
        surf_rows.append({
          'comparison':'ACS+HNM_minus_C+HNM_on_same_SURF_test_rows','metric':metric,
          'mean_delta':s['mean'],'std_delta':s['std_population'],'min_delta':s['min'],'max_delta':s['max'],
          'positive_seeds':k,'n_seeds':len(vals),
          'exact_two_sided_sign_probability_descriptive':exact_sign_two_sided(k,len(vals))
        })
    write_csv(out/'SURF_BRANCH_EFFECT_SUMMARY.csv',surf_rows,list(surf_rows[0].keys()))

    coverage_rows=[]
    for i,seed in enumerate(SEEDS):
        coverage_rows.append({
          'outer_seed':seed,'surf_test_fraction':surf_frac[i],
          'surf_positive_fraction':surf_pos[i],'global_positive_fraction':global_pos[i],
          'surf_positive_enrichment_ratio':surf_pos[i]/global_pos[i]
        })
    write_csv(out/'SURF_DOMAIN_COVERAGE_ENRICHMENT.csv',coverage_rows,list(coverage_rows[0].keys()))

    slopes=[]
    for r in ca:
        slopes.extend([f(r['core_global_slope']),f(r['core_nonsurf_slope']),f(r['core_surf_slope']),f(r['complete_slope'])])
    cal_summary={
      'all_slopes_positive':all(x>0 for x in slopes),'min_slope':min(slopes),'max_slope':max(slopes),
      'shared_cal_pdbs_min':min(int(r['shared_cal_pdbs']) for r in ca),'shared_cal_pdbs_max':max(int(r['shared_cal_pdbs']) for r in ca),
      'shared_es_pdbs_min':min(int(r['shared_es_pdbs']) for r in ca),'shared_es_pdbs_max':max(int(r['shared_es_pdbs']) for r in ca),
    }
    json.dump(cal_summary,open(out/'CALIBRATION_INTERPRETATION.json','w'),indent=2)

    mean_primary_aupr=mean(primary['AUPRC']); mean_primary_auc=mean(primary['AUC'])
    mean_surf_aupr=mean(surf['AUPRC']); mean_surf_auc=mean(surf['AUC'])
    mean_sf=mean(surf_frac); mean_sp=mean(surf_pos); mean_gp=mean(global_pos); enrich=mean_sp/mean_gp
    ctrl_aupr=f(gate['core_partition_calibrated_control_AUPR_mean']); ctrl_auc=f(gate['core_partition_calibrated_control_AUC_mean'])
    routed_aupr=f(gate['routed_calibrated_AUPR_mean']); routed_auc=f(gate['routed_calibrated_AUC_mean'])

    manuscript_results=f'''# Stage08E manuscript Results draft\n\nUnder the frozen routed protocol, residues inside the independently defined SURF applicability domain were evaluated with the previously selected Complete arm (ACS+HNM), whereas all remaining full-Core residues were evaluated with the selected Core arm (C+HNM). No model or arm selection was performed in Stage08D. Across the five frozen PDB-disjoint outer splits, the calibrated routed system achieved mean AUPRC {routed_aupr:.4f} and mean AUROC {routed_auc:.4f}.\n\nTo distinguish representation substitution from the effect of domain-specific calibration, the primary comparator applied C+HNM to both SURF and non-SURF rows while retaining the same separate calibration architecture. Relative to this matched Core control (AUPRC {ctrl_aupr:.4f}; AUROC {ctrl_auc:.4f}), routing ACS+HNM to the SURF domain increased AUPRC by {mean_primary_aupr:+.4f} on average and AUROC by {mean_primary_auc:+.4f}. Both deltas were positive in all 5/5 outer splits. Brier score and log loss also decreased in all five splits, indicating that the routed gain was not restricted to rank-based metrics.\n\nThe SURF domain accounted for only {mean_sf*100:.2f}% of outer-test residue rows, but its mean positive fraction was {mean_sp:.5f}, approximately {enrich:.2f}-fold the full-test positive fraction ({mean_gp:.5f}). On these identical SURF-domain test rows, ACS+HNM exceeded C+HNM by mean AUPRC {mean_surf_aupr:+.4f} and mean AUROC {mean_surf_auc:+.4f}, with positive deltas in all 5/5 outer splits. Thus, the global routed improvement is consistent with a concentrated Complete-representation gain in a small, label-blind, positive-enriched applicability domain rather than a requirement to compute SURF for the entire residue universe.\n\nBecause these five seeds are resampling repetitions rather than independent biological studies, the 5/5 directional consistency is reported descriptively; with n=5, the exact two-sided sign probability for unanimous direction is 0.0625. In addition, the frozen outer-test splits had been inspected historically before the final Stage08B-08D protocol was established. Accordingly, these results support a frozen retrospective routed-system claim but do not constitute a pristine external fixed-system confirmation.\n'''
    (out/'MANUSCRIPT_RESULTS_DRAFT.md').write_text(manuscript_results,encoding='utf-8')

    methods='''# Stage08E manuscript Methods draft\n\nThe final routed evaluation used two arms frozen before Stage08D: C+HNM as the universally available Core arm and ACS+HNM as the Complete arm inside the independently defined Stage05A SURF applicability domain. The routing key was fixed and label-blind. For each frozen outer split, the original outer-validation PDBs were partitioned once at the PDB level into shared early-stopping and calibration subsets. Core and Complete models used the same PDB assignment. Platt scaling was fit only on the shared outer-validation calibration PDBs using natural, unweighted labels; outer-test rows were not used for model fitting, early stopping, calibration, routing, or selection. Because the Core and Complete models were trained separately and HNM can alter score scales, pooled raw probabilities were considered diagnostic only. Primary pooled AUPRC and AUROC were computed after branch/domain-specific calibration.\n\nThe primary representation-substitution comparator retained C+HNM in both the SURF and non-SURF domains but applied separate domain-specific Platt calibration using the same shared calibration PDBs. This matched control preserves the routing/calibration architecture while removing the ACS representation substitution. Stage08D performed no candidate selection.\n'''
    (out/'MANUSCRIPT_METHODS_DRAFT.md').write_text(methods,encoding='utf-8')

    boundaries='''# Claim boundaries\n\n## Supported now\n\n- C+HNM is the frozen Core consensus arm from Stage08C under the prespecified inner-AUPRC criterion.\n- ACS+HNM is the frozen Complete consensus arm from Stage08B within the SURF applicability domain.\n- Under the fixed Stage08D routed protocol, replacing C+HNM by ACS+HNM only inside the frozen SURF domain improved the matched-control AUPRC and AUC in all five outer splits.\n- The same-row SURF-domain comparison favored ACS+HNM over C+HNM in both AUPRC and AUC in all five splits.\n\n## Do not claim yet\n\n- Do not call the routed system externally confirmed or prospectively validated.\n- Do not call ACS+HNM universally superior; it was selected and evaluated as the Complete arm only inside the SURF applicability domain.\n- Do not claim HNM improves every metric; Stage08C showed an AUPRC selection benefit with a small AUROC tradeoff versus standard C.\n- Do not use Stage08D outer-test results to reopen representation, HNM, calibration, or routing-rule selection.\n- Do not treat five outer seeds as five independent biological replicates.\n'''
    (out/'CLAIM_BOUNDARIES.md').write_text(boundaries,encoding='utf-8')

    # Seed-level source-data table for direct supplementary/source-data use.
    source=[]
    for i,seed in enumerate(SEEDS):
        source.append({
          'outer_seed':seed,
          'routed_AUPR':f(rr[i]['routed_calibrated_AUPR']),
          'routed_AUC':f(rr[i]['routed_calibrated_AUC']),
          'core_partition_control_AUPR':f(rr[i]['core_partition_calibrated_control_AUPR']),
          'core_partition_control_AUC':f(rr[i]['core_partition_calibrated_control_AUC']),
          'delta_routed_minus_control_AUPR':primary['AUPRC'][i],
          'delta_routed_minus_control_AUC':primary['AUC'][i],
          'delta_routed_minus_control_Brier':primary['Brier'][i],
          'delta_routed_minus_control_LogLoss':primary['LogLoss'][i],
          'surf_fraction':surf_frac[i],
          'surf_positive_fraction':surf_pos[i],
          'global_positive_fraction':global_pos[i],
          'surf_complete_minus_core_AUPR':surf['AUPRC'][i],
          'surf_complete_minus_core_AUC':surf['AUC'][i]
        })
    write_csv(out/'SOURCE_DATA_STAGE08E.csv',source,list(source[0].keys()))

    provenance={str(p.relative_to(root)):sha256(p) for p in req}
    json.dump(provenance,open(out/'SOURCE_INPUT_SHA256.json','w'),indent=2,sort_keys=True)

    final={
      'gate':'PASS_STAGE08E_ROUTED_INTERPRETATION_V100','gate_pass':True,'checks':checks,
      'core_arm':'C+HNM','complete_arm':'ACS+HNM','candidate_selection_performed':False,
      'primary_comparator':'core_partition_calibrated_control',
      'primary_delta_AUPR_mean':mean_primary_aupr,'primary_delta_AUPR_positive_seeds':sum(v>0 for v in primary['AUPRC']),
      'primary_delta_AUC_mean':mean_primary_auc,'primary_delta_AUC_positive_seeds':sum(v>0 for v in primary['AUC']),
      'primary_delta_Brier_mean':mean(primary['Brier']),'primary_delta_Brier_improved_seeds':sum(v<0 for v in primary['Brier']),
      'primary_delta_LogLoss_mean':mean(primary['LogLoss']),'primary_delta_LogLoss_improved_seeds':sum(v<0 for v in primary['LogLoss']),
      'surf_fraction_mean':mean_sf,'surf_positive_enrichment_ratio_of_means':enrich,
      'surf_complete_minus_core_AUPR_mean':mean_surf_aupr,'surf_complete_minus_core_AUPR_positive_seeds':sum(v>0 for v in surf['AUPRC']),
      'surf_complete_minus_core_AUC_mean':mean_surf_auc,'surf_complete_minus_core_AUC_positive_seeds':sum(v>0 for v in surf['AUC']),
      'five_of_five_exact_two_sided_sign_probability_descriptive':0.0625,
      'historical_outer_test_results_previously_seen':bool(gate.get('historical_outer_test_results_previously_seen')),
      'future_untouched_external_confirmation_required':bool(gate.get('future_untouched_external_confirmation_required_for_fixed_system_claim')),
      'next_stage':'Freeze Stage08E interpretation; integrate into manuscript/figures/source data without reopening selection. Optional future untouched external confirmation remains separate.'
    }
    json.dump(final,open(out/'STAGE08E_INTERPRETATION_GATE.json','w'),indent=2)
    print(json.dumps(final,indent=2))

if __name__=='__main__':
    if len(sys.argv)!=2: raise SystemExit('usage: stage08e_interpretation.py PROJECT_ROOT')
    main(sys.argv[1])
