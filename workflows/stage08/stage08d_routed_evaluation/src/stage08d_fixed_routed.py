#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

SEEDS = [3407, 3420, 3433, 3446, 3459]
HNM_TOPK = 20
HNM_WEIGHT = 5.0
VAL_CAL_FRACTION = 0.50
FULL_ROWS = 6_252_187
FULL_DIM = 96
SURF_ROWS = 431_226
SURF_DIM = 57
CORE_ARM = "C+HNM"
COMPLETE_ARM = "ACS+HNM"
CORE_DIM = 46
COMPLETE_DIM = 147

PARAMS = dict(
    n_estimators=6000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    min_child_weight=1.0,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    early_stopping_rounds=400,
)


def sha256_file(p: Path, block: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def jload(p: Path):
    return json.loads(Path(p).read_text())


def dumpj(p: Path, obj) -> None:
    Path(p).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def onehot_base(base: np.ndarray) -> np.ndarray:
    b = np.asarray(base, dtype=np.int64)
    if b.size and (b.min() < 0 or b.max() > 3):
        raise SystemExit(f"[FATAL] base.u8 outside 0..3: min={b.min()} max={b.max()}")
    return np.eye(4, dtype=np.float32)[b]


def fit_xgb(Xtr, ytr, Xes, yes, seed, sample_weight=None, device="cuda", threads=32):
    kw = dict(PARAMS)
    kw.update(random_state=int(seed), n_jobs=int(threads), device=device)
    m = XGBClassifier(**kw)
    m.fit(Xtr, ytr, sample_weight=sample_weight, eval_set=[(Xes, yes)], verbose=False)
    return m


def metric_bundle(y, p):
    y = np.asarray(y, dtype=np.uint8)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"AUPR": float("nan"), "AUC": float("nan"), "Brier": float("nan"), "LogLoss": float("nan")}
    return {
        "AUPR": float(average_precision_score(y, p)),
        "AUC": float(roc_auc_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "LogLoss": float(log_loss(y, p, labels=[0, 1])),
    }


def hard_negative_mask(y, pred, chain, topk=20):
    y = np.asarray(y)
    pred = np.asarray(pred)
    chain = np.asarray(chain)
    out = np.zeros(len(y), dtype=bool)
    neg = np.flatnonzero(y == 0)
    if len(neg) == 0:
        return out
    cneg = chain[neg]
    if len(cneg) > 1 and np.any(cneg[1:] < cneg[:-1]):
        order = np.argsort(cneg, kind="mergesort")
        neg = neg[order]
        cneg = cneg[order]
    starts = np.r_[0, np.flatnonzero(cneg[1:] != cneg[:-1]) + 1]
    ends = np.r_[starts[1:], len(neg)]
    for a, b in zip(starts, ends):
        idx = neg[a:b]
        k = min(int(topk), len(idx))
        oo = np.lexsort((idx, -pred[idx]))[:k]
        out[idx[oo]] = True
    return out


def stable_rank(tag: str, seed: int, pdb_index: int) -> int:
    h = hashlib.sha256(f"{tag}|{seed}|{pdb_index}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def split_shared_es_cal_pdbs(val_idx, pdb, y, surf_mask, seed: int, cal_fraction=0.5):
    """Create one shared PDB-disjoint outer-validation ES/calibration split.

    PDBs that contain SURF-domain validation rows are stratified by whether those
    SURF rows contain a positive. PDBs without SURF rows are stratified by their
    full-Core positive status. This creates a single validation partition used by
    both Core and Complete branches, so branch calibration/control comparisons
    are aligned at the PDB level. No outer-test row participates.
    """
    val_idx = np.asarray(val_idx, dtype=np.int64)
    vp = np.asarray(pdb[val_idx], dtype=np.int64)
    uniq, inv = np.unique(vp, return_inverse=True)
    if len(uniq) < 6:
        raise SystemExit(f"[FATAL] too few outer-validation PDBs: {len(uniq)}")

    full_has_pos = np.zeros(len(uniq), dtype=np.uint8)
    np.maximum.at(full_has_pos, inv, np.asarray(y[val_idx], dtype=np.uint8))
    surf_row = np.asarray(surf_mask[val_idx], dtype=bool)
    surf_count = np.zeros(len(uniq), dtype=np.int64)
    np.add.at(surf_count, inv, surf_row.astype(np.int64))
    surf_has_pos = np.zeros(len(uniq), dtype=np.uint8)
    if np.any(surf_row):
        np.maximum.at(
            surf_has_pos,
            inv[surf_row],
            np.asarray(y[val_idx[surf_row]], dtype=np.uint8),
        )

    # Three practical strata are sufficient and deterministic:
    # SURF-present+positive, SURF-present+no-positive, no-SURF (split by full label).
    strata = {}
    for i, pp in enumerate(uniq):
        if surf_count[i] > 0:
            key = f"surf_pos_{int(surf_has_pos[i])}"
        else:
            key = f"nosurf_fullpos_{int(full_has_pos[i])}"
        strata.setdefault(key, []).append(int(pp))

    surf_pos_n = len(strata.get("surf_pos_1", []))
    surf_neg_n = len(strata.get("surf_pos_0", []))
    if surf_pos_n < 2 or surf_neg_n < 2:
        raise SystemExit(
            f"[FATAL] insufficient SURF validation PDB strata for shared ES/calibration: "
            f"surf_positive_pdb={surf_pos_n} surf_no_positive_pdb={surf_neg_n}"
        )

    cal_pdb = set()
    for key, group in sorted(strata.items()):
        ordered = sorted(
            group,
            key=lambda pp: stable_rank(f"RNANET_STAGE08D_SHARED_VAL_CAL_{key}", seed, pp),
        )
        if len(ordered) == 1:
            # Singleton non-SURF strata do not determine Complete class validity;
            # assign deterministically by hash parity.
            if stable_rank(f"RNANET_STAGE08D_SINGLETON_{key}", seed, ordered[0]) % 2 == 0:
                cal_pdb.add(ordered[0])
            continue
        n = max(1, int(round(len(ordered) * float(cal_fraction))))
        n = min(n, len(ordered) - 1)
        cal_pdb.update(ordered[:n])

    all_pdb = set(map(int, uniq))
    es_pdb = all_pdb - cal_pdb
    if not cal_pdb or not es_pdb or cal_pdb & es_pdb:
        raise SystemExit("[FATAL] invalid shared validation PDB partition")
    return np.asarray(sorted(es_pdb), dtype=np.int64), np.asarray(sorted(cal_pdb), dtype=np.int64)


def rows_for_pdb_set(idx, pdb, pdb_set):
    idx = np.asarray(idx, dtype=np.int64)
    mask = np.isin(np.asarray(pdb[idx], dtype=np.int64), np.asarray(pdb_set, dtype=np.int64))
    return idx[mask]

def fit_platt(y, raw_p):
    y = np.asarray(y, dtype=np.uint8)
    if len(np.unique(y)) < 2:
        raise SystemExit("[FATAL] Platt calibration requires both classes")
    p = np.clip(np.asarray(raw_p, dtype=np.float64), 1e-6, 1 - 1e-6)
    x = np.log(p / (1.0 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000, random_state=0)
    lr.fit(x, y)
    return {"slope": float(lr.coef_[0, 0]), "intercept": float(lr.intercept_[0])}


def apply_platt(raw_p, pars):
    p = np.clip(np.asarray(raw_p, dtype=np.float64), 1e-6, 1 - 1e-6)
    x = np.log(p / (1.0 - p))
    z = float(pars["intercept"]) + float(pars["slope"]) * x
    z = np.clip(z, -50.0, 50.0)
    q = 1.0 / (1.0 + np.exp(-z))
    return np.clip(q, 1e-12, 1.0 - 1e-12)


def core_feature_contract(mc):
    fs = mc["feature_sets"]
    C = list(map(int, fs["CY-C"]))
    if len(C) != 42 or int(mc["noB_dimensions"]["CY-C"]) != 42:
        raise SystemExit(f"[FATAL] CY-C no-B dim mismatch: {len(C)}")
    return C


def complete_feature_contract(mc):
    fs = mc["feature_sets"]
    A = list(map(int, fs["CY-A"]))
    C = list(map(int, fs["CY-C"]))
    AC = list(map(int, fs["AC"]))
    if len(A) != 46 or len(C) != 42 or len(AC) != 86:
        raise SystemExit(f"[FATAL] A/C/AC no-B dim mismatch: A={len(A)} C={len(C)} AC={len(AC)}")
    core_pos = {orig: i for i, orig in enumerate(AC)}
    if len(core_pos) != 86:
        raise SystemExit("[FATAL] duplicate columns in AC")
    for x in set(A + C):
        if x not in core_pos:
            raise SystemExit(f"[FATAL] A/C col {x} absent from AC")
    return AC


def core_subset(X, base, idx, c_orig):
    idx = np.asarray(idx, dtype=np.int64)
    cols = np.asarray(c_orig, dtype=np.int64)
    out = np.empty((len(idx), len(cols) + 4), dtype=np.float32)
    out[:, : len(cols)] = np.asarray(X[idx[:, None], cols[None, :]], dtype=np.float32)
    out[:, len(cols) :] = onehot_base(np.asarray(base[idx]))
    if np.isinf(out).any():
        raise SystemExit("[FATAL] +/-inf in Core C matrix")
    return np.ascontiguousarray(out)


def build_complete_matrix(X, base, full_idx, S, ac_orig):
    ii = np.asarray(full_idx, dtype=np.int64)
    ac = np.asarray(ac_orig, dtype=np.int64)
    out = np.empty((len(ii), len(ac) + SURF_DIM + 4), dtype=np.float32)
    out[:, : len(ac)] = np.asarray(X[ii[:, None], ac[None, :]], dtype=np.float32)
    out[:, len(ac) : len(ac) + SURF_DIM] = np.asarray(S, dtype=np.float32)
    out[:, len(ac) + SURF_DIM :] = onehot_base(np.asarray(base[ii]))
    if out.shape[1] != COMPLETE_DIM:
        raise SystemExit(f"[FATAL] Complete dim mismatch: {out.shape}")
    if np.isinf(out).any():
        raise SystemExit("[FATAL] +/-inf in Complete ACS matrix")
    return np.ascontiguousarray(out)


def source_audit(root: Path, out: Path):
    core = root / "05_ligand_full_core/matrix"
    surf = root / "05_ligand_surf_ad_sensitivity/05_surf_matrix"
    bdir = root / "08_v2_systematic_selection/complete_nested_v1_0_1"
    cdir = root / "08_v2_systematic_selection/full_core_nested_v1_0_0"

    req_core = [
        "matrix_contract.json", "X_intrinsic.f32", "y.u8", "base.u8",
        "residue_index.i32", "row_chain_idx.i32", "row_pdb_idx.i32", "chain_row_ranges.csv",
    ]
    req_surf = [
        "STAGE05A_SURF_AD_MATRIX_GATE.json", "domain_full_row_index.i64", "SURF57.f32",
        "SURF57_feature_names.txt", "domain_chain_row_ranges.csv",
    ]
    upstream = [
        bdir / "STAGE08B_COMPLETE_NESTED_GATE.json",
        bdir / "CONSENSUS_COMPLETE_ARM.json",
        cdir / "STAGE08C_FULL_CORE_NESTED_GATE.json",
        cdir / "CONSENSUS_CORE_ARM.json",
    ]
    splits = [root / f"05_ligand_full_core/splits/pdb_split_seed{s}.npy" for s in SEEDS]
    missing = [str(core / x) for x in req_core if not (core / x).exists()]
    missing += [str(surf / x) for x in req_surf if not (surf / x).exists()]
    missing += [str(p) for p in upstream + splits if not p.exists()]
    if missing:
        raise SystemExit("[FATAL] missing Stage08D sources:\n" + "\n".join(missing))

    mc = jload(core / "matrix_contract.json")
    sg = jload(surf / "STAGE05A_SURF_AD_MATRIX_GATE.json")
    bg = jload(bdir / "STAGE08B_COMPLETE_NESTED_GATE.json")
    bc = jload(bdir / "CONSENSUS_COMPLETE_ARM.json")
    cg = jload(cdir / "STAGE08C_FULL_CORE_NESTED_GATE.json")
    cc = jload(cdir / "CONSENSUS_CORE_ARM.json")

    if not mc.get("gate_pass") or mc.get("matrix_shape") != [FULL_ROWS, FULL_DIM] or mc.get("B_factor_primary") is not False:
        raise SystemExit("[FATAL] full-Core matrix contract mismatch")
    if not sg.get("gate_pass") or int(sg.get("rows", -1)) != SURF_ROWS or int(sg.get("SURF_residue_dim", -1)) != SURF_DIM or sg.get("B_factor_primary") is not False:
        raise SystemExit("[FATAL] SURF applicability-domain matrix contract mismatch")
    if not bg.get("gate_pass") or bg.get("gate") != "PASS_STAGE08B_COMPLETE_NESTED_SELECTION_V101":
        raise SystemExit("[FATAL] Stage08B v1.0.1 not PASS")
    if not bc.get("gate_pass") or bc.get("selected_arm") != COMPLETE_ARM:
        raise SystemExit(f"[FATAL] expected Complete consensus {COMPLETE_ARM}, got {bc.get('selected_arm')}")
    if bc.get("outer_test_metrics_used_for_consensus_selection") is not False or bc.get("test17_used_for_selection") is not False:
        raise SystemExit("[FATAL] Stage08B consensus provenance invalid")
    if not cg.get("gate_pass") or cg.get("gate") != "PASS_STAGE08C_FULL_CORE_NESTED_SELECTION_V100":
        raise SystemExit("[FATAL] Stage08C v1.0.0 not PASS")
    if not cc.get("gate_pass") or cc.get("selected_arm") != CORE_ARM:
        raise SystemExit(f"[FATAL] expected Core consensus {CORE_ARM}, got {cc.get('selected_arm')}")
    if cc.get("outer_test_metrics_used_for_consensus_selection") is not False or cc.get("SURF_used_for_core_selection") is not False or cc.get("test17_used_for_selection") is not False:
        raise SystemExit("[FATAL] Stage08C consensus provenance invalid")
    if cg.get("consensus_core_arm") != CORE_ARM or cg.get("upstream_complete_consensus_arm") != COMPLETE_ARM:
        raise SystemExit("[FATAL] Stage08C upstream arm binding mismatch")

    expected = {
        core / "X_intrinsic.f32": FULL_ROWS * FULL_DIM * 4,
        core / "y.u8": FULL_ROWS,
        core / "base.u8": FULL_ROWS,
        core / "residue_index.i32": FULL_ROWS * 4,
        core / "row_chain_idx.i32": FULL_ROWS * 4,
        core / "row_pdb_idx.i32": FULL_ROWS * 4,
        surf / "domain_full_row_index.i64": SURF_ROWS * 8,
        surf / "SURF57.f32": SURF_ROWS * SURF_DIM * 4,
    }
    bad = []
    for p, nbytes in expected.items():
        if p.stat().st_size != nbytes:
            bad.append((str(p), p.stat().st_size, nbytes))
    if bad:
        raise SystemExit("[FATAL] raw memmap size mismatch: " + repr(bad))

    src = {
        "gate": "PASS_STAGE08D_EXACT_FIXED_ROUTED_SOURCE_BINDING",
        "gate_pass": True,
        "full_core_rows": FULL_ROWS,
        "surf_domain_rows": SURF_ROWS,
        "core_arm": CORE_ARM,
        "complete_arm": COMPLETE_ARM,
        "B_factor_included": False,
        "route_domain_is_frozen_label_blind_stage05a_surf_ad": True,
        "test17_used": False,
        "full_core_contract_sha256": sha256_file(core / "matrix_contract.json"),
        "surf_contract_sha256": sha256_file(surf / "STAGE05A_SURF_AD_MATRIX_GATE.json"),
        "surf57_sha256": sha256_file(surf / "SURF57.f32"),
        "domain_index_sha256": sha256_file(surf / "domain_full_row_index.i64"),
        "stage08b_gate_sha256": sha256_file(bdir / "STAGE08B_COMPLETE_NESTED_GATE.json"),
        "stage08b_consensus_sha256": sha256_file(bdir / "CONSENSUS_COMPLETE_ARM.json"),
        "stage08c_gate_sha256": sha256_file(cdir / "STAGE08C_FULL_CORE_NESTED_GATE.json"),
        "stage08c_consensus_sha256": sha256_file(cdir / "CONSENSUS_CORE_ARM.json"),
        "outer_split_sha256": {str(s): sha256_file(p) for s, p in zip(SEEDS, splits)},
    }
    dumpj(out / "STAGE08D_SOURCE_BINDING.json", src)
    return core, surf, mc, sg, src


def train_hnm_fixed(Xtr, ytr, chtr, Xes, yes, seed, device, threads):
    stage1 = fit_xgb(Xtr, ytr, Xes, yes, seed, device=device, threads=threads)
    ptr = stage1.predict_proba(Xtr)[:, 1]
    hmask = hard_negative_mask(ytr, ptr, chtr, HNM_TOPK)
    w = np.ones(len(ytr), dtype=np.float32)
    w[hmask] = HNM_WEIGHT
    final = fit_xgb(Xtr, ytr, Xes, yes, seed, sample_weight=w, device=device, threads=threads)
    meta = {
        "stage1_best_iteration": int(getattr(stage1, "best_iteration", -1)),
        "final_best_iteration": int(getattr(final, "best_iteration", -1)),
        "hard_negatives": int(hmask.sum()),
        "train_rows": int(len(ytr)),
        "early_stop_rows": int(len(yes)),
    }
    return stage1, final, meta


def add_metric_columns(row, prefix, m):
    for k, v in m.items():
        row[f"{prefix}_{k}"] = v


def run(args):
    root = Path(args.root).resolve()
    out = root / "08_v2_systematic_selection/routed_fixed_v1_0_0"
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    (out / "seed_results").mkdir(exist_ok=True)

    core, surf, mc, sg, src = source_audit(root, out)
    source_binding_sha256 = sha256_file(out / "STAGE08D_SOURCE_BINDING.json")
    device = os.environ.get("XGB_DEVICE", "cuda")
    threads = int(os.environ.get("THREADS", "32"))

    X = np.memmap(core / "X_intrinsic.f32", np.float32, "r", shape=(FULL_ROWS, FULL_DIM))
    y = np.asarray(np.memmap(core / "y.u8", np.uint8, "r", shape=(FULL_ROWS,)))
    base = np.memmap(core / "base.u8", np.uint8, "r", shape=(FULL_ROWS,))
    pdb = np.asarray(np.memmap(core / "row_pdb_idx.i32", np.int32, "r", shape=(FULL_ROWS,)))
    chain = np.asarray(np.memmap(core / "row_chain_idx.i32", np.int32, "r", shape=(FULL_ROWS,)))
    resid = np.asarray(np.memmap(core / "residue_index.i32", np.int32, "r", shape=(FULL_ROWS,)))
    full_idx = np.asarray(np.memmap(surf / "domain_full_row_index.i64", np.int64, "r", shape=(SURF_ROWS,)))
    S = np.memmap(surf / "SURF57.f32", np.float32, "r", shape=(SURF_ROWS, SURF_DIM))

    if len(np.unique(full_idx)) != SURF_ROWS or np.any(full_idx[1:] <= full_idx[:-1]) or full_idx.min() < 0 or full_idx.max() >= FULL_ROWS:
        raise SystemExit("[FATAL] SURF domain_full_row_index contract violated")
    surf_mask = np.zeros(FULL_ROWS, dtype=bool)
    surf_mask[full_idx] = True

    c_orig = core_feature_contract(mc)
    ac_orig = complete_feature_contract(mc)
    Zcomplete = build_complete_matrix(X, base, full_idx, S, ac_orig)
    if len(c_orig) + 4 != CORE_DIM:
        raise SystemExit("[FATAL] Core C dimension mismatch")

    rows = []
    branch_rows = []
    split_rows = []
    cal_rows = []

    for seed in SEEDS:
        seed_split_start = len(split_rows)
        seed_ck = out / "seed_results" / f"seed{seed}.json"
        seed_model_dir = out / "models" / f"seed{seed}"
        pred_path = out / "predictions" / f"seed{seed}_routed_test.npz"
        if seed_ck.exists() and pred_path.exists() and (seed_model_dir / "core_C_HNM.json").exists() and (seed_model_dir / "complete_ACS_HNM.json").exists():
            try:
                old = jload(seed_ck)
            except Exception:
                old = None
            if (
                old
                and old.get("protocol") == "STAGE08D_V100_SHARED_CAL_CONTROL"
                and old.get("source_binding_sha256") == source_binding_sha256
                and old.get("core_arm") == CORE_ARM
                and old.get("complete_arm") == COMPLETE_ARM
            ):
                rows.append(old["routed_result"])
                branch_rows.append(old["branch_result"])
                cal_rows.append(old["calibration"])
                split_rows.extend(old["split_rows"])
                print(
                    f"[RESUME SEED {seed}] routed AUPR={old['routed_result']['routed_calibrated_AUPR']:.6f} "
                    f"AUC={old['routed_result']['routed_calibrated_AUC']:.6f}",
                    flush=True,
                )
                continue

        split_path = root / f"05_ligand_full_core/splits/pdb_split_seed{seed}.npy"
        code = np.load(split_path, allow_pickle=False)
        if pdb.min() < 0 or pdb.max() >= len(code):
            raise SystemExit("[FATAL] row_pdb_idx outside split code range")
        rc = code[pdb]
        tr = np.flatnonzero(rc == 0)
        va = np.flatnonzero(rc == 1)
        te = np.flatnonzero(rc == 2)
        for nm, ix in [("train", tr), ("validation", va), ("test", te)]:
            if len(np.unique(y[ix])) < 2:
                raise SystemExit(f"[FATAL] seed {seed} {nm} lacks both classes")
            split_rows.append({
                "outer_seed": seed, "branch": "full_core", "split": nm,
                "rows": len(ix), "positive_rows": int(y[ix].sum()),
                "positive_fraction": float(y[ix].mean()), "pdbs": int(len(np.unique(pdb[ix]))),
            })

        # Create one shared PDB-level partition of the frozen outer validation.
        # Both branches use the same ES/calibration PDB assignment.
        es_pdbs, cal_pdbs = split_shared_es_cal_pdbs(
            va, pdb, y, surf_mask, seed, VAL_CAL_FRACTION
        )

        # Core model: fixed C+HNM. Full outer-train is fit; shared outer-val PDBs
        # provide disjoint early-stop and calibration subsets.
        core_es = rows_for_pdb_set(va, pdb, es_pdbs)
        core_cal = rows_for_pdb_set(va, pdb, cal_pdbs)
        if len(np.unique(y[core_es])) < 2 or len(np.unique(y[core_cal])) < 2:
            raise SystemExit(f"[FATAL] seed {seed} Core ES/calibration lacks both classes")
        for nm, ix in [("early_stop", core_es), ("calibration", core_cal)]:
            split_rows.append({
                "outer_seed": seed, "branch": "core_C_HNM", "split": nm,
                "rows": len(ix), "positive_rows": int(y[ix].sum()),
                "positive_fraction": float(y[ix].mean()), "pdbs": int(len(np.unique(pdb[ix]))),
            })

        Xtr_c = core_subset(X, base, tr, c_orig)
        Xes_c = core_subset(X, base, core_es, c_orig)
        _, core_model, core_meta = train_hnm_fixed(
            Xtr_c, y[tr], chain[tr], Xes_c, y[core_es], seed, device, threads
        )
        seed_model_dir.mkdir(exist_ok=True)
        core_model.save_model(str(seed_model_dir / "core_C_HNM.json"))

        Xcal_c = core_subset(X, base, core_cal, c_orig)
        pcal_core = core_model.predict_proba(Xcal_c)[:, 1]
        core_global_platt = fit_platt(y[core_cal], pcal_core)
        cal_non_surf_mask = ~surf_mask[core_cal]
        cal_surf_mask = surf_mask[core_cal]
        if len(np.unique(y[core_cal[cal_non_surf_mask]])) < 2:
            raise SystemExit(f"[FATAL] seed {seed} Core non-SURF calibration lacks both classes")
        if len(np.unique(y[core_cal[cal_surf_mask]])) < 2:
            raise SystemExit(f"[FATAL] seed {seed} Core SURF calibration lacks both classes")
        core_nonsurf_platt = fit_platt(y[core_cal[cal_non_surf_mask]], pcal_core[cal_non_surf_mask])
        core_surf_platt = fit_platt(y[core_cal[cal_surf_mask]], pcal_core[cal_surf_mask])

        Xte_c = core_subset(X, base, te, c_orig)
        pte_core_raw = core_model.predict_proba(Xte_c)[:, 1]
        pte_core_cal = apply_platt(pte_core_raw, core_global_platt)
        m_core_raw = metric_bundle(y[te], pte_core_raw)
        m_core_cal = metric_bundle(y[te], pte_core_cal)

        # Complete model: fixed ACS+HNM inside the frozen Stage05A SURF domain.
        surf_pdb = pdb[full_idx]
        surf_y = y[full_idx]
        surf_chain = chain[full_idx]
        surf_rc = code[surf_pdb]
        ctr = np.flatnonzero(surf_rc == 0)
        cva = np.flatnonzero(surf_rc == 1)
        cte = np.flatnonzero(surf_rc == 2)
        for nm, pos in [("train", ctr), ("validation", cva), ("test", cte)]:
            if len(np.unique(surf_y[pos])) < 2:
                raise SystemExit(f"[FATAL] seed {seed} Complete {nm} lacks both classes")
            split_rows.append({
                "outer_seed": seed, "branch": "complete_ACS_HNM", "split": nm,
                "rows": len(pos), "positive_rows": int(surf_y[pos].sum()),
                "positive_fraction": float(surf_y[pos].mean()), "pdbs": int(len(np.unique(surf_pdb[pos]))),
            })

        comp_es = cva[np.isin(surf_pdb[cva], es_pdbs)]
        comp_cal = cva[np.isin(surf_pdb[cva], cal_pdbs)]
        if len(np.unique(surf_y[comp_es])) < 2 or len(np.unique(surf_y[comp_cal])) < 2:
            raise SystemExit(f"[FATAL] seed {seed} Complete ES/calibration lacks both classes")
        if set(map(int, np.unique(surf_pdb[comp_es]))) & set(map(int, np.unique(surf_pdb[comp_cal]))):
            raise SystemExit(f"[FATAL] seed {seed} Complete ES/calibration PDB overlap")
        for nm, pos in [("early_stop", comp_es), ("calibration", comp_cal)]:
            split_rows.append({
                "outer_seed": seed, "branch": "complete_ACS_HNM", "split": nm,
                "rows": len(pos), "positive_rows": int(surf_y[pos].sum()),
                "positive_fraction": float(surf_y[pos].mean()), "pdbs": int(len(np.unique(surf_pdb[pos]))),
            })

        Xtr_a = np.ascontiguousarray(Zcomplete[ctr], dtype=np.float32)
        Xes_a = np.ascontiguousarray(Zcomplete[comp_es], dtype=np.float32)
        _, complete_model, complete_meta = train_hnm_fixed(
            Xtr_a, surf_y[ctr], surf_chain[ctr], Xes_a, surf_y[comp_es], seed, device, threads
        )
        complete_model.save_model(str(seed_model_dir / "complete_ACS_HNM.json"))

        Xcal_a = np.ascontiguousarray(Zcomplete[comp_cal], dtype=np.float32)
        pcal_complete = complete_model.predict_proba(Xcal_a)[:, 1]
        complete_platt = fit_platt(surf_y[comp_cal], pcal_complete)
        Xte_a = np.ascontiguousarray(Zcomplete[cte], dtype=np.float32)
        pte_complete_raw = complete_model.predict_proba(Xte_a)[:, 1]
        pte_complete_cal = apply_platt(pte_complete_raw, complete_platt)
        m_complete_raw = metric_bundle(surf_y[cte], pte_complete_raw)
        m_complete_cal = metric_bundle(surf_y[cte], pte_complete_cal)
        # Exact row alignment for routed replacement.
        surf_test_full = full_idx[cte]
        loc = np.searchsorted(te, surf_test_full)
        if len(loc) != len(surf_test_full) or np.any(loc >= len(te)) or not np.array_equal(te[loc], surf_test_full):
            raise SystemExit(f"[FATAL] seed {seed} SURF test rows do not align to full-Core test rows")
        non_surf_loc_mask = ~surf_mask[te]
        surf_loc_mask = surf_mask[te]
        if int(surf_loc_mask.sum()) != len(cte):
            raise SystemExit(f"[FATAL] seed {seed} routed SURF row count mismatch")

        # Direct Complete-vs-Core comparison on exactly the same SURF test rows.
        pte_core_on_surf_raw = pte_core_raw[loc]
        m_core_on_surf_raw = metric_bundle(surf_y[cte], pte_core_on_surf_raw)

        # Primary global route uses branch-specific Platt calibration fitted on
        # the shared PDB-disjoint outer-validation calibration PDBs. Raw pooled
        # route is diagnostic only because branch raw score scales differ.
        routed_raw = np.asarray(pte_core_raw, dtype=np.float64).copy()
        routed_raw[loc] = pte_complete_raw
        routed_cal = np.empty(len(te), dtype=np.float64)
        routed_cal[non_surf_loc_mask] = apply_platt(
            pte_core_raw[non_surf_loc_mask], core_nonsurf_platt
        )
        routed_cal[loc] = pte_complete_cal

        # Calibration-matched control: keep the Core C+HNM model on both route
        # domains, but calibrate its non-SURF and SURF scores separately using
        # the same shared calibration PDB partition. This isolates the effect of
        # replacing Core by Complete from any gain due only to domain calibration.
        core_partition_cal = np.empty(len(te), dtype=np.float64)
        core_partition_cal[non_surf_loc_mask] = apply_platt(
            pte_core_raw[non_surf_loc_mask], core_nonsurf_platt
        )
        core_partition_cal[loc] = apply_platt(
            pte_core_raw[loc], core_surf_platt
        )

        m_route_raw = metric_bundle(y[te], routed_raw)
        m_route_cal = metric_bundle(y[te], routed_cal)
        m_core_partition_cal = metric_bundle(y[te], core_partition_cal)
        m_core_nonsurf_raw = metric_bundle(y[te][non_surf_loc_mask], pte_core_raw[non_surf_loc_mask])
        m_core_nonsurf_cal = metric_bundle(
            y[te][non_surf_loc_mask],
            apply_platt(pte_core_raw[non_surf_loc_mask], core_nonsurf_platt),
        )

        rr = {
            "outer_seed": seed,
            "core_arm": CORE_ARM,
            "complete_arm": COMPLETE_ARM,
            "test_rows": int(len(te)),
            "test_positive_rows": int(y[te].sum()),
            "test_positive_fraction": float(y[te].mean()),
            "surf_test_rows": int(len(cte)),
            "surf_test_fraction": float(len(cte) / len(te)),
            "nonsurf_test_rows": int(non_surf_loc_mask.sum()),
            "core_hard_negatives": core_meta["hard_negatives"],
            "complete_hard_negatives": complete_meta["hard_negatives"],
            "core_best_iteration": core_meta["final_best_iteration"],
            "complete_best_iteration": complete_meta["final_best_iteration"],
        }
        add_metric_columns(rr, "core_only_raw", m_core_raw)
        add_metric_columns(rr, "core_only_global_calibrated", m_core_cal)
        add_metric_columns(rr, "core_partition_calibrated_control", m_core_partition_cal)
        add_metric_columns(rr, "routed_raw_diagnostic", m_route_raw)
        add_metric_columns(rr, "routed_calibrated", m_route_cal)
        rr["delta_routed_minus_core_partition_AUPR"] = rr["routed_calibrated_AUPR"] - rr["core_partition_calibrated_control_AUPR"]
        rr["delta_routed_minus_core_partition_AUC"] = rr["routed_calibrated_AUC"] - rr["core_partition_calibrated_control_AUC"]
        rr["delta_routed_minus_core_partition_Brier"] = rr["routed_calibrated_Brier"] - rr["core_partition_calibrated_control_Brier"]
        rr["delta_routed_minus_core_partition_LogLoss"] = rr["routed_calibrated_LogLoss"] - rr["core_partition_calibrated_control_LogLoss"]
        rr["delta_routed_minus_core_global_AUPR"] = rr["routed_calibrated_AUPR"] - rr["core_only_global_calibrated_AUPR"]
        rr["delta_routed_minus_core_global_AUC"] = rr["routed_calibrated_AUC"] - rr["core_only_global_calibrated_AUC"]
        rows.append(rr)

        br = {
            "outer_seed": seed,
            "surf_test_rows": int(len(cte)),
            "surf_positive_rows": int(surf_y[cte].sum()),
            "surf_positive_fraction": float(surf_y[cte].mean()),
        }
        add_metric_columns(br, "core_on_surf_raw", m_core_on_surf_raw)
        add_metric_columns(br, "complete_on_surf_raw", m_complete_raw)
        add_metric_columns(br, "complete_on_surf_calibrated", m_complete_cal)
        add_metric_columns(br, "core_on_nonsurf_raw", m_core_nonsurf_raw)
        add_metric_columns(br, "core_on_nonsurf_calibrated", m_core_nonsurf_cal)
        br["delta_complete_minus_core_surf_AUPR"] = br["complete_on_surf_raw_AUPR"] - br["core_on_surf_raw_AUPR"]
        br["delta_complete_minus_core_surf_AUC"] = br["complete_on_surf_raw_AUC"] - br["core_on_surf_raw_AUC"]
        branch_rows.append(br)

        cal_info = {
            "outer_seed": seed,
            "core_global_slope": core_global_platt["slope"],
            "core_global_intercept": core_global_platt["intercept"],
            "core_nonsurf_slope": core_nonsurf_platt["slope"],
            "core_nonsurf_intercept": core_nonsurf_platt["intercept"],
            "core_surf_slope": core_surf_platt["slope"],
            "core_surf_intercept": core_surf_platt["intercept"],
            "complete_slope": complete_platt["slope"],
            "complete_intercept": complete_platt["intercept"],
            "shared_es_pdbs": int(len(es_pdbs)),
            "shared_cal_pdbs": int(len(cal_pdbs)),
            "core_cal_rows": int(len(core_cal)),
            "core_nonsurf_cal_rows": int(cal_non_surf_mask.sum()),
            "core_surf_cal_rows": int(cal_surf_mask.sum()),
            "complete_cal_rows": int(len(comp_cal)),
        }
        cal_rows.append(cal_info)
        dumpj(seed_model_dir / "calibration.json", cal_info)

        np.savez_compressed(
            pred_path,
            full_row_index=te.astype(np.int64),
            pdb_index=pdb[te].astype(np.int32),
            chain_index=chain[te].astype(np.int32),
            residue_index=resid[te].astype(np.int32),
            label=y[te].astype(np.uint8),
            is_surf=surf_mask[te].astype(np.uint8),
            core_raw=np.asarray(pte_core_raw, dtype=np.float32),
            core_global_calibrated=np.asarray(pte_core_cal, dtype=np.float32),
            core_partition_calibrated_control=np.asarray(core_partition_cal, dtype=np.float32),
            routed_raw_diagnostic=np.asarray(routed_raw, dtype=np.float32),
            routed_calibrated=np.asarray(routed_cal, dtype=np.float32),
        )

        seed_record = {
            "protocol": "STAGE08D_V100_SHARED_CAL_CONTROL",
            "source_binding_sha256": source_binding_sha256,
            "core_arm": CORE_ARM,
            "complete_arm": COMPLETE_ARM,
            "routed_result": rr,
            "branch_result": br,
            "calibration": cal_info,
            "split_rows": split_rows[seed_split_start:],
        }
        dumpj(seed_ck, seed_record)

        print(
            f"[SEED {seed}] routed calibrated AUPR={m_route_cal['AUPR']:.6f} "
            f"AUC={m_route_cal['AUC']:.6f} | core-only calibrated "
            f"AUPR={m_core_cal['AUPR']:.6f} AUC={m_core_cal['AUC']:.6f} | "
            f"SURF rows={len(cte)}",
            flush=True,
        )

        del Xtr_c, Xes_c, Xcal_c, Xte_c, Xtr_a, Xes_a, Xcal_a, Xte_a
        del core_model, complete_model, pte_core_raw, pte_complete_raw, routed_raw, routed_cal, core_partition_cal

    rdf = pd.DataFrame(rows)
    bdf = pd.DataFrame(branch_rows)
    sdf = pd.DataFrame(split_rows)
    cdf = pd.DataFrame(cal_rows)
    rdf.to_csv(out / "ROUTED_RESULTS.csv", index=False)
    bdf.to_csv(out / "SURF_BRANCH_COMPARISON.csv", index=False)
    sdf.to_csv(out / "SPLIT_AUDIT.csv", index=False)
    cdf.to_csv(out / "CALIBRATION_AUDIT.csv", index=False)

    summary = {
        "gate": "PASS_STAGE08D_FIXED_CORE_COMPLETE_ROUTED_EVALUATION_V100",
        "gate_pass": True,
        "core_arm": CORE_ARM,
        "complete_arm": COMPLETE_ARM,
        "route": "Stage05A SURF-AD rows -> ACS+HNM; all other full-Core rows -> C+HNM",
        "route_domain_is_label_blind_and_frozen_before_stage08d": True,
        "candidate_selection_performed_in_stage08d": False,
        "test17_used": False,
        "outer_test_used_for_fitting": False,
        "outer_test_used_for_calibration": False,
        "outer_test_used_for_routing_rule": False,
        "outer_validation_protocol": "one shared PDB-disjoint 50/50 early-stop/calibration partition of frozen outer-validation PDBs; both Core and Complete use the same PDB assignment",
        "calibration": "Platt scaling on logit(raw score), fit only on shared frozen outer-validation calibration PDBs with natural unweighted labels",
        "primary_global_metric_scores": "routed_calibrated",
        "primary_representation_substitution_comparator": "core_partition_calibrated_control",
        "raw_pooled_routed_scores_are_diagnostic_only": True,
        "raw_pooled_routed_reason": "Core and Complete models are separately trained under different applicability domains and HNM weighting; raw score scales are not assumed comparable across branches.",
        "outer_seeds": SEEDS,
        "mean_surf_test_fraction": float(rdf.surf_test_fraction.mean()),
        "routed_calibrated_AUPR_mean": float(rdf.routed_calibrated_AUPR.mean()),
        "routed_calibrated_AUPR_std": float(rdf.routed_calibrated_AUPR.std(ddof=0)),
        "routed_calibrated_AUC_mean": float(rdf.routed_calibrated_AUC.mean()),
        "routed_calibrated_AUC_std": float(rdf.routed_calibrated_AUC.std(ddof=0)),
        "core_only_global_calibrated_AUPR_mean": float(rdf.core_only_global_calibrated_AUPR.mean()),
        "core_only_global_calibrated_AUPR_std": float(rdf.core_only_global_calibrated_AUPR.std(ddof=0)),
        "core_only_global_calibrated_AUC_mean": float(rdf.core_only_global_calibrated_AUC.mean()),
        "core_only_global_calibrated_AUC_std": float(rdf.core_only_global_calibrated_AUC.std(ddof=0)),
        "core_partition_calibrated_control_AUPR_mean": float(rdf.core_partition_calibrated_control_AUPR.mean()),
        "core_partition_calibrated_control_AUC_mean": float(rdf.core_partition_calibrated_control_AUC.mean()),
        "primary_delta_routed_minus_core_partition_AUPR_mean": float(rdf.delta_routed_minus_core_partition_AUPR.mean()),
        "primary_delta_routed_minus_core_partition_AUPR_std": float(rdf.delta_routed_minus_core_partition_AUPR.std(ddof=0)),
        "primary_delta_routed_minus_core_partition_AUPR_positive_seeds": int((rdf.delta_routed_minus_core_partition_AUPR > 0).sum()),
        "primary_delta_routed_minus_core_partition_AUC_mean": float(rdf.delta_routed_minus_core_partition_AUC.mean()),
        "primary_delta_routed_minus_core_partition_AUC_std": float(rdf.delta_routed_minus_core_partition_AUC.std(ddof=0)),
        "primary_delta_routed_minus_core_partition_AUC_positive_seeds": int((rdf.delta_routed_minus_core_partition_AUC > 0).sum()),
        "delta_routed_minus_core_global_AUPR_mean": float(rdf.delta_routed_minus_core_global_AUPR.mean()),
        "delta_routed_minus_core_global_AUC_mean": float(rdf.delta_routed_minus_core_global_AUC.mean()),
        "complete_minus_core_surf_AUPR_mean": float(bdf.delta_complete_minus_core_surf_AUPR.mean()),
        "complete_minus_core_surf_AUPR_positive_seeds": int((bdf.delta_complete_minus_core_surf_AUPR > 0).sum()),
        "complete_minus_core_surf_AUC_mean": float(bdf.delta_complete_minus_core_surf_AUC.mean()),
        "complete_minus_core_surf_AUC_positive_seeds": int((bdf.delta_complete_minus_core_surf_AUC > 0).sum()),
        "all_platt_slopes_positive": bool((cdf[["core_global_slope", "core_nonsurf_slope", "core_surf_slope", "complete_slope"]].to_numpy() > 0).all()),
        "historical_outer_test_results_previously_seen": True,
        "future_untouched_external_confirmation_required_for_fixed_system_claim": True,
        "source_binding_gate": src["gate"],
        "next_stage": "Stage08E interpretation/manuscript integration only after auditing Stage08D routed deltas; do not reopen arm selection from Stage08D outer-test results.",
    }
    dumpj(out / "STAGE08D_FIXED_ROUTED_GATE.json", summary)

    frozen = {
        "gate": "PASS_STAGE08D_ROUTED_SYSTEM_SPECIFICATION_FROZEN",
        "gate_pass": True,
        "core_branch": {"domain": "full-Core rows outside frozen SURF AD", "arm": CORE_ARM, "dim": CORE_DIM},
        "complete_branch": {"domain": "frozen Stage05A SURF AD", "arm": COMPLETE_ARM, "dim": COMPLETE_DIM},
        "routing_key": "domain_full_row_index.i64",
        "routing_rule_uses_labels": False,
        "branch_models_use_B_factor": False,
        "branch_models_use_test17": False,
        "global_score_combination": "branch/domain-specific Platt calibration on shared outer-validation calibration PDBs before routing",
        "calibration_matched_core_control": "Core C+HNM on both domains with separate non-SURF/SURF calibration using the same shared calibration PDBs",
        "no_stage08d_model_or_arm_selection": True,
    }
    dumpj(out / "FROZEN_ROUTED_SYSTEM.json", frozen)
    print(json.dumps(summary, indent=2), flush=True)


def selftest():
    y = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=np.uint8)
    p = np.array([0.1, 0.9, 0.8, 0.7, 0.3, 0.2, 0.6, 0.4, 0.5, 0.05])
    ch = np.array([1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
    m = hard_negative_mask(y, p, ch, 2)
    assert set(np.flatnonzero(m)) == {1, 2, 6, 7}

    # Deterministic shared PDB-disjoint validation split with valid SURF strata.
    fpdb = np.repeat(np.arange(20), 3)
    fy = np.zeros(60, dtype=np.uint8)
    for pp in list(range(6)) + [12, 13]:
        fy[np.flatnonzero(fpdb == pp)[0]] = 1
    fsurf = np.isin(fpdb, np.arange(12))
    ix = np.arange(60)
    es_pdb1, ca_pdb1 = split_shared_es_cal_pdbs(ix, fpdb, fy, fsurf, 3407, 0.5)
    es_pdb2, ca_pdb2 = split_shared_es_cal_pdbs(ix, fpdb, fy, fsurf, 3407, 0.5)
    assert np.array_equal(es_pdb1, es_pdb2) and np.array_equal(ca_pdb1, ca_pdb2)
    assert not set(es_pdb1) & set(ca_pdb1)
    es1 = rows_for_pdb_set(ix, fpdb, es_pdb1)
    ca1 = rows_for_pdb_set(ix, fpdb, ca_pdb1)
    assert len(np.unique(fy[es1])) == 2 and len(np.unique(fy[ca1])) == 2
    surf_es = es1[fsurf[es1]]; surf_ca = ca1[fsurf[ca1]]
    assert len(np.unique(fy[surf_es])) == 2 and len(np.unique(fy[surf_ca])) == 2

    # Platt scaling is finite and monotone when slope is positive.
    py = np.array([0, 0, 0, 1, 1, 1], dtype=np.uint8)
    pp = np.array([0.05, 0.15, 0.35, 0.55, 0.75, 0.95])
    pars = fit_platt(py, pp)
    qq = apply_platt(pp, pars)
    assert np.isfinite(qq).all() and ((qq > 0) & (qq < 1)).all()
    assert pars["slope"] > 0

    # Exact routed replacement regression.
    te = np.array([1, 3, 4, 8, 10, 12], dtype=np.int64)
    surf_full = np.array([3, 10], dtype=np.int64)
    loc = np.searchsorted(te, surf_full)
    assert np.array_equal(te[loc], surf_full)
    core = np.arange(len(te), dtype=float)
    comp = np.array([100.0, 200.0])
    routed = core.copy(); routed[loc] = comp
    assert routed.tolist() == [0.0, 100.0, 2.0, 3.0, 200.0, 5.0]

    # Frozen dimensions and arms.
    assert CORE_ARM == "C+HNM" and COMPLETE_ARM == "ACS+HNM"
    assert CORE_DIM == 46 and COMPLETE_DIM == 147
    print("[PASS] Stage08D fixed routed evaluation selftest")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        if not a.root:
            raise SystemExit("--root required")
        run(a)
