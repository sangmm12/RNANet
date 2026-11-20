#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

SEEDS = [3407, 3420, 3433, 3446, 3459]
INNER_FOLDS = 3
INNER_ES_PDB_FRACTION = 0.10
HNM_TOPK = 20
HNM_WEIGHT = 5.0
BASE_ORDER = ["A", "C", "S", "AC", "AS", "CS", "ACS"]
EXPECTED_ARMS = [x for r in BASE_ORDER for x in (r, r + "+HNM")]

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


def sha256_file(p: Path, block=8 * 1024 * 1024):
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def jload(p):
    return json.loads(Path(p).read_text())


def dumpj(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def onehot_base(base):
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


def metrics(y, p):
    y = np.asarray(y, dtype=np.uint8)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return {"AUPR": float("nan"), "AUC": float("nan")}
    return {
        "AUPR": float(average_precision_score(y, p)),
        "AUC": float(roc_auc_score(y, p)),
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
    order = np.argsort(cneg, kind="mergesort")
    neg = neg[order]
    cneg = cneg[order]
    starts = np.r_[0, np.flatnonzero(cneg[1:] != cneg[:-1]) + 1]
    ends = np.r_[starts[1:], len(neg)]
    for a, b in zip(starts, ends):
        idx = neg[a:b]
        k = min(int(topk), len(idx))
        # deterministic: highest prediction, then smallest global/local index
        oo = np.lexsort((idx, -pred[idx]))[:k]
        out[idx[oo]] = True
    return out


def arm_defs(core_contract):
    fs = core_contract["feature_sets"]
    A = list(map(int, fs["CY-A"]))
    C = list(map(int, fs["CY-C"]))
    AC = list(map(int, fs["AC"]))
    if len(A) != int(core_contract["noB_dimensions"]["CY-A"]):
        raise SystemExit("[FATAL] CY-A dim mismatch")
    if len(C) != int(core_contract["noB_dimensions"]["CY-C"]):
        raise SystemExit("[FATAL] CY-C dim mismatch")
    if len(AC) != int(core_contract["noB_dimensions"]["AC"]):
        raise SystemExit("[FATAL] AC dim mismatch")
    core_pos = {orig: i for i, orig in enumerate(AC)}
    for x in set(A + C):
        if x not in core_pos:
            raise SystemExit(f"[FATAL] A/C col {x} absent from AC")
    Apos = [core_pos[x] for x in A]
    Cpos = [core_pos[x] for x in C]
    ACpos = list(range(len(AC)))
    s0 = len(AC)
    spos = list(range(s0, s0 + 57))
    bpos = list(range(s0 + 57, s0 + 57 + 4))
    defs = {
        "A": Apos + bpos,
        "C": Cpos + bpos,
        "S": spos + bpos,
        "AC": ACpos + bpos,
        "AS": Apos + spos + bpos,
        "CS": Cpos + spos + bpos,
        "ACS": ACpos + spos + bpos,
    }
    for k, v in defs.items():
        if len(v) != len(set(v)):
            raise SystemExit(f"[FATAL] duplicate columns in {k}")
    expected_dims = {"A": 50, "C": 46, "S": 61, "AC": 90, "AS": 107, "CS": 103, "ACS": 147}
    got_dims = {k: len(v) for k, v in defs.items()}
    if got_dims != expected_dims:
        raise SystemExit(f"[FATAL] arm dimension contract mismatch: got={got_dims} expected={expected_dims}")
    return defs, AC


def stable_fold(pdb_index: int, outer_seed: int, nfold=3):
    h = hashlib.sha256(f"RNANET_STAGE08B_INNER_SCORE|{outer_seed}|{pdb_index}".encode()).digest()
    return int.from_bytes(h[:8], "little") % int(nfold)


def stable_rank(tag: str, seed: int, fold: int, pdb_index: int):
    h = hashlib.sha256(f"{tag}|{seed}|{fold}|{pdb_index}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def split_fit_es_by_pdb(pool_idx, pdb, y, outer_seed, inner_fold, frac=0.10):
    """Split an inner-fit pool into PDB-disjoint fit and early-stop subsets.

    The inner score fold is not passed here, so it cannot influence early stopping.
    To stabilize rare positives, PDBs are stratified only by whether they contain at
    least one positive row, using labels already inside the inner-fit pool.
    """
    pool_idx = np.asarray(pool_idx, dtype=np.int64)
    pool_pdb = np.asarray(pdb[pool_idx], dtype=np.int64)
    uniq = np.unique(pool_pdb)
    if len(uniq) < 10:
        raise SystemExit(f"[FATAL] too few PDBs for inner fit/ES split: {len(uniq)}")
    has_pos = {}
    for pp in uniq:
        rows = pool_idx[pool_pdb == pp]
        has_pos[int(pp)] = bool(np.any(y[rows] == 1))
    pos_pdb = [int(pp) for pp in uniq if has_pos[int(pp)]]
    neg_pdb = [int(pp) for pp in uniq if not has_pos[int(pp)]]
    if len(pos_pdb) < 2 or len(neg_pdb) < 2:
        raise SystemExit(
            f"[FATAL] insufficient positive/negative PDB strata for inner ES split: "
            f"pos_pdb={len(pos_pdb)} neg_pdb={len(neg_pdb)}"
        )

    def choose(group):
        ordered = sorted(group, key=lambda pp: stable_rank("RNANET_STAGE08B_INNER_ES", outer_seed, inner_fold, pp))
        n = max(1, int(round(len(ordered) * float(frac))))
        n = min(n, len(ordered) - 1)
        return set(ordered[:n])

    es_pdb = choose(pos_pdb) | choose(neg_pdb)
    is_es = np.isin(pool_pdb, np.array(sorted(es_pdb), dtype=np.int64))
    es_idx = pool_idx[is_es]
    fit_idx = pool_idx[~is_es]
    if len(np.intersect1d(np.unique(pdb[fit_idx]), np.unique(pdb[es_idx]))) != 0:
        raise SystemExit("[FATAL] inner fit/ES PDB overlap")
    if len(np.unique(y[fit_idx])) < 2 or len(np.unique(y[es_idx])) < 2:
        raise SystemExit(
            f"[FATAL] inner fit/ES lacks both classes seed={outer_seed} fold={inner_fold} "
            f"fit_pos={int(y[fit_idx].sum())}/{len(fit_idx)} es_pos={int(y[es_idx].sum())}/{len(es_idx)}"
        )
    return fit_idx, es_idx


def build_complete_matrix(core_dir: Path, surf_dir: Path, mc, sg):
    nfull = int(mc["rows"])
    dfull = int(mc["matrix_shape"][1])
    nad = int(sg["rows"])
    sdim = int(sg["SURF_residue_dim"])
    X = np.memmap(core_dir / "X_intrinsic.f32", np.float32, "r", shape=(nfull, dfull))
    base = np.memmap(core_dir / "base.u8", np.uint8, "r", shape=(nfull,))
    y = np.memmap(core_dir / "y.u8", np.uint8, "r", shape=(nfull,))
    pdb = np.memmap(core_dir / "row_pdb_idx.i32", np.int32, "r", shape=(nfull,))
    chain = np.memmap(core_dir / "row_chain_idx.i32", np.int32, "r", shape=(nfull,))
    resid = np.memmap(core_dir / "residue_index.i32", np.int32, "r", shape=(nfull,))
    idx = np.memmap(surf_dir / "domain_full_row_index.i64", np.int64, "r", shape=(nad,))
    S = np.memmap(surf_dir / "SURF57.f32", np.float32, "r", shape=(nad, sdim))
    ii = np.asarray(idx)
    if len(ii) != nad or len(np.unique(ii)) != nad or np.any(ii[1:] <= ii[:-1]) or ii.min() < 0 or ii.max() >= nfull:
        raise SystemExit("[FATAL] SURF domain_full_row_index contract violated")
    defs, ac_orig = arm_defs(mc)
    Z = np.empty((nad, len(ac_orig) + sdim + 4), dtype=np.float32)
    Z[:, : len(ac_orig)] = np.asarray(X[ii[:, None], np.asarray(ac_orig)[None, :]], dtype=np.float32)
    Z[:, len(ac_orig) : len(ac_orig) + sdim] = np.asarray(S, dtype=np.float32)
    Z[:, len(ac_orig) + sdim :] = onehot_base(np.asarray(base[ii]))
    if np.isinf(Z).any():
        raise SystemExit("[FATAL] +/-inf detected in complete-domain predictor matrix")
    return Z, defs, ii, np.asarray(y[ii]), np.asarray(pdb[ii]), np.asarray(chain[ii]), np.asarray(resid[ii])


def source_audit(root: Path, out: Path):
    core = root / "05_ligand_full_core/matrix"
    surf = root / "05_ligand_surf_ad_sensitivity/05_surf_matrix"
    req_core = [
        "matrix_contract.json", "X_intrinsic.f32", "y.u8", "base.u8",
        "residue_index.i32", "row_chain_idx.i32", "row_pdb_idx.i32", "chain_row_ranges.csv",
    ]
    req_surf = [
        "STAGE05A_SURF_AD_MATRIX_GATE.json", "domain_full_row_index.i64", "SURF57.f32",
        "SURF57_feature_names.txt", "domain_chain_row_ranges.csv",
    ]
    missing = [str(core / x) for x in req_core if not (core / x).exists()] + [str(surf / x) for x in req_surf if not (surf / x).exists()]
    if missing:
        raise SystemExit("[FATAL] missing exact source files:\n" + "\n".join(missing))
    mc = jload(core / "matrix_contract.json")
    sg = jload(surf / "STAGE05A_SURF_AD_MATRIX_GATE.json")
    if not mc.get("gate_pass") or mc.get("matrix_shape") != [6252187, 96] or mc.get("B_factor_primary") is not False:
        raise SystemExit("[FATAL] full-core matrix contract mismatch")
    if not sg.get("gate_pass") or int(sg.get("rows", -1)) != 431226 or int(sg.get("SURF_residue_dim", -1)) != 57 or sg.get("B_factor_primary") is not False:
        raise SystemExit("[FATAL] SURF matrix contract mismatch")
    expected = {
        core / "X_intrinsic.f32": 6252187 * 96 * 4,
        core / "y.u8": 6252187,
        core / "base.u8": 6252187,
        core / "residue_index.i32": 6252187 * 4,
        core / "row_chain_idx.i32": 6252187 * 4,
        core / "row_pdb_idx.i32": 6252187 * 4,
        surf / "domain_full_row_index.i64": 431226 * 8,
        surf / "SURF57.f32": 431226 * 57 * 4,
    }
    bad = []
    for p, n in expected.items():
        if p.stat().st_size != n:
            bad.append((str(p), p.stat().st_size, n))
    if bad:
        raise SystemExit("[FATAL] raw memmap size mismatch: " + repr(bad))
    src = {
        "gate": "PASS_STAGE08B_EXACT_SOURCE_BINDING",
        "gate_pass": True,
        "full_core_rows": 6252187,
        "full_core_dim": 96,
        "full_core_positive_rows": int(mc.get("label_positive_rows", -1)),
        "surf_rows": 431226,
        "surf_dim": 57,
        "surf57_size_bytes": (surf / "SURF57.f32").stat().st_size,
        "expected_surf57_size_bytes": 431226 * 57 * 4,
        "full_core_contract_sha256": sha256_file(core / "matrix_contract.json"),
        "surf_contract_sha256": sha256_file(surf / "STAGE05A_SURF_AD_MATRIX_GATE.json"),
        "surf57_sha256": sha256_file(surf / "SURF57.f32"),
        "domain_index_sha256": sha256_file(surf / "domain_full_row_index.i64"),
        "B_factor_primary": False,
        "test17_used_for_selection": False,
    }
    dumpj(out / "STAGE08B_SOURCE_BINDING.json", src)
    return core, surf, mc, sg, src


def subset(Z, idx, cols):
    return np.ascontiguousarray(Z[np.asarray(idx)[:, None], np.asarray(cols)[None, :]], dtype=np.float32)


def valid_fold_checkpoint(p: Path, outer_seed: int, fold: int):
    if not p.exists():
        return None
    try:
        d = pd.read_csv(p)
    except Exception:
        return None
    if len(d) != 14 or set(d.get("arm", [])) != set(EXPECTED_ARMS):
        return None
    if set(d.get("outer_seed", [])) != {outer_seed} or set(d.get("inner_fold", [])) != {fold}:
        return None
    return d.to_dict("records")


def run(args):
    root = Path(args.root).resolve()
    out = root / "08_v2_systematic_selection/complete_nested_v1_0_1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "inner").mkdir(exist_ok=True)
    (out / "outer_predictions").mkdir(exist_ok=True)
    (out / "outer").mkdir(exist_ok=True)

    core, surf, mc, sg, src = source_audit(root, out)
    Z, defs, full_idx, y, pdb, chain, resid = build_complete_matrix(core, surf, mc, sg)
    nad = len(y)
    if nad != 431226:
        raise SystemExit("[FATAL] realized domain row count mismatch")

    split_audit = []
    inner_rows = []
    outer_rows = []
    device = os.environ.get("XGB_DEVICE", "cuda")
    threads = int(os.environ.get("THREADS", "32"))

    for outer_seed in SEEDS:
        split_npy = root / f"05_ligand_full_core/splits/pdb_split_seed{outer_seed}.npy"
        if not split_npy.exists():
            raise SystemExit(f"[FATAL] missing {split_npy}")
        code = np.load(split_npy, allow_pickle=False)
        if pdb.min() < 0 or pdb.max() >= len(code):
            raise SystemExit("[FATAL] row_pdb_idx outside split code range")
        rc = code[pdb]
        tr0 = np.flatnonzero(rc == 0)
        va0 = np.flatnonzero(rc == 1)
        te0 = np.flatnonzero(rc == 2)
        for nm, ix in [("train", tr0), ("val", va0), ("test", te0)]:
            split_audit.append({
                "outer_seed": outer_seed,
                "level": "outer",
                "inner_fold": -1,
                "split": nm,
                "rows": len(ix),
                "positive_rows": int(y[ix].sum()),
                "pdbs": int(len(np.unique(pdb[ix]))),
            })
            if len(np.unique(y[ix])) < 2:
                raise SystemExit(f"[FATAL] seed {outer_seed} {nm} lacks both classes")

        # v1.0.1 bug fix: define inner-fold codes ONLY on original outer-train rows.
        # Outer val/test PDBs intentionally remain -1 and are never looked up in a train-only map.
        inner_code = np.full(nad, -1, dtype=np.int8)
        inner_code[tr0] = np.fromiter(
            (stable_fold(int(pp), outer_seed, INNER_FOLDS) for pp in pdb[tr0]),
            dtype=np.int8,
            count=len(tr0),
        )
        if np.any(inner_code[va0] != -1) or np.any(inner_code[te0] != -1):
            raise SystemExit("[FATAL] outer val/test received inner-fold assignments")

        seed_inner_rows = []
        for fold in range(INNER_FOLDS):
            ck = out / "inner" / f"seed{outer_seed}_fold{fold}.csv"
            old = valid_fold_checkpoint(ck, outer_seed, fold)
            if old is not None:
                seed_inner_rows.extend(old)
                print(f"[RESUME] outer={outer_seed} inner_fold={fold} checkpoint complete", flush=True)
                continue

            score_idx = np.flatnonzero((rc == 0) & (inner_code == fold))
            fit_pool = np.flatnonzero((rc == 0) & (inner_code != fold))
            fit_idx, es_idx = split_fit_es_by_pdb(fit_pool, pdb, y, outer_seed, fold, INNER_ES_PDB_FRACTION)

            # Strong separation audit.
            p_fit = set(map(int, np.unique(pdb[fit_idx])))
            p_es = set(map(int, np.unique(pdb[es_idx])))
            p_score = set(map(int, np.unique(pdb[score_idx])))
            if p_fit & p_es or p_fit & p_score or p_es & p_score:
                raise SystemExit(f"[FATAL] inner PDB overlap seed={outer_seed} fold={fold}")
            if np.any(rc[fit_idx] != 0) or np.any(rc[es_idx] != 0) or np.any(rc[score_idx] != 0):
                raise SystemExit(f"[FATAL] inner rows escaped original outer-train seed={outer_seed} fold={fold}")
            for nm, ix in [("fit", fit_idx), ("early_stop", es_idx), ("score", score_idx)]:
                if len(np.unique(y[ix])) < 2:
                    raise SystemExit(f"[FATAL] inner {nm} lacks both classes seed={outer_seed} fold={fold}")
                split_audit.append({
                    "outer_seed": outer_seed,
                    "level": "inner",
                    "inner_fold": fold,
                    "split": nm,
                    "rows": len(ix),
                    "positive_rows": int(y[ix].sum()),
                    "pdbs": int(len(np.unique(pdb[ix]))),
                })

            fold_rows = []
            for rep in BASE_ORDER:
                cols = defs[rep]
                Xfit = subset(Z, fit_idx, cols)
                Xes = subset(Z, es_idx, cols)
                Xscore = subset(Z, score_idx, cols)
                yfit = y[fit_idx]
                yes = y[es_idx]
                yscore = y[score_idx]
                fit_seed = outer_seed * 100 + fold

                base_model = fit_xgb(Xfit, yfit, Xes, yes, fit_seed, device=device, threads=threads)
                pscore = base_model.predict_proba(Xscore)[:, 1]
                mm = metrics(yscore, pscore)
                row = {
                    "outer_seed": outer_seed,
                    "inner_fold": fold,
                    "representation": rep,
                    "hnm": False,
                    "arm": rep,
                    "dim": len(cols),
                    "fit_rows": len(fit_idx),
                    "es_rows": len(es_idx),
                    "score_rows": len(score_idx),
                    "score_positive": int(yscore.sum()),
                    "AUPR": mm["AUPR"],
                    "AUC": mm["AUC"],
                    "best_iteration": int(getattr(base_model, "best_iteration", -1)),
                    "hard_negatives": 0,
                }
                fold_rows.append(row)

                # HNM ranking is restricted to inner-fit rows. The stage-1 model itself
                # uses the separate inner early-stop split; the untouched score fold is
                # never used for stage-1 fitting, early stopping, HNM ranking, or weighting.
                pfit = base_model.predict_proba(Xfit)[:, 1]
                hmask = hard_negative_mask(yfit, pfit, chain[fit_idx], HNM_TOPK)
                w = np.ones(len(fit_idx), dtype=np.float32)
                w[hmask] = HNM_WEIGHT
                hm = fit_xgb(Xfit, yfit, Xes, yes, fit_seed, sample_weight=w, device=device, threads=threads)
                ph = hm.predict_proba(Xscore)[:, 1]
                mh = metrics(yscore, ph)
                rowh = {
                    "outer_seed": outer_seed,
                    "inner_fold": fold,
                    "representation": rep,
                    "hnm": True,
                    "arm": rep + "+HNM",
                    "dim": len(cols),
                    "fit_rows": len(fit_idx),
                    "es_rows": len(es_idx),
                    "score_rows": len(score_idx),
                    "score_positive": int(yscore.sum()),
                    "AUPR": mh["AUPR"],
                    "AUC": mh["AUC"],
                    "best_iteration": int(getattr(hm, "best_iteration", -1)),
                    "hard_negatives": int(hmask.sum()),
                }
                fold_rows.append(rowh)
                del Xfit, Xes, Xscore, base_model, hm

            fdf = pd.DataFrame(fold_rows)
            if len(fdf) != 14 or set(fdf.arm) != set(EXPECTED_ARMS):
                raise SystemExit(f"[FATAL] incomplete inner fold seed={outer_seed} fold={fold}")
            fdf.to_csv(ck, index=False)
            seed_inner_rows.extend(fold_rows)
            print(f"[INNER] outer={outer_seed} fold={fold} complete arms=14", flush=True)

        inner_rows.extend(seed_inner_rows)
        idf = pd.DataFrame(seed_inner_rows)
        if len(idf) != INNER_FOLDS * 14:
            raise SystemExit(f"[FATAL] seed {outer_seed} inner rows={len(idf)} expected={INNER_FOLDS*14}")
        summ = idf.groupby(["arm", "representation", "hnm", "dim"], as_index=False).agg(
            inner_AUPR_mean=("AUPR", "mean"),
            inner_AUPR_std=("AUPR", "std"),
            inner_AUC_mean=("AUC", "mean"),
            inner_best_iteration_median=("best_iteration", "median"),
        )
        summ["standard_priority"] = (~summ["hnm"]).astype(int)
        summ = summ.sort_values(
            ["inner_AUPR_mean", "dim", "standard_priority", "arm"],
            ascending=[False, True, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        summ.to_csv(out / f"INNER_SUMMARY_seed{outer_seed}.csv", index=False)
        win = summ.iloc[0]

        outer_json = out / "outer" / f"seed{outer_seed}_selected_result.json"
        pred_path = out / "outer_predictions" / f"seed{outer_seed}.csv.gz"
        if outer_json.exists() and pred_path.exists():
            old = jload(outer_json)
            if int(old.get("outer_seed", -1)) == outer_seed and old.get("selected_arm") == str(win["arm"]):
                outer_rows.append(old)
                print(
                    f"[RESUME OUTER {outer_seed}] selected={old['selected_arm']} "
                    f"test_AUPR={old['outer_test_AUPR']:.6f} test_AUC={old['outer_test_AUC']:.6f}",
                    flush=True,
                )
                continue

        rep = str(win["representation"])
        use_hnm = bool(win["hnm"])
        cols = defs[rep]
        Xtr = subset(Z, tr0, cols)
        Xva = subset(Z, va0, cols)
        Xte = subset(Z, te0, cols)
        fit_seed = outer_seed
        base_model = fit_xgb(Xtr, y[tr0], Xva, y[va0], fit_seed, device=device, threads=threads)
        hcount = 0
        final_model = base_model
        if use_hnm:
            ptr = base_model.predict_proba(Xtr)[:, 1]
            hmask = hard_negative_mask(y[tr0], ptr, chain[tr0], HNM_TOPK)
            w = np.ones(len(tr0), dtype=np.float32)
            w[hmask] = HNM_WEIGHT
            hcount = int(hmask.sum())
            final_model = fit_xgb(Xtr, y[tr0], Xva, y[va0], fit_seed, sample_weight=w, device=device, threads=threads)
        pte = final_model.predict_proba(Xte)[:, 1]
        mt = metrics(y[te0], pte)
        orec = {
            "outer_seed": outer_seed,
            "selected_arm": str(win["arm"]),
            "representation": rep,
            "hnm": use_hnm,
            "dim": len(cols),
            "inner_AUPR_mean": float(win["inner_AUPR_mean"]),
            "inner_AUC_mean": float(win["inner_AUC_mean"]),
            "outer_test_AUPR": mt["AUPR"],
            "outer_test_AUC": mt["AUC"],
            "best_iteration": int(getattr(final_model, "best_iteration", -1)),
            "hard_negatives": hcount,
            "train_rows": len(tr0),
            "es_val_rows": len(va0),
            "test_rows": len(te0),
        }
        outer_rows.append(orec)
        dumpj(outer_json, orec)
        pd.DataFrame({
            "full_row_index": full_idx[te0],
            "pdb_index": pdb[te0],
            "chain_index": chain[te0],
            "residue_index": resid[te0],
            "label": y[te0],
            "pred": pte,
        }).to_csv(pred_path, index=False, compression="gzip")
        del Xtr, Xva, Xte, base_model, final_model
        print(
            f"[OUTER {outer_seed}] selected={win['arm']} test_AUPR={mt['AUPR']:.6f} test_AUC={mt['AUC']:.6f}",
            flush=True,
        )

    idf = pd.DataFrame(inner_rows)
    odf = pd.DataFrame(outer_rows)
    sadf = pd.DataFrame(split_audit).drop_duplicates()
    idf.to_csv(out / "INNER_CANDIDATE_METRICS.csv", index=False)
    odf.to_csv(out / "OUTER_SELECTED_RESULTS.csv", index=False)
    sadf.to_csv(out / "SPLIT_AUDIT.csv", index=False)

    # Development consensus: inner score folds only; outer-test metrics are never used to pick it.
    cons = idf.groupby(["arm", "representation", "hnm", "dim"], as_index=False).agg(
        inner_AUPR_mean=("AUPR", "mean"),
        inner_AUPR_std=("AUPR", "std"),
        inner_AUC_mean=("AUC", "mean"),
        n_inner=("AUPR", "size"),
    )
    cons["standard_priority"] = (~cons["hnm"]).astype(int)
    cons = cons.sort_values(
        ["inner_AUPR_mean", "dim", "standard_priority", "arm"],
        ascending=[False, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    cons.to_csv(out / "CONSENSUS_INNER_CV_RANKING.csv", index=False)
    c = cons.iloc[0]
    sel = odf["selected_arm"].value_counts().rename_axis("arm").reset_index(name="outer_selection_count")
    sel.to_csv(out / "SELECTION_FREQUENCY.csv", index=False)

    consensus = {
        "gate": "PASS_STAGE08B_COMPLETE_CONSENSUS_ARM_FROZEN_FOR_FUTURE_EXTERNAL_CONFIRMATION",
        "gate_pass": True,
        "selected_arm": str(c["arm"]),
        "representation": str(c["representation"]),
        "hnm": bool(c["hnm"]),
        "dim_with_base_onehot": int(c["dim"]),
        "selection_metric": "mean untouched inner-score PDB-disjoint AUPRC across 5 outer seeds x 3 folds",
        "inner_AUPR_mean": float(c["inner_AUPR_mean"]),
        "inner_AUC_mean": float(c["inner_AUC_mean"]),
        "outer_test_metrics_used_for_consensus_selection": False,
        "test17_used_for_selection": False,
        "historical_outer_test_results_previously_seen": True,
        "future_untouched_external_confirmation_required_for_fixed_arm_claim": True,
    }
    dumpj(out / "CONSENSUS_COMPLETE_ARM.json", consensus)

    gate = {
        "gate": "PASS_STAGE08B_COMPLETE_NESTED_SELECTION_V101",
        "gate_pass": True,
        "analysis_domain": "Stage05A realized SURF applicability domain",
        "rows": 431226,
        "surf_dim": 57,
        "candidate_representations": BASE_ORDER,
        "training_modes": ["standard", "HNM"],
        "candidate_arms_n": 14,
        "outer_seeds": SEEDS,
        "inner_folds": INNER_FOLDS,
        "inner_protocol": "outer-train only: PDB-disjoint inner score fold; remaining PDBs split again into PDB-disjoint fit and early-stop subsets",
        "inner_early_stop_pdb_fraction": INNER_ES_PDB_FRACTION,
        "candidate_selection_data": "untouched inner-score folds only",
        "final_early_stopping_data": "original frozen outer-validation only",
        "outer_test_used_for_selection": False,
        "test17_used_for_selection": False,
        "historical_outer_test_results_previously_seen": True,
        "B_factor_included": False,
        "base_onehot_appended_to_every_arm": True,
        "HNM_rule": {
            "top_negative_per_training_chain": 20,
            "hard_negative_weight": 5.0,
            "hard_negative_ranking_rows": "fit/train rows only",
            "stage1_early_stopping": "separate early-stop split",
            "inner_score_used_for_hnm": False,
            "outer_test_used_for_hnm": False,
            "test17_used_for_hnm": False,
        },
        "v100_failure": "aborted before model training: train-only pdb2fold map was incorrectly applied to all SURF rows, causing KeyError on outer val/test PDB indices",
        "v101_fix": "inner fold codes are assigned only to original outer-train rows; outer val/test remain -1; inner early stopping is separated from inner scoring",
        "resumable_inner_fold_checkpoints": True,
        "nested_selected_outer_AUPR_mean": float(odf.outer_test_AUPR.mean()),
        "nested_selected_outer_AUPR_std": float(odf.outer_test_AUPR.std(ddof=0)),
        "nested_selected_outer_AUC_mean": float(odf.outer_test_AUC.mean()),
        "nested_selected_outer_AUC_std": float(odf.outer_test_AUC.std(ddof=0)),
        "outer_selected_arms": odf[["outer_seed", "selected_arm"]].to_dict("records"),
        "consensus_complete_arm": consensus["selected_arm"],
        "consensus_selection_uses_outer_test": False,
        "source_binding_gate": src["gate"],
        "next_stage": "Stage08C full-universe Core selection among A/C/AC x standard/HNM, then Stage08D routed evaluation.",
    }
    dumpj(out / "STAGE08B_COMPLETE_NESTED_GATE.json", gate)
    print(json.dumps(gate, indent=2), flush=True)


def selftest():
    y = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=np.uint8)
    p = np.array([.1, .9, .8, .7, .3, .2, .6, .4, .5, .05])
    ch = np.array([1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
    m = hard_negative_mask(y, p, ch, 2)
    assert set(np.flatnonzero(m)) == {1, 2, 6, 7}

    # Regression test for the v1.0.0 KeyError: only train rows receive inner codes.
    fake_pdb = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int32)
    fake_rc = np.array([0, 0, 0, 0, 1, 1, 2, 2, 0, 0], dtype=np.int8)
    tr = np.flatnonzero(fake_rc == 0)
    va = np.flatnonzero(fake_rc == 1)
    te = np.flatnonzero(fake_rc == 2)
    inner = np.full(len(fake_pdb), -1, dtype=np.int8)
    inner[tr] = np.fromiter((stable_fold(int(pp), 3407, 3) for pp in fake_pdb[tr]), dtype=np.int8, count=len(tr))
    assert np.all(inner[va] == -1) and np.all(inner[te] == -1)
    assert np.all(inner[tr] >= 0)

    # Deterministic and PDB-disjoint fit/ES split test with both classes in each stratum.
    fpdb = np.repeat(np.arange(20), 2)
    fy = np.zeros(40, dtype=np.uint8)
    for pp in [0, 1, 2, 3, 4, 5]:
        fy[np.flatnonzero(fpdb == pp)[0]] = 1
    pool = np.arange(40)
    fit1, es1 = split_fit_es_by_pdb(pool, fpdb, fy, 3407, 0, 0.2)
    fit2, es2 = split_fit_es_by_pdb(pool, fpdb, fy, 3407, 0, 0.2)
    assert np.array_equal(fit1, fit2) and np.array_equal(es1, es2)
    assert not set(np.unique(fpdb[fit1])) & set(np.unique(fpdb[es1]))
    assert len(np.unique(fy[fit1])) == 2 and len(np.unique(fy[es1])) == 2

    print("[PASS] Stage08B v1.0.1 KeyError regression + PDB-disjoint inner ES + HNM selftest")


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
