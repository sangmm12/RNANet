#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
BASE_ORDER = ["A", "C", "AC"]
EXPECTED_ARMS = [x for r in BASE_ORDER for x in (r, r + "+HNM")]
EXPECTED_DIMS = {"A": 50, "C": 46, "AC": 90}
FULL_ROWS = 6_252_187
FULL_DIM = 96

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
    """Frozen Stage08B HNM rule: top-k predicted negatives per training chain.

    Input rows are expected to preserve full-matrix order. If chain ids are not
    monotone after negative filtering, a deterministic stable sort is applied.
    """
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
        # deterministic: highest prediction, then smallest local row index
        oo = np.lexsort((idx, -pred[idx]))[:k]
        out[idx[oo]] = True
    return out


def core_arm_defs(core_contract):
    fs = core_contract["feature_sets"]
    A = list(map(int, fs["CY-A"]))
    C = list(map(int, fs["CY-C"]))
    AC = list(map(int, fs["AC"]))
    dims = core_contract["noB_dimensions"]
    if len(A) != int(dims["CY-A"]) or len(A) != 46:
        raise SystemExit(f"[FATAL] CY-A dim mismatch: {len(A)}")
    if len(C) != int(dims["CY-C"]) or len(C) != 42:
        raise SystemExit(f"[FATAL] CY-C dim mismatch: {len(C)}")
    if len(AC) != int(dims["AC"]) or len(AC) != 86:
        raise SystemExit(f"[FATAL] AC dim mismatch: {len(AC)}")
    core_pos = {orig: i for i, orig in enumerate(AC)}
    if len(core_pos) != len(AC):
        raise SystemExit("[FATAL] duplicate source columns in AC feature set")
    for x in set(A + C):
        if x not in core_pos:
            raise SystemExit(f"[FATAL] A/C col {x} absent from AC")
    Apos = [core_pos[x] for x in A]
    Cpos = [core_pos[x] for x in C]
    ACpos = list(range(len(AC)))
    bpos = list(range(len(AC), len(AC) + 4))
    defs = {
        "A": Apos + bpos,
        "C": Cpos + bpos,
        "AC": ACpos + bpos,
    }
    got = {k: len(v) for k, v in defs.items()}
    if got != EXPECTED_DIMS:
        raise SystemExit(f"[FATAL] arm dimension contract mismatch: got={got} expected={EXPECTED_DIMS}")
    for k, v in defs.items():
        if len(v) != len(set(v)):
            raise SystemExit(f"[FATAL] duplicate columns in {k}")
    return defs, AC


def stable_fold(pdb_index: int, outer_seed: int, nfold=3) -> int:
    h = hashlib.sha256(f"RNANET_STAGE08C_INNER_SCORE|{outer_seed}|{pdb_index}".encode()).digest()
    return int.from_bytes(h[:8], "little") % int(nfold)


def stable_rank(tag: str, seed: int, fold: int, pdb_index: int) -> int:
    h = hashlib.sha256(f"{tag}|{seed}|{fold}|{pdb_index}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def assign_inner_codes_only_to_outer_train(nrows, tr0, pdb, outer_seed, nfold=3):
    """Efficient exact PDB-level assignment; val/test rows remain -1."""
    inner_code = np.full(int(nrows), -1, dtype=np.int8)
    train_pdbs = np.unique(pdb[tr0]).astype(np.int64)
    if len(train_pdbs) == 0:
        raise SystemExit("[FATAL] no outer-train PDBs")
    # row_pdb_idx is an integer index into the frozen split vector. A compact map
    # avoids hashing millions of rows while preserving the same PDB-level rule.
    max_pdb = int(np.max(pdb))
    p2f = np.full(max_pdb + 1, -1, dtype=np.int8)
    p2f[train_pdbs] = np.fromiter(
        (stable_fold(int(pp), outer_seed, nfold) for pp in train_pdbs),
        dtype=np.int8,
        count=len(train_pdbs),
    )
    inner_code[tr0] = p2f[pdb[tr0]]
    if np.any(inner_code[tr0] < 0):
        raise SystemExit("[FATAL] some outer-train rows lack inner-fold assignment")
    return inner_code


def split_fit_es_by_pdb(pool_idx, pdb, y, outer_seed, inner_fold, frac=0.10):
    """PDB-disjoint fit/early-stop split inside the inner-fit pool.

    Stratification uses only whether a PDB contains >=1 positive row in the
    current inner-fit pool; the untouched inner-score fold is not consulted.
    """
    pool_idx = np.asarray(pool_idx, dtype=np.int64)
    pool_pdb = np.asarray(pdb[pool_idx], dtype=np.int64)
    uniq, inv = np.unique(pool_pdb, return_inverse=True)
    if len(uniq) < 10:
        raise SystemExit(f"[FATAL] too few PDBs for inner fit/ES split: {len(uniq)}")
    has_pos = np.zeros(len(uniq), dtype=np.uint8)
    np.maximum.at(has_pos, inv, np.asarray(y[pool_idx], dtype=np.uint8))
    pos_pdb = uniq[has_pos == 1].astype(np.int64).tolist()
    neg_pdb = uniq[has_pos == 0].astype(np.int64).tolist()
    if len(pos_pdb) < 2 or len(neg_pdb) < 2:
        raise SystemExit(
            f"[FATAL] insufficient positive/negative PDB strata for inner ES split: "
            f"pos_pdb={len(pos_pdb)} neg_pdb={len(neg_pdb)}"
        )

    def choose(group):
        ordered = sorted(
            map(int, group),
            key=lambda pp: stable_rank("RNANET_STAGE08C_INNER_ES", outer_seed, inner_fold, pp),
        )
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
            f"fit_pos={int(y[fit_idx].sum())}/{len(fit_idx)} "
            f"es_pos={int(y[es_idx].sum())}/{len(es_idx)}"
        )
    return fit_idx, es_idx


def build_core_matrix(core_dir: Path, mc, chunk_rows: int = 250_000):
    n = int(mc["rows"])
    d = int(mc["matrix_shape"][1])
    X = np.memmap(core_dir / "X_intrinsic.f32", np.float32, "r", shape=(n, d))
    base = np.memmap(core_dir / "base.u8", np.uint8, "r", shape=(n,))
    y = np.memmap(core_dir / "y.u8", np.uint8, "r", shape=(n,))
    pdb = np.memmap(core_dir / "row_pdb_idx.i32", np.int32, "r", shape=(n,))
    chain = np.memmap(core_dir / "row_chain_idx.i32", np.int32, "r", shape=(n,))
    resid = np.memmap(core_dir / "residue_index.i32", np.int32, "r", shape=(n,))
    defs, ac_orig = core_arm_defs(mc)

    # Materialize only frozen no-B AC source columns plus the 4D base one-hot.
    # Chunking avoids an additional multi-GB advanced-indexing temporary.
    Z = np.empty((n, len(ac_orig) + 4), dtype=np.float32)
    ac_cols = np.asarray(ac_orig, dtype=np.int64)
    for a in range(0, n, int(chunk_rows)):
        b = min(n, a + int(chunk_rows))
        Z[a:b, : len(ac_orig)] = np.asarray(X[a:b, :][:, ac_cols], dtype=np.float32)
        Z[a:b, len(ac_orig) :] = onehot_base(np.asarray(base[a:b]))
    if np.isinf(Z).any():
        raise SystemExit("[FATAL] +/-inf detected in full-core predictor matrix")
    return Z, defs, np.asarray(y), np.asarray(pdb), np.asarray(chain), np.asarray(resid)


def source_audit(root: Path, out: Path):
    core = root / "05_ligand_full_core/matrix"
    req = [
        "matrix_contract.json",
        "X_intrinsic.f32",
        "y.u8",
        "base.u8",
        "residue_index.i32",
        "row_chain_idx.i32",
        "row_pdb_idx.i32",
        "chain_row_ranges.csv",
    ]
    missing = [str(core / x) for x in req if not (core / x).exists()]
    split_files = [root / f"05_ligand_full_core/splits/pdb_split_seed{s}.npy" for s in SEEDS]
    missing += [str(p) for p in split_files if not p.exists()]

    bdir = root / "08_v2_systematic_selection/complete_nested_v1_0_1"
    b_gate_p = bdir / "STAGE08B_COMPLETE_NESTED_GATE.json"
    b_cons_p = bdir / "CONSENSUS_COMPLETE_ARM.json"
    missing += [str(p) for p in [b_gate_p, b_cons_p] if not p.exists()]
    if missing:
        raise SystemExit("[FATAL] missing exact source/upstream files:\n" + "\n".join(missing))

    mc = jload(core / "matrix_contract.json")
    if not mc.get("gate_pass"):
        raise SystemExit("[FATAL] matrix_contract gate_pass is false")
    if mc.get("matrix_shape") != [FULL_ROWS, FULL_DIM]:
        raise SystemExit(f"[FATAL] full-core matrix shape mismatch: {mc.get('matrix_shape')}")
    if mc.get("B_factor_primary") is not False:
        raise SystemExit("[FATAL] B-factor unexpectedly enabled")
    dims = mc.get("noB_dimensions", {})
    if int(dims.get("CY-A", -1)) != 46 or int(dims.get("CY-C", -1)) != 42 or int(dims.get("AC", -1)) != 86:
        raise SystemExit(f"[FATAL] no-B dimensions mismatch: {dims}")

    expected = {
        core / "X_intrinsic.f32": FULL_ROWS * FULL_DIM * 4,
        core / "y.u8": FULL_ROWS,
        core / "base.u8": FULL_ROWS,
        core / "residue_index.i32": FULL_ROWS * 4,
        core / "row_chain_idx.i32": FULL_ROWS * 4,
        core / "row_pdb_idx.i32": FULL_ROWS * 4,
    }
    bad = []
    for p, nbytes in expected.items():
        if p.stat().st_size != nbytes:
            bad.append((str(p), p.stat().st_size, nbytes))
    if bad:
        raise SystemExit("[FATAL] raw memmap size mismatch: " + repr(bad))

    b_gate = jload(b_gate_p)
    b_cons = jload(b_cons_p)
    if not b_gate.get("gate_pass") or b_gate.get("gate") != "PASS_STAGE08B_COMPLETE_NESTED_SELECTION_V101":
        raise SystemExit("[FATAL] Stage08B v1.0.1 gate not PASS")
    if not b_cons.get("gate_pass") or b_cons.get("selected_arm") != "ACS+HNM":
        raise SystemExit(f"[FATAL] unexpected Stage08B consensus: {b_cons.get('selected_arm')}")
    if b_cons.get("outer_test_metrics_used_for_consensus_selection") is not False:
        raise SystemExit("[FATAL] Stage08B consensus reports outer-test use")

    split_hashes = {str(s): sha256_file(p) for s, p in zip(SEEDS, split_files)}
    src = {
        "gate": "PASS_STAGE08C_EXACT_FULL_CORE_SOURCE_BINDING",
        "gate_pass": True,
        "analysis_domain": "Stage05A full Core universe",
        "full_core_rows": FULL_ROWS,
        "full_core_matrix_dim": FULL_DIM,
        "core_candidate_dims_before_base_onehot": {"A": 46, "C": 42, "AC": 86},
        "core_candidate_dims_with_base_onehot": EXPECTED_DIMS,
        "B_factor_primary": False,
        "SURF_used_for_core_selection": False,
        "test17_used_for_selection": False,
        "matrix_contract_sha256": sha256_file(core / "matrix_contract.json"),
        "outer_split_sha256": split_hashes,
        "upstream_stage08b_gate_sha256": sha256_file(b_gate_p),
        "upstream_stage08b_consensus_sha256": sha256_file(b_cons_p),
        "upstream_complete_consensus_arm": b_cons["selected_arm"],
    }
    dumpj(out / "STAGE08C_SOURCE_BINDING.json", src)
    return core, mc, src, b_cons


def subset(Z, idx, cols):
    idx = np.asarray(idx, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    return np.ascontiguousarray(Z[idx[:, None], cols[None, :]], dtype=np.float32)


def valid_fold_checkpoint(p: Path, outer_seed: int, fold: int):
    if not p.exists():
        return None
    try:
        d = pd.read_csv(p)
    except Exception:
        return None
    if len(d) != len(EXPECTED_ARMS) or set(d.get("arm", [])) != set(EXPECTED_ARMS):
        return None
    if set(d.get("outer_seed", [])) != {outer_seed} or set(d.get("inner_fold", [])) != {fold}:
        return None
    return d.to_dict("records")


def audit_split_row(outer_seed, level, inner_fold, split, ix, y, pdb):
    n = int(len(ix))
    pos = int(y[ix].sum())
    return {
        "outer_seed": int(outer_seed),
        "level": level,
        "inner_fold": int(inner_fold),
        "split": split,
        "rows": n,
        "positive_rows": pos,
        "positive_fraction": (float(pos) / float(n)) if n else float("nan"),
        "pdbs": int(len(np.unique(pdb[ix]))),
    }


def run(args):
    root = Path(args.root).resolve()
    out = root / "08_v2_systematic_selection/full_core_nested_v1_0_0"
    out.mkdir(parents=True, exist_ok=True)
    (out / "inner").mkdir(exist_ok=True)
    (out / "outer_predictions").mkdir(exist_ok=True)
    (out / "outer").mkdir(exist_ok=True)

    core, mc, src, b_cons = source_audit(root, out)
    Z, defs, y, pdb, chain, resid = build_core_matrix(core, mc)
    n = len(y)
    if n != FULL_ROWS:
        raise SystemExit(f"[FATAL] full-core row count mismatch: {n}")

    split_audit = []
    inner_rows = []
    outer_rows = []
    device = os.environ.get("XGB_DEVICE", "cuda")
    threads = int(os.environ.get("THREADS", "32"))

    for outer_seed in SEEDS:
        split_npy = root / f"05_ligand_full_core/splits/pdb_split_seed{outer_seed}.npy"
        code = np.load(split_npy, allow_pickle=False)
        if pdb.min() < 0 or pdb.max() >= len(code):
            raise SystemExit("[FATAL] row_pdb_idx outside split code range")
        rc = code[pdb]
        tr0 = np.flatnonzero(rc == 0)
        va0 = np.flatnonzero(rc == 1)
        te0 = np.flatnonzero(rc == 2)
        for nm, ix in [("train", tr0), ("val", va0), ("test", te0)]:
            split_audit.append(audit_split_row(outer_seed, "outer", -1, nm, ix, y, pdb))
            if len(np.unique(y[ix])) < 2:
                raise SystemExit(f"[FATAL] seed {outer_seed} {nm} lacks both classes")

        inner_code = assign_inner_codes_only_to_outer_train(n, tr0, pdb, outer_seed, INNER_FOLDS)
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
            fit_idx, es_idx = split_fit_es_by_pdb(
                fit_pool, pdb, y, outer_seed, fold, INNER_ES_PDB_FRACTION
            )

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
                split_audit.append(audit_split_row(outer_seed, "inner", fold, nm, ix, y, pdb))

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

                base_model = fit_xgb(
                    Xfit, yfit, Xes, yes, fit_seed, device=device, threads=threads
                )
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
                    "score_positive_fraction": float(yscore.mean()),
                    "AUPR": mm["AUPR"],
                    "AUC": mm["AUC"],
                    "best_iteration": int(getattr(base_model, "best_iteration", -1)),
                    "hard_negatives": 0,
                }
                fold_rows.append(row)

                # Frozen HNM: stage-1 ranking uses fit rows only. The untouched
                # score fold is never used for fitting, early stopping, ranking,
                # weighting, or thresholding.
                pfit_pred = base_model.predict_proba(Xfit)[:, 1]
                hmask = hard_negative_mask(yfit, pfit_pred, chain[fit_idx], HNM_TOPK)
                w = np.ones(len(fit_idx), dtype=np.float32)
                w[hmask] = HNM_WEIGHT
                hm = fit_xgb(
                    Xfit,
                    yfit,
                    Xes,
                    yes,
                    fit_seed,
                    sample_weight=w,
                    device=device,
                    threads=threads,
                )
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
                    "score_positive_fraction": float(yscore.mean()),
                    "AUPR": mh["AUPR"],
                    "AUC": mh["AUC"],
                    "best_iteration": int(getattr(hm, "best_iteration", -1)),
                    "hard_negatives": int(hmask.sum()),
                }
                fold_rows.append(rowh)
                del Xfit, Xes, Xscore, base_model, hm, pscore, pfit_pred, ph, w, hmask

            fdf = pd.DataFrame(fold_rows)
            if len(fdf) != len(EXPECTED_ARMS) or set(fdf.arm) != set(EXPECTED_ARMS):
                raise SystemExit(f"[FATAL] incomplete inner fold seed={outer_seed} fold={fold}")
            fdf.to_csv(ck, index=False)
            seed_inner_rows.extend(fold_rows)
            print(f"[INNER] outer={outer_seed} fold={fold} complete arms={len(EXPECTED_ARMS)}", flush=True)

        inner_rows.extend(seed_inner_rows)
        idf = pd.DataFrame(seed_inner_rows)
        if len(idf) != INNER_FOLDS * len(EXPECTED_ARMS):
            raise SystemExit(
                f"[FATAL] seed {outer_seed} inner rows={len(idf)} "
                f"expected={INNER_FOLDS * len(EXPECTED_ARMS)}"
            )
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
                    f"test_AUPR={old['outer_test_AUPR']:.6f} "
                    f"test_AUC={old['outer_test_AUC']:.6f}",
                    flush=True,
                )
                continue

        rep = str(win["representation"])
        use_hnm = bool(win["hnm"])
        cols = defs[rep]
        Xtr = subset(Z, tr0, cols)
        Xva = subset(Z, va0, cols)
        Xte = subset(Z, te0, cols)
        base_model = fit_xgb(Xtr, y[tr0], Xva, y[va0], outer_seed, device=device, threads=threads)
        hcount = 0
        final_model = base_model
        if use_hnm:
            ptr = base_model.predict_proba(Xtr)[:, 1]
            hmask = hard_negative_mask(y[tr0], ptr, chain[tr0], HNM_TOPK)
            w = np.ones(len(tr0), dtype=np.float32)
            w[hmask] = HNM_WEIGHT
            hcount = int(hmask.sum())
            final_model = fit_xgb(
                Xtr,
                y[tr0],
                Xva,
                y[va0],
                outer_seed,
                sample_weight=w,
                device=device,
                threads=threads,
            )
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
            "outer_test_positive_fraction": float(y[te0].mean()),
            "best_iteration": int(getattr(final_model, "best_iteration", -1)),
            "hard_negatives": hcount,
            "train_rows": len(tr0),
            "es_val_rows": len(va0),
            "test_rows": len(te0),
        }
        outer_rows.append(orec)
        dumpj(outer_json, orec)
        pd.DataFrame(
            {
                "full_row_index": te0,
                "pdb_index": pdb[te0],
                "chain_index": chain[te0],
                "residue_index": resid[te0],
                "label": y[te0],
                "pred": pte,
            }
        ).to_csv(pred_path, index=False, compression="gzip")
        del Xtr, Xva, Xte, base_model, final_model, pte
        print(
            f"[OUTER {outer_seed}] selected={win['arm']} "
            f"test_AUPR={mt['AUPR']:.6f} test_AUC={mt['AUC']:.6f}",
            flush=True,
        )

    idf = pd.DataFrame(inner_rows)
    odf = pd.DataFrame(outer_rows)
    sadf = pd.DataFrame(split_audit).drop_duplicates()
    idf.to_csv(out / "INNER_CANDIDATE_METRICS.csv", index=False)
    odf.to_csv(out / "OUTER_SELECTED_RESULTS.csv", index=False)
    sadf.to_csv(out / "SPLIT_AUDIT.csv", index=False)

    # Consensus is development-only and uses only untouched inner-score folds.
    # Outer-test metrics are never used to choose the fixed Core arm.
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
        "gate": "PASS_STAGE08C_CORE_CONSENSUS_ARM_FROZEN_FOR_ROUTED_EVALUATION",
        "gate_pass": True,
        "selected_arm": str(c["arm"]),
        "representation": str(c["representation"]),
        "hnm": bool(c["hnm"]),
        "dim_with_base_onehot": int(c["dim"]),
        "selection_metric": "mean untouched inner-score PDB-disjoint AUPRC across 5 outer seeds x 3 folds",
        "inner_AUPR_mean": float(c["inner_AUPR_mean"]),
        "inner_AUC_mean": float(c["inner_AUC_mean"]),
        "outer_test_metrics_used_for_consensus_selection": False,
        "SURF_used_for_core_selection": False,
        "test17_used_for_selection": False,
        "historical_outer_test_results_previously_seen": True,
        "future_untouched_external_confirmation_required_for_fixed_arm_claim": True,
        "paired_complete_consensus_arm_for_stage08d": b_cons["selected_arm"],
    }
    dumpj(out / "CONSENSUS_CORE_ARM.json", consensus)

    gate = {
        "gate": "PASS_STAGE08C_FULL_CORE_NESTED_SELECTION_V100",
        "gate_pass": True,
        "analysis_domain": "Stage05A full Core universe",
        "rows": FULL_ROWS,
        "candidate_representations": BASE_ORDER,
        "training_modes": ["standard", "HNM"],
        "candidate_arms_n": len(EXPECTED_ARMS),
        "arm_dimensions_with_base_onehot": EXPECTED_DIMS,
        "outer_seeds": SEEDS,
        "inner_folds": INNER_FOLDS,
        "inner_protocol": "outer-train only: PDB-disjoint inner score fold; remaining PDBs split again into PDB-disjoint fit and early-stop subsets",
        "inner_early_stop_pdb_fraction": INNER_ES_PDB_FRACTION,
        "candidate_selection_data": "untouched inner-score folds only",
        "final_early_stopping_data": "original frozen outer-validation only",
        "outer_test_used_for_selection": False,
        "SURF_used_for_core_selection": False,
        "test17_used_for_selection": False,
        "historical_outer_test_results_previously_seen": True,
        "B_factor_included": False,
        "base_onehot_appended_to_every_arm": True,
        "HNM_rule": {
            "top_negative_per_training_chain": HNM_TOPK,
            "hard_negative_weight": HNM_WEIGHT,
            "hard_negative_ranking_rows": "fit/train rows only",
            "stage1_early_stopping": "separate early-stop split",
            "inner_score_used_for_hnm": False,
            "outer_test_used_for_hnm": False,
            "test17_used_for_hnm": False,
        },
        "resumable_inner_fold_checkpoints": True,
        "nested_selected_outer_AUPR_mean": float(odf.outer_test_AUPR.mean()),
        "nested_selected_outer_AUPR_std": float(odf.outer_test_AUPR.std(ddof=0)),
        "nested_selected_outer_AUC_mean": float(odf.outer_test_AUC.mean()),
        "nested_selected_outer_AUC_std": float(odf.outer_test_AUC.std(ddof=0)),
        "outer_selected_arms": odf[["outer_seed", "selected_arm"]].to_dict("records"),
        "consensus_core_arm": consensus["selected_arm"],
        "consensus_selection_uses_outer_test": False,
        "upstream_complete_consensus_arm": b_cons["selected_arm"],
        "source_binding_gate": src["gate"],
        "next_stage": "Stage08D fixed Core/Complete routed evaluation: SURF-domain rows use Stage08B ACS+HNM; all other full-core rows use Stage08C consensus Core arm.",
    }
    dumpj(out / "STAGE08C_FULL_CORE_NESTED_GATE.json", gate)
    print(json.dumps(gate, indent=2), flush=True)


def selftest():
    # Frozen HNM rule.
    y = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=np.uint8)
    p = np.array([0.1, 0.9, 0.8, 0.7, 0.3, 0.2, 0.6, 0.4, 0.5, 0.05])
    ch = np.array([1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
    m = hard_negative_mask(y, p, ch, 2)
    assert set(np.flatnonzero(m)) == {1, 2, 6, 7}

    # Same result if chains arrive non-monotonically.
    perm = np.array([4, 5, 0, 1, 6, 2, 7, 3, 8, 9])
    mm = hard_negative_mask(y[perm], p[perm], ch[perm], 2)
    picked_original = set(perm[np.flatnonzero(mm)])
    assert picked_original == {1, 2, 6, 7}

    # Regression: only outer-train rows get inner codes.
    fake_pdb = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int32)
    fake_rc = np.array([0, 0, 0, 0, 1, 1, 2, 2, 0, 0], dtype=np.int8)
    tr = np.flatnonzero(fake_rc == 0)
    va = np.flatnonzero(fake_rc == 1)
    te = np.flatnonzero(fake_rc == 2)
    inner = assign_inner_codes_only_to_outer_train(len(fake_pdb), tr, fake_pdb, 3407, 3)
    assert np.all(inner[va] == -1) and np.all(inner[te] == -1)
    assert np.all(inner[tr] >= 0)
    for pp in np.unique(fake_pdb[tr]):
        rows = tr[fake_pdb[tr] == pp]
        assert len(np.unique(inner[rows])) == 1

    # Efficient PDB-disjoint fit/ES split is deterministic and class-valid.
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

    # Feature-manifest regression: A=46, C=42, union AC=86 (2 shared cols),
    # then append the frozen 4D base one-hot exactly once per arm.
    A = list(range(46))
    C = [0, 1] + list(range(46, 86))
    AC = list(range(86))
    fake_contract = {
        "feature_sets": {"CY-A": A, "CY-C": C, "AC": AC},
        "noB_dimensions": {"CY-A": 46, "CY-C": 42, "AC": 86},
    }
    fake_defs, fake_ac = core_arm_defs(fake_contract)
    assert len(fake_ac) == 86
    assert {k: len(v) for k, v in fake_defs.items()} == {"A": 50, "C": 46, "AC": 90}
    assert EXPECTED_DIMS == {"A": 50, "C": 46, "AC": 90}
    assert EXPECTED_ARMS == ["A", "A+HNM", "C", "C+HNM", "AC", "AC+HNM"]

    print("[PASS] Stage08C full-core nested selection selftest")


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
