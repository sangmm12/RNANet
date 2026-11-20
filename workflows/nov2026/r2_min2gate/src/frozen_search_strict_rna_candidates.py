#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Strict-RNA sequence-divergent structural candidate search

目标：
- 在 Step4 输出的 Master Table（几何 + RNA-FM）里，
  找到“融合特征距离很近（结构/语义近）但序列很远（低同源）”的链对。
- 输出 High_Confidence_Candidates.csv 供 Step6 做结构对齐验证（RMSD/TM-score/RNA-align）。

关键修复（相对 v2）：
1) 不再使用 python-Levenshtein 的 ratio() 作为“序列相似度”默认值。
   ratio() 对等长序列的下界是 0.5（完全不同也 >=0.5），会导致 --seq_sim_max=0.35 永远触发不了。
   现在默认使用 “归一化编辑距离相似度”：
       SeqSim = 1 - (edit_distance / max_len)   ∈ [0,1]
   对等长完全不同的序列，SeqSim≈0（符合直觉）。
2) 若环境缺少 C 实现的 Levenshtein / rapidfuzz，则自动退化为 k-mer Jaccard（默认 k=3），避免 O(L^2) DP 过慢。
3) 增加 seq_sim 的分布诊断（抽样近邻对），便于调参。
"""

import argparse
import warnings
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import os

def _pick_feature_columns(df: pd.DataFrame):
    """从 master table 中识别特征列，并返回 (geom_cols, fm_cols, drop_cols)"""
    geom_cols = []
    fm_cols = []
    drop_cols = []

    for c in df.columns:
        if c in ("Chain_ID", "ResidueIndex"):
            drop_cols.append(c)
            continue
        if c.startswith("UMAP_"):
            drop_cols.append(c)
            continue
        # step4 可能加的辅助标签
        if c in ("geo_density", "conv_score", "is_core", "is_outlier", "is_convergence"):
            drop_cols.append(c)
            continue

        if c.startswith("fm_"):
            fm_cols.append(c)
        else:
            # 剩下的都按“几何特征”处理（包括 phi/psi/ContactCount/各 k 指标等）
            geom_cols.append(c)

    return geom_cols, fm_cols, drop_cols


def _impute_and_scale(X: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """NaN/Inf -> 中位数填补，然后标准化 (z-score)"""
    X = X.astype(np.float64, copy=True)
    bad = ~np.isfinite(X)
    if bad.any():
        med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        # 仍可能存在全 NaN 列
        med = np.where(np.isfinite(med), med, 0.0)
        X[bad] = np.take(med, np.where(bad)[1])

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    Xs = (X - mean) / std
    return Xs, (mean, std)


def _load_sequences(path: str) -> Dict[str, str]:
    """读取 RNA_Sequences.csv（至少包含 Chain_ID/RNAID 与 Sequence 两列）"""
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    # 容错：Chain_ID / RNAID / id
    id_col = None
    for cand in ("chain_id", "rnaid", "id"):
        if cand in cols:
            id_col = df.columns[cols.index(cand)]
            break
    if id_col is None:
        raise ValueError(f"{path} 未找到 Chain_ID/RNAID 列。现有列: {list(df.columns)[:10]} ...")
    seq_col = None
    for cand in ("sequence", "seq"):
        if cand in cols:
            seq_col = df.columns[cols.index(cand)]
            break
    if seq_col is None:
        raise ValueError(f"{path} 未找到 Sequence 列。现有列: {list(df.columns)[:10]} ...")

    seqs = {}
    for _, r in df[[id_col, seq_col]].dropna().iterrows():
        seqs[str(r[id_col]).strip()] = str(r[seq_col]).strip().upper().replace("T", "U")
    return seqs


def _parse_pdb(chain_id: str) -> str:
    """从 Chain_ID 中提取 PDB（用于跨 PDB 约束）"""
    # 常见格式：8AUV_B / 9FXO_S5 / 7OIF_4 ...
    if "_" in chain_id:
        return chain_id.split("_", 1)[0]
    return chain_id


def _kmer_jaccard(a: str, b: str, k: int = 3) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if len(a) < k or len(b) < k:
        return (1.0 if a == b else 0.0)
    sa = {a[i:i+k] for i in range(len(a) - k + 1)}
    sb = {b[i:i+k] for i in range(len(b) - k + 1)}
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _edit_distance_similarity(a: str, b: str) -> float:
    """
    默认序列相似度：SeqSim = 1 - dist / max_len
    dist 优先用 C 实现（Levenshtein / rapidfuzz），缺失时退化为 k-mer Jaccard。
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    # 1) python-Levenshtein（最快）
    try:
        import Levenshtein  # type: ignore
        dist = Levenshtein.distance(a, b)
        m = max(len(a), len(b))
        return 1.0 - (dist / m if m else 0.0)
    except Exception:
        pass

    # 2) rapidfuzz（通常也很快）
    try:
        from rapidfuzz.distance import Levenshtein as RFLev  # type: ignore
        dist = RFLev.distance(a, b)
        m = max(len(a), len(b))
        return 1.0 - (dist / m if m else 0.0)
    except Exception:
        pass

    # 3) 退化：k-mer Jaccard（避免 O(L^2) DP 太慢）
    return _kmer_jaccard(a, b, k=3)


def _seq_similarity(a: str, b: str, metric: str, kmer_k: int) -> float:
    if metric == "lev_norm":
        return _edit_distance_similarity(a, b)
    if metric == "kmer":
        return _kmer_jaccard(a, b, k=kmer_k)
    if metric == "identity":
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        m = min(len(a), len(b))
        same = sum((a[i] == b[i]) for i in range(m))
        return same / max(len(a), len(b))
    raise ValueError(f"unknown metric={metric}")


def main():
    ap = argparse.ArgumentParser(description="Step5: 搜索非同源结构类似物候选对（geo-close, seq-far）")
    ap.add_argument("--master", default="RNA_Chain_Master_Table.csv",
                    help="Step4 输出的 Master table（融合了几何 + RNA-FM）。")
    ap.add_argument("--seq", default="RNA_Sequences.csv",
                    help="包含 Chain_ID(or RNAID) 与 Sequence 的表。")
    ap.add_argument("--out", default="High_Confidence_Candidates.csv",
                    help="输出候选对 CSV（供 Step6）。")

    ap.add_argument("--kNN", type=int, default=100,
                    help="每条链只在融合特征空间里找 kNN 个近邻，避免 O(N^2)。")
    ap.add_argument("--dist_percentile", type=float, default=10.0,
                    help="融合距离阈值：取 kNN 距离分布的前 p%% 作为 '足够近'。")
    ap.add_argument("--seq_sim_max", type=float, default=0.35,
                    help="序列相似度上限（越小表示越不相似）。默认配合 lev_norm 可用。")
    ap.add_argument("--geo_percentile", type=float, default=None,
                    help="可选：Geo_Dist 双阈值过滤：只保留 Geo_Dist 落在最小前 p% 的候选对（例如 10 表示最小 10%）。不设置则跳过。")
    ap.add_argument("--fm_percentile", type=float, default=None,
                    help="可选：FM_Dist 双阈值过滤：只保留 FM_Dist 落在最小前 p% 的候选对。不设置则跳过。")
    ap.add_argument("--dedup_pdbpair", choices=["yes","no"], default="yes",
                    help="是否对 (PDB_A,PDB_B) 去重（无向），保留最高 Score 的链对。")
    ap.add_argument("--emit_topk", default="200,500",
                    help="额外写出 TopK 文件（供 Step6 直接使用），例如 '200,500'。空字符串则不输出。")
    ap.add_argument("--topN", type=int, default=5000,
                    help="最终输出的候选对上限（按 Score 排序截断）。")

    ap.add_argument("--w_geo", type=float, default=1.0, help="融合距离中几何部分权重")
    ap.add_argument("--w_fm", type=float, default=1.0, help="融合距离中 RNA-FM 部分权重")
    ap.add_argument("--allow_same_pdb", action="store_true",
                    help="允许同一 PDB 内的链对（默认只保留跨 PDB）")

    # 新增：序列相似度定义
    ap.add_argument("--seq_metric", choices=["lev_norm", "kmer", "identity"], default="lev_norm",
                    help="序列相似度度量：lev_norm=1-dist/maxlen（推荐）；kmer=Jaccard；identity=位置一致率。")
    ap.add_argument("--kmer_k", type=int, default=3, help="--seq_metric=kmer 时使用的 k")
    ap.add_argument("--diag_samples", type=int, default=20000,
                    help="诊断：抽样多少个(近邻)对来打印 SeqSim/Dist 分布；0 关闭。")

    args = ap.parse_args()

    print("--- 启动 Strict-RNA candidate search (geometry/RNA-FM thresholds, PDB-pair deduplication) ---")

    df = pd.read_csv(args.master)
    if "Chain_ID" not in df.columns:
        raise ValueError("Master table 缺少 Chain_ID 列。")
    df = df.drop_duplicates(subset=["Chain_ID"]).reset_index(drop=True)

    geom_cols, fm_cols, drop_cols = _pick_feature_columns(df)
    if drop_cols:
        warnings.warn(f"检测到 {len(drop_cols)} 列未被纳入距离计算（将忽略），例如：{drop_cols[:8]}")
    if not geom_cols:
        raise ValueError("未识别到几何特征列（geom_cols 为空）。请检查 Master Table 列名。")
    if not fm_cols:
        warnings.warn("未识别到 fm_* 列（fm_cols 为空）。这将退化为纯几何近邻搜索。")

    print(f"[INFO] chains={len(df)} | geom_dim={len(geom_cols)} | fm_dim={len(fm_cols)}")
    chain_ids = df["Chain_ID"].tolist()

    seqs = _load_sequences(args.seq)
    missing_seq = sum((cid not in seqs) for cid in chain_ids)
    if missing_seq:
        warnings.warn(f"{missing_seq}/{len(chain_ids)} 条链在 {args.seq} 中找不到序列，将在计算 SeqSim 时跳过。")

    Xg_s, _ = _impute_and_scale(df[geom_cols].to_numpy())
    if fm_cols:
        Xf_s, _ = _impute_and_scale(df[fm_cols].to_numpy())
    else:
        Xf_s = np.zeros((len(df), 0), dtype=np.float64)

    X_fused = np.concatenate([np.sqrt(args.w_geo) * Xg_s, np.sqrt(args.w_fm) * Xf_s], axis=1)

    k = max(5, min(int(args.kNN), len(df) - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X_fused)
    dists, idxs = nn.kneighbors(X_fused, return_distance=True)

    all_knn = dists[:, 1:].reshape(-1)
    all_knn = all_knn[np.isfinite(all_knn)]
    dist_th = float(np.percentile(all_knn, float(args.dist_percentile)))
    print(f"[INFO] fused distance threshold = {dist_th:.4f} (kNN distances top {args.dist_percentile:.1f}%)")

    require_cross_pdb = not bool(args.allow_same_pdb)

    # ---- 诊断：抽样打印 SeqSim 分布，帮助你知道阈值应不应该是 0.35 / 0.2 / 0.1 ----
    if int(args.diag_samples) > 0:
        rng = np.random.default_rng(42)
        sims = []
        fused = []
        tries = int(args.diag_samples)
        for _ in range(tries):
            i = int(rng.integers(0, len(df)))
            r = int(rng.integers(1, idxs.shape[1]))
            j = int(idxs[i, r])
            if j == i:
                continue
            cid_i = chain_ids[i]; cid_j = chain_ids[j]
            if require_cross_pdb and (_parse_pdb(cid_i) == _parse_pdb(cid_j)):
                continue
            si = seqs.get(cid_i, ""); sj = seqs.get(cid_j, "")
            if not si or not sj:
                continue
            sims.append(_seq_similarity(si, sj, args.seq_metric, args.kmer_k))
            fused.append(float(dists[i, r]))
        if sims:
            sims = np.asarray(sims)
            fused = np.asarray(fused)
            q = np.percentile(sims, [1, 5, 10, 25, 50, 75, 90, 95, 99])
            print(f"[DIAG] SeqSim(metric={args.seq_metric}) percentiles: "
                  f"p1={q[0]:.3f} p5={q[1]:.3f} p10={q[2]:.3f} p25={q[3]:.3f} "
                  f"p50={q[4]:.3f} p75={q[5]:.3f} p90={q[6]:.3f} p95={q[7]:.3f} p99={q[8]:.3f}")
        else:
            print("[DIAG] 无法抽样到可计算 SeqSim 的近邻对（可能是 seq 缺失或跨 PDB 过滤过强）。")

    pairs = {}
    skipped_no_seq = 0

    for i in tqdm(range(len(df)), desc="Scanning kNN pairs"):
        cid_i = chain_ids[i]
        pdb_i = _parse_pdb(cid_i)
        seq_i = seqs.get(cid_i, "")

        for r in range(1, idxs.shape[1]):
            j = int(idxs[i, r])
            if j <= i:
                continue

            fused_dist = float(dists[i, r])
            if (not np.isfinite(fused_dist)) or (fused_dist > dist_th):
                continue

            cid_j = chain_ids[j]
            pdb_j = _parse_pdb(cid_j)
            if require_cross_pdb and (pdb_i == pdb_j):
                continue

            seq_j = seqs.get(cid_j, "")
            if not seq_i or not seq_j:
                skipped_no_seq += 1
                continue

            seq_sim = _seq_similarity(seq_i, seq_j, args.seq_metric, args.kmer_k)
            if seq_sim > float(args.seq_sim_max):
                continue

            geo_dist = float(np.linalg.norm(Xg_s[i] - Xg_s[j]))
            fm_dist = float(np.linalg.norm(Xf_s[i] - Xf_s[j])) if Xf_s.shape[1] else float("nan")
            seq_diff = 1.0 - float(seq_sim)

            # 分数：越“近”越好（小 fused_dist），越“远序列”越好（大 seq_diff）
            score = seq_diff / (fused_dist + 1e-9)

            key = (cid_i, cid_j)
            row = {
                "RNA_A": cid_i,
                "RNA_B": cid_j,
                "PDB_A": pdb_i,
                "PDB_B": pdb_j,
                "Feature_Dist": fused_dist,
                "Geo_Dist": geo_dist,
                "FM_Dist": fm_dist,
                "SeqSim": float(seq_sim),
                "SeqDiff": float(seq_diff),
                "Score": float(score),
                "is_convergence": 1,
                "seq_metric": args.seq_metric,
            }
            prev = pairs.get(key)
            if (prev is None) or (row["Score"] > prev["Score"]):
                pairs[key] = row

    cand_df = pd.DataFrame(list(pairs.values()))
    if cand_df.empty:
        print("[WARN] 没有找到候选对。")
        print("  你可以按下面顺序放宽：")
        print("  (1) --dist_percentile 20 或 30（允许更远一点的结构/语义近邻）")
        print("  (2) --kNN 200 或 300（扩大每个点的候选邻居池）")
        print("  (3) --seq_sim_max 0.40 或 0.50（允许更“像”的序列也先进入候选，再靠 Step6 结构验证筛掉）")
        print("  (4) 或改用 --seq_metric kmer --kmer_k 3/4（更偏移位不敏感）")
        return

    # 可选：Geo/Fm 双阈值过滤（在候选对集合内取最小的前 p%）
    if args.geo_percentile is not None:
        gp = float(args.geo_percentile)
        if 0 < gp < 100:
            geo_valid = cand_df["Geo_Dist"].replace([np.inf, -np.inf], np.nan).dropna()
            if len(geo_valid) > 0:
                geo_th = float(np.percentile(geo_valid.values, gp))
                before = len(cand_df)
                cand_df = cand_df[cand_df["Geo_Dist"] <= geo_th].copy()
                print(f"[FILTER] Geo_Dist <= p{gp:g} ({geo_th:.4f}) -> {len(cand_df)}/{before} pairs")
    if args.fm_percentile is not None:
        fp = float(args.fm_percentile)
        if 0 < fp < 100:
            fm_valid = cand_df["FM_Dist"].replace([np.inf, -np.inf], np.nan).dropna()
            if len(fm_valid) > 0:
                fm_th = float(np.percentile(fm_valid.values, fp))
                before = len(cand_df)
                cand_df = cand_df[cand_df["FM_Dist"] <= fm_th].copy()
                print(f"[FILTER] FM_Dist <= p{fp:g} ({fm_th:.4f}) -> {len(cand_df)}/{before} pairs")

    cand_df = cand_df.sort_values(["Score", "SeqDiff"], ascending=[False, False])
    # 可选：对 (PDB_A, PDB_B) 去重（无向），保留最优链对
    if args.dedup_pdbpair == "yes" and len(cand_df) > 0:
        before = len(cand_df)
        pdbpair = cand_df.apply(lambda r: "|".join(sorted([str(r["PDB_A"]), str(r["PDB_B"])])), axis=1)
        cand_df = cand_df.assign(_pdbpair=pdbpair)
        cand_df = cand_df.drop_duplicates("_pdbpair", keep="first").drop(columns=["_pdbpair"])
        cand_df = cand_df.reset_index(drop=True)
        print(f"[DEDUP] pdbpair unique -> {len(cand_df)}/{before} pairs")

    if int(args.topN) > 0:
        cand_df = cand_df.head(int(args.topN)).reset_index(drop=True)

    cand_df.to_csv(args.out, index=False)

    # 额外写出 TopK 子集（供 Step6 直接喂入）
    emit = str(args.emit_topk).strip()
    if emit:
        base = args.out
        root, ext = os.path.splitext(base)
        ks = []
        for tok in emit.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                ks.append(int(tok))
            except Exception:
                pass
        for k in sorted(set([x for x in ks if x > 0])):
            out_k = f"{root}.top{k}{ext if ext else '.csv'}"
            cand_df.head(k).to_csv(out_k, index=False)
            print(f"[OK] 写出 Top{k}: {out_k}")

    print(f"[OK] 输出候选对: {args.out}")
    print(f"[INFO] candidates={len(cand_df)} | skipped_no_seq={skipped_no_seq}")
    print("[INFO] Top 5:\n" + cand_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
