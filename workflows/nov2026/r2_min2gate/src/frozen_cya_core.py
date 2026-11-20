#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNA CY-A A5 feature extractor, v1.0.0.

Frozen default four-angle RNA conformational contract:
  eta     = C4'(i-1) - P(i) - C4'(i) - P(i+1)
  theta   = P(i) - C4'(i) - P(i+1) - C4'(i+1)
  chi     = O4'(i) - C1'(i) - N9(i) - C4(i) for purines,
            O4'(i) - C1'(i) - N1(i) - C2(i) for pyrimidines
  puckerP = Altona-Sundaralingam ribose pseudorotation phase computed
            from nu0..nu4 ring torsions.

The four circular coordinates are embedded as (S1)^4 -> R8 using cos/sin pairs.
The A5 core at each neighborhood scale k is:
  lag_mean, lag_p90, sl_abs_mean, phase_dispersion, pca_effdim

Compatibility goals:
  * accepts legacy RNA --input/--output CLI and protein-A5 --pdb-dir/--out-dir aliases
  * writes <structure>_features.csv so existing ligand/interface merge code can read it
  * retains legacy RNA ContactCount/ContactOrder semantics by default
  * computes sequential RNA angles within each chain independently (no cross-chain torsions)

This script intentionally does not label DNA as RNA. Dataset-level strict polymer typing
must be enforced upstream from mmCIF entity_poly metadata for retrieval analyses.
"""

import argparse
import csv
import glob
import math
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

A5_CORE = ["lag_mean", "lag_p90", "sl_abs_mean", "phase_dispersion", "pca_effdim"]
PURINES = {"A", "G", "RA", "RG", "ADE", "GUA"}
PYRIMIDINES = {"C", "U", "RC", "RU", "CYT", "URA", "URI"}


def _pick_atom(res, name: str):
    hits = [a for a in res.get_atoms() if a.get_id().strip() == name]
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]
    best = None
    best_occ = -1.0
    for atom in hits:
        occ = atom.get_occupancy()
        occ = 0.0 if occ is None else float(occ)
        if occ > best_occ:
            best_occ = occ
            best = atom
    return best if best is not None else hits[0]


def _coord(atom) -> np.ndarray:
    if atom is None:
        return np.full(3, np.nan, dtype=float)
    return np.asarray(atom.get_coord(), dtype=float)


def _finite4(*pts) -> bool:
    return all(np.all(np.isfinite(np.asarray(p, dtype=float))) for p in pts)


def dihedral(p0, p1, p2, p3) -> float:
    if not _finite4(p0, p1, p2, p3):
        return np.nan
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p3 = np.asarray(p3, dtype=float)
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    n1 = np.linalg.norm(b1)
    if n1 < 1e-12:
        return np.nan
    b1 = b1 / n1
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    nv = np.linalg.norm(v)
    nw = np.linalg.norm(w)
    if nv < 1e-12 or nw < 1e-12:
        return np.nan
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.arctan2(y, x))


def circular_mean(vals) -> float:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    z = np.exp(1j * arr).mean()
    if abs(z) < 1e-15:
        return 0.0
    return float(np.angle(z))


def circular_mean_window(vals: List[float], i: int, window: int) -> float:
    if window <= 1:
        return float(vals[i]) if np.isfinite(vals[i]) else np.nan
    half = window // 2
    lo = max(0, i - half)
    hi = min(len(vals), i + half + 1)
    sub = [x for x in vals[lo:hi] if np.isfinite(x)]
    return circular_mean(sub) if sub else np.nan


def smooth_and_impute(vals: List[float], window: int) -> List[float]:
    out = [circular_mean_window(vals, i, window) for i in range(len(vals))]
    mu = circular_mean(out)
    return [float(x) if np.isfinite(x) else float(mu) for x in out]


def parse_structure(path: str):
    ext = os.path.splitext(path)[1].lower()
    parser = MMCIFParser(QUIET=True) if ext in (".cif", ".mmcif") else PDBParser(QUIET=True)
    return parser.get_structure(os.path.basename(path), path)


def residue_atom_dict(res) -> Dict[str, object]:
    names = [
        "P", "O5'", "C5'", "C4'", "C3'", "O3'",
        "O4'", "C1'", "C2'", "N9", "C4", "N1", "C2"
    ]
    return {name: _pick_atom(res, name) for name in names}


def extract_nucleic_chains(structure, chain_mode: str = "ALL") -> List[List[Dict]]:
    model = next(structure.get_models())
    mode = str(chain_mode)
    mode_up = mode.upper()
    raw = []
    for ch in model:
        if mode_up not in ("ALL", "LONGEST") and ch.id != mode:
            continue
        rr = []
        for res in ch:
            c4_atom = _pick_atom(res, "C4'")
            if c4_atom is None:
                continue
            atoms = residue_atom_dict(res)
            bvals = []
            for aname in ["P", "O5'", "C5'", "C4'", "C3'", "O3'"]:
                atom = atoms.get(aname)
                if atom is not None:
                    try:
                        bvals.append(float(atom.get_bfactor()))
                    except Exception:
                        pass
            rr.append({
                "res_obj": res,
                "chain": ch.id,
                "resseq": int(res.id[1]),
                "icode": str(res.id[2]).strip(),
                "resname": str(res.get_resname()).strip().upper(),
                "atoms": atoms,
                "c4": np.asarray(c4_atom.get_coord(), dtype=float),
                "B_avg": float(np.mean(bvals)) if bvals else np.nan,
            })
        if rr:
            raw.append(rr)
    if mode_up == "LONGEST" and raw:
        raw = [max(raw, key=len)]
    return raw


def linked(prev: Dict, cur: Dict, cutoff: float) -> bool:
    a = _coord(prev["atoms"].get("O3'"))
    b = _coord(cur["atoms"].get("P"))
    if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
        return float(np.linalg.norm(a - b)) <= float(cutoff)
    # Conservative fallback for incomplete structures.
    return (prev["chain"] == cur["chain"]) and (cur["resseq"] - prev["resseq"] == 1)


def glycosidic_chi(r: Dict) -> float:
    a = r["atoms"]
    c_o4 = _coord(a.get("O4'"))
    c_c1 = _coord(a.get("C1'"))
    name = r["resname"].upper()
    if name in PURINES:
        return dihedral(c_o4, c_c1, _coord(a.get("N9")), _coord(a.get("C4")))
    if name in PYRIMIDINES:
        return dihedral(c_o4, c_c1, _coord(a.get("N1")), _coord(a.get("C2")))
    v = dihedral(c_o4, c_c1, _coord(a.get("N9")), _coord(a.get("C4")))
    if np.isfinite(v):
        return v
    return dihedral(c_o4, c_c1, _coord(a.get("N1")), _coord(a.get("C2")))


def pucker_phase(r: Dict) -> float:
    a = r["atoms"]
    c4 = _coord(a.get("C4'"))
    o4 = _coord(a.get("O4'"))
    c1 = _coord(a.get("C1'"))
    c2 = _coord(a.get("C2'"))
    c3 = _coord(a.get("C3'"))
    nu0 = dihedral(c4, o4, c1, c2)
    nu1 = dihedral(o4, c1, c2, c3)
    nu2 = dihedral(c1, c2, c3, c4)
    nu3 = dihedral(c2, c3, c4, o4)
    nu4 = dihedral(c3, c4, o4, c1)
    nus = np.asarray([nu0, nu1, nu2, nu3, nu4], dtype=float)
    if not np.all(np.isfinite(nus)):
        return np.nan
    num = nu4 + nu1 - nu3 - nu0
    den = 2.0 * nu2 * (math.sin(math.radians(36.0)) + math.sin(math.radians(72.0)))
    p = math.atan2(num, den)
    if p < 0.0:
        p += 2.0 * math.pi
    return float(p)


def build_four_angles(chains: List[List[Dict]], window: int, link_cutoff: float):
    residues = []
    rows = []
    for chain_res in chains:
        n = len(chain_res)
        eta = []
        theta = []
        chi = []
        puck = []
        for i, r in enumerate(chain_res):
            prev = chain_res[i - 1] if i > 0 else None
            nxt = chain_res[i + 1] if i + 1 < n else None
            prev_ok = prev is not None and linked(prev, r, link_cutoff)
            next_ok = nxt is not None and linked(r, nxt, link_cutoff)
            if prev_ok and next_ok:
                eta.append(dihedral(
                    _coord(prev["atoms"].get("C4'")),
                    _coord(r["atoms"].get("P")),
                    _coord(r["atoms"].get("C4'")),
                    _coord(nxt["atoms"].get("P")),
                ))
            else:
                eta.append(np.nan)
            if next_ok:
                theta.append(dihedral(
                    _coord(r["atoms"].get("P")),
                    _coord(r["atoms"].get("C4'")),
                    _coord(nxt["atoms"].get("P")),
                    _coord(nxt["atoms"].get("C4'")),
                ))
            else:
                theta.append(np.nan)
            chi.append(glycosidic_chi(r))
            puck.append(pucker_phase(r))

        eta_s = smooth_and_impute(eta, window)
        theta_s = smooth_and_impute(theta, window)
        chi_s = smooth_and_impute(chi, window)
        puck_s = smooth_and_impute(puck, window)
        for i, r in enumerate(chain_res):
            residues.append(r)
            rows.append((eta_s[i], theta_s[i], chi_s[i], puck_s[i]))

    X = np.asarray(rows, dtype=float)
    return residues, X


def torus_embed(X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return np.empty((0, 8), dtype=float)
    parts = []
    for j in range(X.shape[1]):
        parts.append(np.cos(X[:, j]))
        parts.append(np.sin(X[:, j]))
    return np.stack(parts, axis=1).astype(float)


def compute_contacts_legacy(residues: List[Dict], cutoff: float) -> Tuple[np.ndarray, np.ndarray]:
    n = len(residues)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    pts = np.stack([r["c4"] for r in residues], axis=0)
    nbrs = NearestNeighbors(radius=float(cutoff), algorithm="ball_tree").fit(pts)
    ind_list = nbrs.radius_neighbors(pts, return_distance=False)
    cc = np.zeros(n, dtype=float)
    co = np.zeros(n, dtype=float)
    for i, inds in enumerate(ind_list):
        jj = [int(j) for j in inds if int(j) != i]
        cc[i] = float(len(jj))
        co[i] = float(np.mean([abs(j - i) for j in jj])) if jj else 0.0
    return cc, co


def standard_J(dim: int) -> np.ndarray:
    """Standard complex structure for interleaved (cos a1, sin a1, ...) coordinates.

    This is deliberately identical in coordinate convention to the revised
    protein CY-A A5 implementation.
    """
    if dim % 2 != 0:
        raise ValueError(f"Expected even torus embedding dimension, got {dim}")
    J = np.zeros((dim, dim), dtype=float)
    for k in range(dim // 2):
        a, b = 2 * k, 2 * k + 1
        J[a, b] = -1.0
        J[b, a] = 1.0
    return J


def local_pca(Y: np.ndarray, idxs: np.ndarray, pca_var: float, min_dim: int, max_dim: int):
    """Local PCA exactly matching the revised protein CY-A A5 convention."""
    if len(idxs) < 2:
        return None, 0, 0.0
    Xn = np.asarray(Y[idxs], dtype=float) - np.asarray(Y[idxs[0]], dtype=float)
    if not np.any(np.var(Xn, axis=0) > 1e-12):
        return None, 0, 0.0
    if max_dim is None:
        max_dim = Y.shape[1] // 2
    n_comp = min(int(max_dim), Xn.shape[0], Xn.shape[1])
    if n_comp < 1:
        return None, 0, 0.0
    pca = PCA(n_components=n_comp).fit(Xn)
    evr = np.asarray(pca.explained_variance_ratio_, dtype=float)
    cum = np.cumsum(evr)
    d_eig = int(np.searchsorted(cum, pca_var) + 1)
    d_eig = max(1, min(d_eig, pca.n_components_))
    d_use = min(max(d_eig, int(min_dim)), pca.n_components_)
    lam = np.asarray(pca.explained_variance_, dtype=float)
    lam = lam[np.isfinite(lam) & (lam > 0.0)]
    if lam.size == 0 or float(np.sum(lam * lam)) <= 1e-18:
        pca_effdim = 0.0
    else:
        pca_effdim = float((lam.sum() ** 2) / np.sum(lam * lam))
    return np.asarray(pca.components_[:d_use], dtype=float), int(d_eig), pca_effdim

def holo_volume_general(vectors: List[np.ndarray], m_complex: int) -> complex:
    """Holomorphic volume in interleaved real coordinates.

    Real vector coordinates are (x1,y1,x2,y2,...), so z_j=x_j+i y_j.
    """
    if len(vectors) != m_complex:
        return 0.0 + 0.0j
    rows = []
    for v in vectors:
        v = np.asarray(v, dtype=float)
        if len(v) != 2 * m_complex:
            return 0.0 + 0.0j
        z = np.asarray([v[2*j] + 1j*v[2*j+1] for j in range(m_complex)], dtype=np.complex128)
        rows.append(z)
    try:
        return complex(np.linalg.det(np.vstack(rows)))
    except Exception:
        return 0.0 + 0.0j


def orientation_invariant_phase2(vals: np.ndarray, mag_eps: float) -> complex:
    vals = np.asarray(vals, dtype=np.complex128)
    vals = vals[np.isfinite(vals.real) & np.isfinite(vals.imag) & (np.abs(vals) > float(mag_eps))]
    if vals.size == 0:
        return np.nan + 1j * np.nan
    unit = vals / np.abs(vals)
    q = unit ** 2
    z = q.mean()
    if not np.isfinite(z.real) or not np.isfinite(z.imag):
        return np.nan + 1j * np.nan
    return complex(z / abs(z)) if abs(z) > 1e-15 else complex(z)


def precompute_neighbors(Y: np.ndarray, maxK: int):
    n = len(Y)
    k_eff = max(1, min(int(maxK), n))
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="auto").fit(Y)
    return nbrs.kneighbors(Y)


def fill_nans_inplace(d: Dict[str, float]):
    for k in list(d):
        try:
            x = float(d[k])
        except Exception:
            x = 0.0
        if not np.isfinite(x):
            x = 0.0
        d[k] = x


def cy_features_for_k(Y, inds_full, center, k, pca_var, m_complex,
                      sl_auto_k=False, max_k_auto=None, phase_mag_eps=1e-12):
    n_total = len(Y)
    if max_k_auto is None:
        max_k_auto = n_total
    cur_k = min(max(int(k), int(m_complex) + 1), n_total)
    U = None
    d_eig = 0
    pca_effdim = 0.0
    while True:
        idxs = inds_full[center][:cur_k]
        U, d_eig, pca_effdim = local_pca(
            Y, idxs, pca_var=pca_var, min_dim=m_complex, max_dim=m_complex
        )
        if (not sl_auto_k) or d_eig >= m_complex or cur_k >= max_k_auto or cur_k >= n_total:
            break
        cur_k = min(cur_k + 2, n_total)

    if U is None:
        feats = dict(lag_mean=0.0, lag_p90=0.0, sl_abs_mean=0.0,
                     pca_effdim=0.0, k_used=int(cur_k))
        return feats, d_eig, np.nan + 1j * np.nan

    J = standard_J(Y.shape[1])
    d_use = U.shape[0]
    lag_vals = []
    for a in range(d_use):
        for b in range(a + 1, d_use):
            lag_vals.append(abs(float((J @ U[a]).dot(U[b]))))
    lag_vals = np.asarray(lag_vals, dtype=float)
    lag_mean = float(lag_vals.mean()) if lag_vals.size else 0.0
    lag_p90 = float(np.percentile(lag_vals, 90)) if lag_vals.size else 0.0

    sl_vals = []
    if d_use >= m_complex:
        for comb in combinations(range(d_use), m_complex):
            sl_vals.append(holo_volume_general([U[j] for j in comb], m_complex))
    sl_vals = np.asarray(sl_vals, dtype=np.complex128)
    sl_abs_mean = float(np.mean(np.abs(sl_vals))) if sl_vals.size else 0.0
    phase2 = orientation_invariant_phase2(sl_vals, mag_eps=phase_mag_eps)

    feats = dict(
        lag_mean=lag_mean,
        lag_p90=lag_p90,
        sl_abs_mean=sl_abs_mean,
        pca_effdim=float(pca_effdim),
        k_used=int(cur_k),
    )
    fill_nans_inplace(feats)
    return feats, d_eig, phase2


def phase_dispersion_for_k(phase2_arr, inds_full, k_used_arr):
    n = len(phase2_arr)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        kk = max(1, min(int(k_used_arr[i]), len(inds_full[i])))
        idx = inds_full[i][:kk]
        z = np.asarray([phase2_arr[j] for j in idx], dtype=np.complex128)
        good = np.isfinite(z.real) & np.isfinite(z.imag)
        z = z[good]
        if z.size == 0:
            out[i] = 0.0
        else:
            out[i] = float(1.0 - np.clip(abs(z.mean()), 0.0, 1.0))
    return out


def angle_global_mean(vals) -> float:
    return circular_mean(vals)


def process_one(path, out_dir, ks, pca_var, window, sl_auto_k, max_k_auto,
                contacts_cutoff, chain_mode, phase_mag_eps, link_cutoff):
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    out_csv = os.path.join(out_dir, f"{stem}_features.csv")
    dim_stats_local = {}
    try:
        structure = parse_structure(path)
        chains = extract_nucleic_chains(structure, chain_mode=chain_mode)
        if not chains:
            os.makedirs(out_dir, exist_ok=True)
            with open(out_csv, "w", newline="") as f:
                csv.writer(f).writerow(["ResidueIndex", "Chain", "ResSeq", "ResName"])
            return path, True, "empty", dim_stats_local

        residues, Xang = build_four_angles(chains, window=window, link_cutoff=link_cutoff)
        if len(residues) == 0:
            return path, True, "empty", dim_stats_local
        Y = torus_embed(Xang)
        m_complex = 4
        N = len(Y)
        ks = sorted(set(int(k) for k in ks))
        maxK = min(max(max(ks), m_complex + 1), N)
        _, inds_full = precompute_neighbors(Y, maxK)
        contact_count, contact_order = compute_contacts_legacy(residues, cutoff=contacts_cutoff)
        bvals = np.asarray([r["B_avg"] for r in residues], dtype=float)

        feats_by_k = {k: {name: np.zeros(N, dtype=float) for name in A5_CORE} for k in ks}
        k_used_by_k = {k: np.zeros(N, dtype=int) for k in ks}
        phase2_by_k = {k: np.full(N, np.nan + 1j * np.nan, dtype=np.complex128) for k in ks}

        for k in ks:
            dim_stats_local.setdefault(k, {})
            for i in range(N):
                feats, d_eig, phase2 = cy_features_for_k(
                    Y, inds_full, i, k, pca_var=pca_var, m_complex=m_complex,
                    sl_auto_k=sl_auto_k, max_k_auto=max_k_auto,
                    phase_mag_eps=phase_mag_eps,
                )
                dim_stats_local[k][d_eig] = dim_stats_local[k].get(d_eig, 0) + 1
                for name in ("lag_mean", "lag_p90", "sl_abs_mean", "pca_effdim"):
                    feats_by_k[k][name][i] = feats[name]
                k_used_by_k[k][i] = feats["k_used"]
                phase2_by_k[k][i] = phase2

        for k in ks:
            feats_by_k[k]["phase_dispersion"] = phase_dispersion_for_k(
                phase2_by_k[k], inds_full, k_used_by_k[k]
            )

        header = [
            "ResidueIndex", "Chain", "ResSeq", "ResName",
            "eta", "theta", "chi", "puckerP",
            "ContactCount", "ContactOrder", "B_avg",
        ]
        for k in ks:
            header.extend([f"{name}_k{k}" for name in A5_CORE])

        os.makedirs(out_dir, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            global_row = [
                -1, "GLOBAL", -1, "GLOBAL",
                angle_global_mean(Xang[:, 0]),
                angle_global_mean(Xang[:, 1]),
                angle_global_mean(Xang[:, 2]),
                angle_global_mean(Xang[:, 3]),
                float(np.nanmean(contact_count)) if contact_count.size else 0.0,
                float(np.nanmean(contact_order)) if contact_order.size else 0.0,
                float(np.nanmean(bvals)) if np.isfinite(bvals).any() else np.nan,
            ]
            for k in ks:
                for name in A5_CORE:
                    arr = np.asarray(feats_by_k[k][name], dtype=float)
                    arr = arr[np.isfinite(arr)]
                    global_row.append(float(arr.mean()) if arr.size else 0.0)
            w.writerow(global_row)

            for i, r in enumerate(residues):
                row = [
                    i + 1, r["chain"], r["resseq"], r["resname"],
                    float(Xang[i, 0]), float(Xang[i, 1]),
                    float(Xang[i, 2]), float(Xang[i, 3]),
                    float(contact_count[i]), float(contact_order[i]),
                    float(bvals[i]) if np.isfinite(bvals[i]) else np.nan,
                ]
                for k in ks:
                    for name in A5_CORE:
                        row.append(float(feats_by_k[k][name][i]))
                w.writerow(row)
        return path, True, f"OK n_res={N}", dim_stats_local
    except Exception as e:
        return path, False, str(e) + "\n" + traceback.format_exc(), dim_stats_local


def main():
    ap = argparse.ArgumentParser(description="RNA CY-A A5: eta,theta,chi,puckerP on (S1)^4")
    ap.add_argument("--input", dest="input_dir", default=None)
    ap.add_argument("--output", dest="output_dir", default=None)
    ap.add_argument("--pdb-dir", dest="pdb_dir", default=None)
    ap.add_argument("--out-dir", dest="out_dir", default=None)
    ap.add_argument("--ks", default="4,12,20,28,36,44")
    ap.add_argument("--pca-var", type=float, default=0.90)
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--sl-auto-k", action="store_true")
    ap.add_argument("--contacts-cutoff", type=float, default=10.0)
    ap.add_argument("--phase-mag-eps", type=float, default=1e-12)
    ap.add_argument("--chain", default="ALL")
    ap.add_argument("--link-cutoff", type=float, default=2.5)
    ap.add_argument("--enhanced", default="B_avg")
    ap.add_argument("--tan-dim", type=int, default=None, help="Accepted for legacy pipeline compatibility; ignored by A5.")
    ap.add_argument("--angle-mode", type=int, default=4, help="Compatibility flag; RNA A5 v1.0.0 is fixed at four angles.")
    ap.add_argument("--dim-stats-out", default="dim_stats_cya_a5.csv")
    args = ap.parse_args()

    in_dir = args.input_dir or args.pdb_dir
    out_dir = args.output_dir or args.out_dir
    if not in_dir or not out_dir:
        ap.error("provide --input/--output or --pdb-dir/--out-dir")
    if int(args.angle_mode) != 4:
        ap.error("RNA CY-A A5 v1.0.0 is frozen to angle-mode 4")
    if args.window < 1 or args.window % 2 == 0:
        ap.error("--window must be a positive odd integer")

    ks = [int(x) for x in str(args.ks).split(",") if x.strip()]
    if not ks:
        ap.error("no valid --ks")
    max_k_auto = max(ks)

    paths = []
    for pat in ("*.pdb", "*.cif", "*.mmcif"):
        paths.extend(glob.glob(os.path.join(in_dir, pat)))
    paths = sorted(set(paths))
    if not paths:
        print(f"[WARN] no structure files found in: {in_dir}")
        return 0
    os.makedirs(out_dir, exist_ok=True)

    worker = partial(
        process_one,
        out_dir=out_dir,
        ks=ks,
        pca_var=args.pca_var,
        window=args.window,
        sl_auto_k=bool(args.sl_auto_k),
        max_k_auto=max_k_auto,
        contacts_cutoff=args.contacts_cutoff,
        chain_mode=args.chain,
        phase_mag_eps=args.phase_mag_eps,
        link_cutoff=args.link_cutoff,
    )

    global_dim = {}
    errors = []
    if args.jobs <= 1:
        results = [worker(p) for p in paths]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(worker, p) for p in paths]
            for fu in as_completed(futs):
                results.append(fu.result())

    for path, ok, msg, dmap in results:
        print(("[OK] " if ok else "[ERR] ") + os.path.basename(path) + ": " + str(msg).splitlines()[0])
        if not ok:
            errors.append((path, msg))
        for k, dd in dmap.items():
            global_dim.setdefault(k, {})
            for d, cnt in dd.items():
                global_dim[k][d] = global_dim[k].get(d, 0) + cnt

    dim_path = os.path.join(out_dir, args.dim_stats_out)
    with open(dim_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "dim_eig", "count", "fraction"])
        for k in sorted(global_dim):
            total = sum(global_dim[k].values())
            for d in sorted(global_dim[k]):
                cnt = global_dim[k][d]
                w.writerow([k, d, cnt, cnt / total if total else 0.0])

    if errors:
        err_path = os.path.join(out_dir, "cya_a5_errors.log")
        with open(err_path, "w", encoding="utf-8") as f:
            for p, msg in errors:
                f.write(f"=== {p} ===\n{msg}\n\n")
        print(f"[FAIL] errors={len(errors)} see {err_path}")
        return 2

    print(f"[DONE] files={len(paths)} out={out_dir} dim_stats={dim_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
