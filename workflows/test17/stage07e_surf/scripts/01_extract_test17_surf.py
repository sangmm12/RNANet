#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SURF_SHA = "7cab38601388a6a66d6250c4d00e0b0c027abb2d5f1c12634f5785ed9a0a47aa"
IO_SHA = "0f4554f67d2770598c89491ef1fa28197c57ec5db087700299d042a4096429dc"
BENCHMARK_COMMIT = "f1cea5cfc04541dda22d0c027f87cb0e20067013"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def json_dump(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def read_fasta(path: Path) -> str:
    lines = [x.strip() for x in path.read_text().splitlines() if x.strip() and not x.startswith(">")]
    return "".join(lines).upper().replace("T", "U")


def pdb_row_key(line: str):
    if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
        return None
    try:
        resname = line[17:20].strip().upper()
        chain = line[21:22].strip()
        resseq = int(line[22:26].strip())
        icode = line[26:27].strip()
    except Exception:
        return None
    return (chain, resseq, icode, resname)


def write_filtered_pdb(src: Path, dst: Path, selected_keys: set[tuple[str,int,str,str]]) -> int:
    lines = []
    atom_lines = 0
    for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
        k = pdb_row_key(line)
        if k is not None and k in selected_keys:
            lines.append(line)
            atom_lines += 1
    lines.append("END")
    dst.write_text("\n".join(lines) + "\n")
    if atom_lines == 0:
        raise RuntimeError(f"no selected coordinate atoms written for {src}")
    return atom_lines


def worker(task: dict) -> dict:
    tool_root = Path(task["tool_root"])
    stage07_pkg = Path(task["stage07_pkg"])
    sys.path.insert(0, str(tool_root.parents[2] / "src"))
    sys.path.insert(0, str(stage07_pkg / "scripts"))
    from rnanet.io import load_rna, parser_for
    from rnanet.surf import extract, FIELDS
    from stage07b_row_contract_r5 import resolve_chain_contract

    structure = Path(task["structure"])
    chain_id = task["chain_id"]
    seq = task["seq"]
    chosen, metas, audit = resolve_chain_contract(structure, chain_id, seq, parser_for)
    if not audit.get("gate_pass"):
        raise RuntimeError(f"R5 row contract failed for {task['chain_key']}: {audit}")
    raw = load_rna(str(structure), chain_id)
    if len(raw) != len(metas):
        raise RuntimeError(f"candidate-row drift for {task['chain_key']}: raw={len(raw)} meta={len(metas)}")
    selected_metas = [metas[i] for i in chosen]
    selected_keys = {(m.chain, int(m.resseq), str(m.icode), str(m.resname).upper()) for m in selected_metas}

    with tempfile.TemporaryDirectory(prefix=f"stage07e_{task['chain_key']}_") as td:
        filtered = Path(td) / f"{task['chain_key']}_RNA_ONLY.pdb"
        atom_lines = write_filtered_pdb(structure, filtered, selected_keys)
        df = extract(str(filtered), chain_id, task["msms_bin"])

    if len(df) != len(seq):
        raise RuntimeError(f"SURF rows != FASTA for {task['chain_key']}: surf={len(df)} fasta={len(seq)}")
    expected_meta = [(m.resseq, m.icode, m.resname.upper()) for m in selected_metas]
    got_meta = [(int(r.ResSeq), str(r.ICode), str(r.ResName).upper()) for r in df.itertuples(index=False)]
    if expected_meta != got_meta:
        raise RuntimeError(f"SURF residue order drift for {task['chain_key']}")

    feature_cols = ["n_verts"]
    for field in FIELDS:
        for stat in ("mean", "std", "min", "max"):
            feature_cols.append(f"surf_{field}_{stat}")
    if len(feature_cols) != 57:
        raise AssertionError(len(feature_cols))
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing SURF columns for {task['chain_key']}: {missing}")

    rows = []
    for i, r in df.iterrows():
        z = {
            "set": task["set_name"],
            "chain_key": task["chain_key"],
            "ResidueIndex": i + 1,
            "ResSeq": int(r["ResSeq"]),
            "ICode": str(r["ICode"]) if pd.notna(r["ICode"]) else "",
            "ResName": str(r["ResName"]),
        }
        for c in feature_cols:
            z[c] = float(r[c]) if pd.notna(r[c]) else np.nan
        rows.append(z)
    vals = df[feature_cols].to_numpy(float)
    finite_fraction = float(np.isfinite(vals).mean())
    zero_vert_rows = int((df["n_verts"].to_numpy(float) == 0).sum())
    return {
        "rows": rows,
        "audit": {
            "set": task["set_name"], "chain_key": task["chain_key"], "status": "PASS",
            "n_rows": len(df), "atom_lines": atom_lines, "finite_fraction": finite_fraction,
            "zero_vertex_rows": zero_vert_rows, "row_contract_digest": audit["row_contract_digest"],
        },
        "feature_cols": feature_cols,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--tool-root", required=True)
    ap.add_argument("--msms-bin", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()

    project = Path(a.project_root).resolve()
    tool_root = Path(a.tool_root).resolve()
    msms = Path(a.msms_bin).resolve()
    if not msms.is_file():
        raise FileNotFoundError(f"MSMS binary not found: {msms}")
    if sha256_file(tool_root.parents[2] / "src/rnanet/surf.py") != SURF_SHA:
        raise RuntimeError("bundled SURF implementation hash mismatch")
    if sha256_file(tool_root.parents[2] / "src/rnanet/io.py") != IO_SHA:
        raise RuntimeError("bundled IO implementation hash mismatch")

    stage07_pkg = project / "stage07_benchmark"
    out = project / "07_test17_external_benchmark" / "v1_0_0"
    result_dir = out / "posthoc_surf_v1_0_0"
    result_dir.mkdir(parents=True, exist_ok=True)

    r5_gate = json.loads((out / "STAGE07B_R5_ROW_CONTRACT_AUDIT.json").read_text())
    cyc_gate = json.loads((out / "STAGE07B_CYC_EXTRACT_GATE.json").read_text())
    if not r5_gate.get("gate_pass") or int(r5_gate.get("chains_passed", 0)) != 77:
        raise RuntimeError("R5 all-77 row-contract gate is not PASS")
    if not cyc_gate.get("gate_pass"):
        raise RuntimeError("Stage07B CY-C extraction gate is not PASS")
    cyc_matrix = out / "rnanet_test17_cyc_matrix.csv.gz"
    if sha256_file(cyc_matrix) != cyc_gate.get("matrix_sha256"):
        raise RuntimeError("frozen CY-C matrix hash drift")

    vendor = out / "vendor" / "rna-small-molecule-review"
    commit = os.popen(f"git -C {vendor} rev-parse HEAD").read().strip()
    if commit != BENCHMARK_COMMIT:
        raise RuntimeError(f"benchmark commit drift: {commit}")

    manifest = pd.read_csv(out / "DATASET_MANIFEST.csv")
    tasks = []
    for _, rec in manifest.iterrows():
        tasks.append({
            "tool_root": str(tool_root), "stage07_pkg": str(stage07_pkg), "msms_bin": str(msms),
            "set_name": str(rec["set"]), "chain_key": str(rec["chain_key"]),
            "chain_id": str(rec["chain_id"]), "structure": str(rec["structure"]),
            "seq": read_fasta(Path(rec["fasta"])),
        })

    all_rows, audits, feature_cols = [], [], None
    failures = []
    with cf.ProcessPoolExecutor(max_workers=max(1, int(a.jobs))) as ex:
        futs = {ex.submit(worker, t): t for t in tasks}
        done = 0
        for fut in cf.as_completed(futs):
            t = futs[fut]; done += 1
            try:
                r = fut.result()
                all_rows.extend(r["rows"]); audits.append(r["audit"])
                if feature_cols is None: feature_cols = r["feature_cols"]
                elif feature_cols != r["feature_cols"]: raise RuntimeError("SURF feature schema drift")
                print(f"[SURF] {done:02d}/77 {t['set_name']} {t['chain_key']} rows={r['audit']['n_rows']} finite={r['audit']['finite_fraction']:.4f} zeroV={r['audit']['zero_vertex_rows']}")
            except Exception as e:
                failures.append({"set": t["set_name"], "chain_key": t["chain_key"], "error": repr(e)})
                print(f"[SURF-FAIL] {done:02d}/77 {t['set_name']} {t['chain_key']} {e}")

    pd.DataFrame(audits + failures).to_csv(result_dir / "SURF_CHAIN_EXTRACTION_AUDIT.csv", index=False)
    if failures:
        gate = {
            "gate": "FAIL_STAGE07E_POSTHOC_SURF_EXTRACT", "gate_pass": False,
            "post_hoc_exploratory": True, "eligible_for_primary_external_claim": False,
            "chains_expected": 77, "chains_succeeded": len(audits), "chains_failed": len(failures),
            "failed_chain_keys": [x["chain_key"] for x in failures],
        }
        json_dump(result_dir / "STAGE07E_SURF_EXTRACT_GATE.json", gate)
        print(json.dumps(gate, indent=2)); raise SystemExit(2)

    surf_df = pd.DataFrame(all_rows).sort_values(["set", "chain_key", "ResidueIndex"], kind="mergesort").reset_index(drop=True)
    base = pd.read_csv(cyc_matrix)
    key = ["set", "chain_key", "ResidueIndex"]
    merged = base.merge(surf_df[key + feature_cols], on=key, how="left", validate="one_to_one")
    if len(merged) != 3637 or int((merged["set"] == "Test17").sum()) != 583:
        raise RuntimeError(f"merged row-count drift: total={len(merged)} test={int((merged['set']=='Test17').sum())}")
    if merged[feature_cols].isna().all(axis=1).any():
        bad = merged.loc[merged[feature_cols].isna().all(axis=1), key]
        raise RuntimeError(f"rows with all SURF57 missing after merge: {bad.head().to_dict('records')}")
    matrix = result_dir / "rnanet_test17_cyc_surf_matrix.csv.gz"
    merged.to_csv(matrix, index=False, compression="gzip")
    (result_dir / "SURF_FEATURES_57.txt").write_text("\n".join(feature_cols) + "\n")
    model_features = [x.strip() for x in (out / "CY-C_MODEL_FEATURES_46.txt").read_text().splitlines() if x.strip()] + feature_cols
    (result_dir / "CY-C_SURF_MODEL_FEATURES_103.txt").write_text("\n".join(model_features) + "\n")
    arr = merged[feature_cols].to_numpy(float)
    gate = {
        "gate": "PASS_STAGE07E_POSTHOC_SURF_EXTRACT", "gate_pass": True,
        "post_hoc_exploratory": True, "test17_seen_before_arm_definition": True,
        "eligible_for_primary_external_claim": False, "test17_tuning_within_arm": False,
        "benchmark_commit": commit, "rows_total": int(len(merged)), "rows_train60": int((merged['set']=='Train60').sum()),
        "rows_test17": int((merged['set']=='Test17').sum()), "surf_dim": len(feature_cols),
        "cyc_plus_onehot_dim": 46, "model_dim": len(model_features),
        "surf_finite_fraction": float(np.isfinite(arr).mean()), "zero_vertex_rows": int((merged['n_verts']==0).sum()),
        "msms_probe_A": 1.5, "msms_density": 3.0, "convex_hull_fallback": False,
        "surface_source_sha256": SURF_SHA, "base_cyc_matrix_sha256": sha256_file(cyc_matrix),
        "cyc_surf_matrix_sha256": sha256_file(matrix),
    }
    json_dump(result_dir / "STAGE07E_SURF_EXTRACT_GATE.json", gate)
    print(json.dumps(gate, indent=2))

if __name__ == "__main__":
    main()
