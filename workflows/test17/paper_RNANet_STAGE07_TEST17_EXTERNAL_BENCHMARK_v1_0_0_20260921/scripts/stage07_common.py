from __future__ import annotations

import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_fasta(path: Path) -> Tuple[str, str]:
    header = ''
    seq: List[str] = []
    for raw in path.read_text(encoding='utf-8', errors='replace').splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith('>'):
            if not header:
                header = line[1:].strip()
            continue
        # Individual benchmark FASTAs should contain sequence only after the header.
        # Ignore comma-separated binary label lines if one is unexpectedly present.
        if re.fullmatch(r'[01](?:,[01])+', line):
            continue
        seq.append(re.sub(r'\s+', '', line).upper().replace('T', 'U'))
    s = ''.join(seq)
    if not s:
        raise ValueError(f'empty FASTA sequence: {path}')
    return header, s


def load_label_pickle(path: Path) -> Any:
    with path.open('rb') as f:
        return pickle.load(f)


def normalize_label_collection(obj: Any, fasta_files: List[Path]) -> Dict[str, np.ndarray]:
    """Map benchmark labels to sorted FASTA stems.

    The public benchmark validator treats the pickle as an ordered 17/60-element
    collection. Some future revisions may use a dict; support both while refusing
    ambiguous length mismatches.
    """
    stems = [p.stem for p in fasta_files]
    if isinstance(obj, dict):
        out: Dict[str, np.ndarray] = {}
        lowered = {str(k).lower(): k for k in obj}
        for stem in stems:
            candidates = [stem, stem.lower(), stem.upper(), stem.replace('_', ''), stem.replace('_', '').lower()]
            key = next((lowered[c.lower()] for c in candidates if c.lower() in lowered), None)
            if key is None:
                raise KeyError(f'no label key for FASTA {stem}; available keys example={list(obj)[:8]}')
            out[stem] = np.asarray(obj[key], dtype=int).reshape(-1)
        return out
    if isinstance(obj, (list, tuple, np.ndarray)):
        if len(obj) != len(stems):
            raise ValueError(f'label object length={len(obj)} != fasta count={len(stems)}')
        return {stem: np.asarray(lab, dtype=int).reshape(-1) for stem, lab in zip(stems, obj)}
    raise TypeError(f'unsupported label pickle type: {type(obj)!r}')


def parse_chain_from_stem(stem: str) -> Tuple[str, str]:
    clean = stem.strip()
    if len(clean) < 5:
        raise ValueError(f'cannot parse PDB+chain from stem={stem!r}')
    pdb_id = clean[:4].upper()
    rest = clean[4:]
    rest = re.sub(r'^[_.:-]+', '', rest)
    # Legacy benchmark chains are one-character PDB chain IDs. Preserve a longer
    # remainder if the benchmark ever uses one, but do not silently drop it.
    chain = rest
    if not chain:
        raise ValueError(f'missing chain suffix in FASTA stem={stem!r}')
    return pdb_id, chain


def find_structure(binding_site: Path, pdb_id: str, chain: str) -> Path:
    data = binding_site / 'data'
    preferred_dirs = [
        data / 'common' / 'pdbFiles',
        data / 'RNAsite' / 'data_v2' / 'normalized_pdb',
    ]
    names = [
        f'{pdb_id}.pdb', f'{pdb_id.lower()}.pdb', f'{pdb_id.upper()}.pdb',
        f'{pdb_id}{chain}.pdb', f'{pdb_id.lower()}{chain}.pdb',
        f'{pdb_id}.cif', f'{pdb_id.lower()}.cif', f'{pdb_id}.mmcif',
    ]
    for d in preferred_dirs:
        if not d.exists():
            continue
        for n in names:
            p = d / n
            if p.exists():
                return p.resolve()
    # Last-resort discovery is intentionally exact on the four-character PDB ID.
    matches: List[Path] = []
    for ext in ('*.pdb', '*.cif', '*.mmcif'):
        for p in data.rglob(ext):
            s = p.stem.upper().replace('_', '')
            if s == pdb_id or s.startswith(pdb_id + chain.upper()):
                matches.append(p.resolve())
    uniq = sorted(set(matches))
    if len(uniq) == 1:
        return uniq[0]
    if not uniq:
        raise FileNotFoundError(f'no structure for {pdb_id}{chain} under {data}')
    raise RuntimeError(f'ambiguous structures for {pdb_id}{chain}: {uniq[:12]}')


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def finite_float(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False
