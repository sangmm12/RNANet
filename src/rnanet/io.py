from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from Bio.PDB import PDBParser, MMCIFParser

@dataclass
class RNAResidue:
    index: int
    chain: str
    resseq: int
    icode: str
    resname: str
    atoms: Dict[str, np.ndarray]

def parser_for(path: str):
    p = str(path).lower()
    return MMCIFParser(QUIET=True) if p.endswith((".cif", ".mmcif")) else PDBParser(QUIET=True)

def _pick_atom(res, name: str):
    cand = [a for a in res.get_atoms() if a.get_id().strip() == name]
    if not cand:
        return None
    if len(cand) == 1:
        return cand[0]
    return max(cand, key=lambda a: float(a.get_occupancy() or 0.0))

def load_rna(path: str, chain: str = "ALL") -> List[RNAResidue]:
    st = parser_for(path).get_structure(Path(path).name, path)
    model = next(st.get_models())
    out: List[RNAResidue] = []
    idx = 1
    for ch in model:
        if chain != "ALL" and ch.id != chain:
            continue
        for res in ch:
            # The public geometry contract needs a nucleotide representative coordinate.
            # Hetero/non-RNA records without P and C4' are excluded.
            if _pick_atom(res, "P") is None and _pick_atom(res, "C4'") is None:
                continue
            atoms: Dict[str, np.ndarray] = {}
            for a in res.get_atoms():
                name = a.get_id().strip()
                atoms[name] = np.asarray(a.coord, dtype=float)
            out.append(RNAResidue(idx, ch.id, int(res.id[1]), str(res.id[2]).strip(), res.resname.strip(), atoms))
            idx += 1
    return out

def atom(r: RNAResidue, name: str) -> Optional[np.ndarray]:
    return r.atoms.get(name)
