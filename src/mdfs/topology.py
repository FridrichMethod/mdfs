from __future__ import annotations

from dataclasses import dataclass

import mdtraj as md
import numpy as np


@dataclass
class TopologyArrays:
    positions: np.ndarray  # (N, 3) float64, in nm (MDTraj native)
    bonds: np.ndarray  # (Nb, 2) int32
    angles: np.ndarray  # (Na, 3) int32
    dihedrals: np.ndarray  # (Nd, 4) int32
    onefour: np.ndarray  # (N14, 2) int32
    types: np.ndarray  # (N,) int32 (element-to-type map)
    charges: np.ndarray  # (N,) float64 (placeholder; set later from params if available)
    atom_names: list[str]
    res_names: list[str]
    chain_ids: list[str]
    box_lengths: np.ndarray | None  # (3,) nm if unit cell present else None


# Minimal element -> type map (extend as needed)
_ELEM_ORDER = ["H", "C", "N", "O", "S", "P", "F", "CL", "BR", "I"]
_ELEM_TO_TYPE: dict[str, int] = {e: i for i, e in enumerate(_ELEM_ORDER)}


def _elem_to_type(sym: str) -> int:
    s = (sym or "").upper()
    if s in _ELEM_TO_TYPE:
        return _ELEM_TO_TYPE[s]
    # normalize common halogens if needed
    if s == "CL":
        return _ELEM_TO_TYPE["CL"]
    if s == "BR":
        return _ELEM_TO_TYPE["BR"]
    return len(_ELEM_TO_TYPE)  # unknown bucket


# -------------------------
# Public API
# -------------------------
def load_topology_mdtraj(pdb_path: str) -> TopologyArrays:
    """
    Load a PDB with MDTraj and produce arrays needed by your energy/integrators.

    Returns coordinates in nm, integer index arrays for bonds/angles/dihedrals,
    1–4 pairs, element-based 'types', and unit-cell lengths if present.
    """
    # Load single-frame structure (coords + Topology); MDTraj uses nm/deg/ps.  :contentReference[oaicite:4]{index=4}
    traj = md.load(pdb_path)

    # Coordinates (first frame); shape (n_frames, N, 3) with units nm.
    if traj.n_frames == 0:
        raise ValueError("No frames found in PDB.")
    positions = traj.xyz[0].astype(np.float64)

    top = (
        traj.topology
    )  # md.Topology: chains/residues/atoms/bonds  :contentReference[oaicite:5]{index=5}

    N = top.n_atoms
    atom_list = list(top.atoms)

    # Bonds directly from MDTraj Topology (zero-based indices).  :contentReference[oaicite:6]{index=6}
    bonds = np.array([(b[0].index, b[1].index) for b in top.bonds], dtype=np.int32)

    # Build angles/dihedrals from the bond graph
    adj = _adjacency_from_bonds(bonds, N)
    angles = _all_angles(adj)
    dihedrals = _all_dihedrals(adj)
    # 1–4 pairs from dihedrals
    onefour = (
        np.unique(np.sort(dihedrals[:, [0, 3]], axis=1), axis=0)
        if dihedrals.size
        else np.zeros((0, 2), np.int32)
    )

    # Atom metadata -> types (element-based) and placeholders for charges
    # MDTraj Atom.element may be None; fall back to atom name's first letter.  :contentReference[oaicite:7]{index=7}
    elems: list[str] = []
    atom_names: list[str] = []
    res_names: list[str] = []
    chain_ids: list[str] = []
    for a in atom_list:
        atom_names.append(a.name)
        res_names.append(a.residue.name)
        chain_ids.append(a.residue.chain.chain_id)
        if a.element is not None and getattr(a.element, "symbol", None):
            elems.append(a.element.symbol.upper())
        else:
            # crude fallback: first alpha char of name
            nm = (a.name or "").strip()
            elems.append(nm[0].upper() if nm else "C")

    types = np.array([_elem_to_type(e) for e in elems], dtype=np.int32)
    charges = np.zeros((N,), dtype=np.float64)  # supply from params later if you have them

    # Unit cell (orthorhombic assumption for convenience)
    # Trajectory exposes either unitcell_lengths (nm) or unitcell_vectors.  :contentReference[oaicite:8]{index=8}
    box_lengths: np.ndarray | None
    if traj.unitcell_lengths is not None:
        # (n_frames, 3) in nm; take frame 0
        box_lengths = np.asarray(traj.unitcell_lengths[0], dtype=np.float64)
        # MDTraj may drop bogus CRYST1 per heuristic; then this is None.  :contentReference[oaicite:9]{index=9}
    elif traj.unitcell_vectors is not None:
        # lengths from vectors' norms
        vecs = np.asarray(traj.unitcell_vectors[0], dtype=np.float64)  # (3,3)
        box_lengths = np.array([np.linalg.norm(vecs[i]) for i in range(3)], dtype=np.float64)
    else:
        box_lengths = None

    return TopologyArrays(
        positions=positions,
        bonds=bonds,
        angles=angles,
        dihedrals=dihedrals,
        onefour=onefour,
        types=types,
        charges=charges,
        atom_names=atom_names,
        res_names=res_names,
        chain_ids=chain_ids,
        box_lengths=box_lengths,
    )


# -------------------------
# Graph helpers (angles/dihedrals)
# -------------------------
def _adjacency_from_bonds(bonds: np.ndarray, n_atoms: int) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(n_atoms)]
    for i, j in bonds:
        i = int(i)
        j = int(j)
        if i == j:
            continue
        adj[i].add(j)
        adj[j].add(i)
    return adj


def _all_angles(adj: list[set[int]]) -> np.ndarray:
    angles = set()
    for j, nbrs in enumerate(adj):
        nbrs_list = sorted(nbrs)
        for a in range(len(nbrs_list)):
            for b in range(a + 1, len(nbrs_list)):
                i = nbrs_list[a]
                k = nbrs_list[b]
                # canonical (i, j, k) with i<k
                if i < k:
                    angles.add((i, j, k))
                else:
                    angles.add((k, j, i))
    if not angles:
        return np.zeros((0, 3), dtype=np.int32)
    return np.array(sorted(angles), dtype=np.int32)


def _all_dihedrals(adj: list[set[int]]) -> np.ndarray:
    dihs = set()
    for j in range(len(adj)):
        for k in adj[j]:
            if j == k:
                continue
            # neighbors excluding each other
            left = [i for i in adj[j] if i != k]
            right = [l for l in adj[k] if l != j]
            for i in left:
                for l in right:
                    if i == l:  # avoid three-atom loops
                        continue
                    # canonicalize to avoid duplicates: prefer (i<l) then (j<=k)
                    if (l, k, j, i) in dihs:
                        continue
                    if (i, j, k, l) in dihs:
                        continue
                    if (i < l) or (i == l and j <= k):
                        dihs.add((i, j, k, l))
                    else:
                        dihs.add((l, k, j, i))
    if not dihs:
        return np.zeros((0, 4), dtype=np.int32)
    return np.array(sorted(dihs), dtype=np.int32)
