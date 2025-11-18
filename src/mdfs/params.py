from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from openmm.app import ForceField  # OpenMM parser/loader

from mdfs.energy import LJMixParams as JaxLJMixParams

# ===============================
# Public dataclasses / bridges
# ===============================


@dataclass
class LJMixParams:
    eps_type: np.ndarray  # (Nt,) kJ/mol
    sig_type: np.ndarray  # (Nt,) nm
    type_index: dict[str, int]  # atom *type* -> compact LJ type id


def lj_to_jax(lj_np: LJMixParams) -> JaxLJMixParams:
    return JaxLJMixParams(
        eps_type=jnp.asarray(lj_np.eps_type),
        sig_type=jnp.asarray(lj_np.sig_type),
    )


def gather_pair_values(pairs: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Gather (Np,) values from an (N,N) matrix (NumPy)."""
    return mat[pairs[:, 0], pairs[:, 1]]


@dataclass
class BondTable:
    # key (ti, tj) with ti<=tj -> (k_r, r0)
    kr_r0: dict[tuple[int, int], tuple[float, float]]


@dataclass
class AngleTable:
    # key (ti, tj, tk) -> (k_theta, theta0)
    kth_th0: dict[tuple[int, int, int], tuple[float, float]]


@dataclass
class TorsionTable:
    # key (ti, tj, tk, tl) -> list[(n, k_n, delta)]
    fourier: dict[tuple[int, int, int, int], list[tuple[int, float, float]]]


@dataclass
class ResidueTemplate:
    # atoms: list of (atomName, atomType, charge)
    atoms: list[tuple[str, str, float]]


@dataclass
class FFParams:
    lj: LJMixParams
    bonds: BondTable
    angles: AngleTable
    torsions: TorsionTable
    residue_templates: dict[str, ResidueTemplate]
    # Global 1–4 scaling (vdW and Coulomb)
    scale14_vdw: float = 0.5
    scale14_elec: float = 1.0 / 1.2


# ===============================
# Internal helpers (OpenMM-only)
# ===============================


def _require(cond: bool, msg: str):
    if not cond:
        raise RuntimeError(msg)


def _collect_generators(ff: ForceField):
    """
    Return useful generators by type name. We access stable internal attributes that
    OpenMM's ForceField builds from FFXML:
      - NonbondedGenerator
      - HarmonicBondGenerator
      - HarmonicAngleGenerator
      - PeriodicTorsionGenerator
    """
    gens = getattr(ff, "_forces", None)
    _require(gens is not None, "OpenMM ForceField internal generators not found (_forces missing).")

    nb_gen = None
    bond_gen = None
    angle_gen = None
    tors_gen = None

    for g in gens:
        gname = g.__class__.__name__
        if "Nonbonded" in gname:
            nb_gen = g
        elif "HarmonicBond" in gname:
            bond_gen = g
        elif "HarmonicAngle" in gname:
            angle_gen = g
        elif "PeriodicTorsion" in gname:
            tors_gen = g

    _require(nb_gen is not None, "Nonbonded generator not present in ForceField.")
    return nb_gen, bond_gen, angle_gen, tors_gen


def _collect_atomtypes(ff: ForceField):
    """
    Build:
      - type_to_class: dict[type_name -> class_name or None]
      - class_to_types: dict[class_name -> list[type_name]]
    From ForceField internals (_atomTypes).
    """
    atomtypes = getattr(ff, "_atomTypes", None)
    _require(atomtypes is not None, "ForceField _atomTypes missing; cannot map type/class.")

    type_to_class: dict[str, str | None] = {}
    class_to_types: dict[str, list[str]] = {}

    # _atomTypes is a dict: name -> AtomType(name, class, element, mass, ...)
    for tname, tdef in atomtypes.items():
        cls = getattr(tdef, "className", None) or getattr(
            tdef, "clazz", None
        )  # className in OpenMM
        type_to_class[tname] = cls
        if cls:
            class_to_types.setdefault(cls, []).append(tname)
    return type_to_class, class_to_types


def _collect_residue_templates(ff: ForceField) -> dict[str, ResidueTemplate]:
    """
    Use ForceField's parsed residue templates (ff._templates) to build typing/charge table.
    """
    templates_map = getattr(ff, "_templates", None)
    _require(
        templates_map is not None, "ForceField _templates missing; cannot read residue templates."
    )

    out: dict[str, ResidueTemplate] = {}
    for rname, tmpl in templates_map.items():
        # tmpl.atoms: list of TemplateAtom objects with .name, .type, .charge
        atoms = []
        for a in tmpl.atoms:
            an = getattr(a, "name", None)
            at = getattr(a, "type", None)
            q = float(getattr(a, "charge", 0.0))
            if an and at:
                atoms.append((an, at, q))
        if atoms:
            out[rname.upper()] = ResidueTemplate(atoms=atoms)
    return out


def _dedupe_lj_type_table(nb_gen, class_to_types: dict[str, list[str]]) -> LJMixParams:
    """
    Build a compact table of LJ (sigma, epsilon) indexed by *type*.
    Nonbonded parameters may be provided per type or per class;
    expand classes to their member types.
    """
    # Nonbonded parameters container
    # OpenMM's NonbondedGenerator typically stores:
    #   _paramsByType: {type_name: (charge, sigma, epsilon)}
    #   _paramsByClass: {class_name: (charge, sigma, epsilon)}
    p_by_type = getattr(nb_gen, "_paramsByType", None) or getattr(nb_gen, "_parameters", {})
    p_by_class = getattr(nb_gen, "_paramsByClass", None) or {}

    sig_types: list[float] = []
    eps_types: list[float] = []
    type_index: dict[str, int] = {}
    seen: dict[tuple[float, float], int] = {}

    # Collect all type keys we care about (union of explicitly-typed and all class-expanded)
    all_types: dict[str, tuple[float, float]] = {}

    # Direct per-type definitions
    for tname, triple in p_by_type.items():
        # triple = (q, sigma, epsilon) in OpenMM units already normalized for XML
        sigma = float(triple[1])
        eps = float(triple[2])
        all_types[tname] = (sigma, eps)

    # Class-based: expand to member types if they don't already have explicit entries
    for cname, triple in p_by_class.items():
        sigma = float(triple[1])
        eps = float(triple[2])
        for tname in class_to_types.get(cname, []):
            all_types.setdefault(tname, (sigma, eps))

    # Deduplicate (σ, ε) pairs and assign compact ids
    for tname, (s, e) in all_types.items():
        key = (s, e)
        if key not in seen:
            seen[key] = len(seen)
            sig_types.append(s)
            eps_types.append(e)
        type_index[tname] = seen[key]

    return LJMixParams(
        eps_type=np.asarray(eps_types, dtype=float),
        sig_type=np.asarray(sig_types, dtype=float),
        type_index=type_index,
    )


def _bond_table_from_generator(bond_gen, lj_types: LJMixParams) -> BondTable:
    """
    Build BondTable keyed by (type_i, type_j) indices (order-invariant).
    """
    if bond_gen is None:
        return BondTable(kr_r0={})

    # Expect a parameter store:
    #   bond_gen._parameters[(type1, type2)] = (length, k)
    params = getattr(bond_gen, "_parameters", None)
    _require(params is not None, "HarmonicBond generator has no _parameters.")

    table: dict[tuple[int, int], tuple[float, float]] = {}
    for (t1, t2), (length, k) in params.items():
        if t1 not in lj_types.type_index or t2 not in lj_types.type_index:
            continue
        i = lj_types.type_index[t1]
        j = lj_types.type_index[t2]
        key = (i, j) if i <= j else (j, i)
        table[key] = (float(k), float(length))  # store (k_r, r0)
    return BondTable(kr_r0=table)


def _angle_table_from_generator(angle_gen, lj_types: LJMixParams) -> AngleTable:
    """
    Build AngleTable keyed by (ti, tj, tk) type indices; store reversed outer for lookup symmetry.
    """
    if angle_gen is None:
        return AngleTable(kth_th0={})
    params = getattr(angle_gen, "_parameters", None)
    _require(params is not None, "HarmonicAngle generator has no _parameters.")

    table: dict[tuple[int, int, int], tuple[float, float]] = {}
    for (t1, t2, t3), (theta0, k) in params.items():
        if (
            t1 not in lj_types.type_index
            or t2 not in lj_types.type_index
            or t3 not in lj_types.type_index
        ):
            continue
        i = lj_types.type_index[t1]
        j = lj_types.type_index[t2]
        k_i = lj_types.type_index[t3]
        table[i, j, k_i] = (float(k), float(theta0))
        table[k_i, j, i] = (float(k), float(theta0))
    return AngleTable(kth_th0=table)


def _torsion_table_from_generator(tors_gen, lj_types: LJMixParams) -> TorsionTable:
    """
    Build TorsionTable keyed by (ti, tj, tk, tl) indices with list of Fourier terms.
    Supports multi-term torsions (periodicity1/k1/phase1...) as pre-expanded by OpenMM.
    """
    if tors_gen is None:
        return TorsionTable(fourier={})
    # PeriodicTorsion generator parameters shape:
    #   tors_gen._parameters[(t1, t2, t3, t4)] = [(period, phase, k), ...]
    params = getattr(tors_gen, "_parameters", None)
    _require(params is not None, "PeriodicTorsion generator has no _parameters.")

    table: dict[tuple[int, int, int, int], list[tuple[int, float, float]]] = {}
    for (t1, t2, t3, t4), term_list in params.items():
        if (
            t1 not in lj_types.type_index
            or t2 not in lj_types.type_index
            or t3 not in lj_types.type_index
            or t4 not in lj_types.type_index
        ):
            continue
        i = lj_types.type_index[t1]
        j = lj_types.type_index[t2]
        k = lj_types.type_index[t3]
        l = lj_types.type_index[t4]
        key = (i, j, k, l)
        rkey = (l, k, j, i)
        dst = table.setdefault(key, [])
        rdst = table.setdefault(rkey, [])
        # Each entry is already per-term (periodicity, phase, k) in OpenMM units
        for per, phase, kval in term_list:
            tup = (int(per), float(kval), float(phase))
            dst.append(tup)
            rdst.append(tup)
    return TorsionTable(fourier=table)


# ===============================
# Public loader (OpenMM-only)
# ===============================


def load_ffxml(
    ffxml_path: str,
    scale14_vdw: float = 0.5,
    scale14_elec: float = 1.0 / 1.2,
) -> FFParams:
    """
    Load a force field FFXML via OpenMM and expose parameters in JAX-friendly tables.

    Returns FFParams with:
      - LJ per-type table (σ, ε) with type_index map
      - Bond/Angle/Proper-Torsion tables keyed by *type indices*
      - Residue templates for typing and charges
      - 1–4 scaling from the Nonbonded generator if present; else provided defaults
    """
    ff = ForceField(ffxml_path)

    # Atom type/class maps + residue templates (typing and charges)
    type_to_class, class_to_types = _collect_atomtypes(ff)
    residue_templates = _collect_residue_templates(ff)

    # Generators
    nb_gen, bond_gen, angle_gen, tors_gen = _collect_generators(ff)

    # Nonbonded LJ per-type (expands class LJ to types when needed)
    lj = _dedupe_lj_type_table(nb_gen, class_to_types)

    # 1–4 scaling directly from nb_gen if available
    c14 = getattr(nb_gen, "_coulomb14scale", None)
    l14 = getattr(nb_gen, "_lj14scale", None)
    scale14_elec_final = float(c14) if c14 is not None else scale14_elec
    scale14_vdw_final = float(l14) if l14 is not None else scale14_vdw

    # Bonded tables from their generators
    bonds = _bond_table_from_generator(bond_gen, lj)
    angles = _angle_table_from_generator(angle_gen, lj)
    torsions = _torsion_table_from_generator(tors_gen, lj)

    return FFParams(
        lj=lj,
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        residue_templates=residue_templates,
        scale14_vdw=scale14_vdw_final,
        scale14_elec=scale14_elec_final,
    )


# ===============================
# Assignment & per-instance API
# ===============================


def assign_types_and_charges_from_templates(
    res_names: Iterable[str],
    atom_names: Iterable[str],
    ff: FFParams,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given per-atom (resname, atomname) from MDTraj, assign (type_id, charge)
    using ForceField residue templates parsed by OpenMM. Unknowns -> (-1, 0.0).
    """
    res_names = list(res_names)
    atom_names = list(atom_names)
    N = len(atom_names)
    types = np.full((N,), -1, dtype=np.int32)
    charges = np.zeros((N,), dtype=float)

    # Fast lookup: res -> atom name -> (type string, q)
    per_res: dict[str, dict[str, tuple[str, float]]] = {}
    for rname, tmpl in ff.residue_templates.items():
        per_res[rname] = {an: (atype, q) for (an, atype, q) in tmpl.atoms}

    for idx, (r, a) in enumerate(zip(res_names, atom_names)):
        rU = (r or "").upper()
        aN = (a or "").strip()
        if rU in per_res and aN in per_res[rU]:
            atype_str, q = per_res[rU][aN]
            if atype_str in ff.lj.type_index:
                types[idx] = ff.lj.type_index[atype_str]
                charges[idx] = q

    return types, charges


def _lookup_bond_param(ti: int, tj: int, table: BondTable) -> tuple[float, float]:
    key = (ti, tj) if ti <= tj else (tj, ti)
    if key not in table.kr_r0:
        raise KeyError(f"Bond type pair not found: {key}")
    return table.kr_r0[key]


def _lookup_angle_param(ti: int, tj: int, tk: int, table: AngleTable) -> tuple[float, float]:
    key = (ti, tj, tk)
    if key in table.kth_th0:
        return table.kth_th0[key]
    rkey = (tk, tj, ti)
    if rkey in table.kth_th0:
        return table.kth_th0[rkey]
    raise KeyError(f"Angle type triplet not found: {key}")


def _lookup_torsion_terms(
    ti: int, tj: int, tk: int, tl: int, table: TorsionTable
) -> list[tuple[int, float, float]]:
    key = (ti, tj, tk, tl)
    if key in table.fourier:
        return table.fourier[key]
    rkey = (tl, tk, tj, ti)
    if rkey in table.fourier:
        return table.fourier[rkey]
    return []  # no terms for this type quadruplet


def build_torsion_fourier_arrays(
    dihedrals: np.ndarray,  # (Nd,4) atom indices
    types: np.ndarray,  # (N,) type ids
    table: TorsionTable,
    t_max: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack variable-length Fourier terms into fixed (Nd, t_max) arrays."""
    Nd = dihedrals.shape[0]
    n = np.zeros((Nd, t_max), dtype=np.int32)
    k_n = np.zeros((Nd, t_max), dtype=float)
    delta = np.zeros((Nd, t_max), dtype=float)
    mask = np.zeros((Nd, t_max), dtype=bool)

    for row, (i, j, k, l) in enumerate(dihedrals.tolist()):
        ti, tj, tk, tl = int(types[i]), int(types[j]), int(types[k]), int(types[l])
        terms = _lookup_torsion_terms(ti, tj, tk, tl, table)
        for m, (per, amp, ph) in enumerate(terms[:t_max]):
            n[row, m] = int(per)
            k_n[row, m] = float(amp)
            delta[row, m] = float(ph)
            mask[row, m] = True

    return n, k_n, delta, mask


def build_bond_instance_params(
    bonds: np.ndarray,  # (Nb,2)
    types: np.ndarray,
    table: BondTable,
) -> tuple[np.ndarray, np.ndarray]:
    Nb = bonds.shape[0]
    kr = np.zeros((Nb,), dtype=float)
    r0 = np.zeros((Nb,), dtype=float)
    for i, (a, b) in enumerate(bonds.tolist()):
        ti, tj = int(types[a]), int(types[b])
        kr[i], r0[i] = _lookup_bond_param(ti, tj, table)
    return kr, r0


def build_angle_instance_params(
    angles: np.ndarray,  # (Na,3)
    types: np.ndarray,
    table: AngleTable,
) -> tuple[np.ndarray, np.ndarray]:
    Na = angles.shape[0]
    kth = np.zeros((Na,), dtype=float)
    th0 = np.zeros((Na,), dtype=float)
    for i, (a, b, c) in enumerate(angles.tolist()):
        ti, tj, tk = int(types[a]), int(types[b]), int(types[c])
        kth[i], th0[i] = _lookup_angle_param(ti, tj, tk, table)
    return kth, th0


def _adjacency_from_bonds(bonds: np.ndarray, n_atoms: int) -> list[set]:
    adj = [set() for _ in range(n_atoms)]
    for i, j in bonds.tolist():
        if i == j:
            continue
        adj[int(i)].add(int(j))
        adj[int(j)].add(int(i))
    return adj


def build_exclusions_and_14_matrices(
    n_atoms: int,
    bonds: np.ndarray,  # (Nb,2)
    dihedrals: np.ndarray,  # (Nd,4)
    scale14_vdw: float,
    scale14_elec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (exclude_mask, scale14_vdw_mat, scale14_elec_mat) each (N,N):
      - 1–2 and 1–3 pairs excluded in nonbonded
      - 1–4 pairs scaled by (scale14_vdw, scale14_elec)
    """
    ex = np.zeros((n_atoms, n_atoms), dtype=bool)
    s_vdw = np.ones((n_atoms, n_atoms), dtype=float)
    s_e = np.ones((n_atoms, n_atoms), dtype=float)

    # 1–2 exclusions
    for i, j in bonds.tolist():
        ex[i, j] = ex[j, i] = True

    # 1–3 exclusions
    adj = _adjacency_from_bonds(bonds, n_atoms)
    for j in range(n_atoms):
        for i in adj[j]:
            for k in adj[j]:
                if i == k:
                    continue
                ex[i, k] = ex[k, i] = True

    # 1–4 scalings (from dihedrals)
    for i, j, k, l in dihedrals.tolist():
        s_vdw[i, l] = s_vdw[l, i] = scale14_vdw
        s_e[i, l] = s_e[l, i] = scale14_elec

    return ex, s_vdw, s_e


def gather_pairwise_from_matrices(pairs: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Alias for compatibility."""
    return gather_pair_values(pairs, mat)


def prepare_bonded_arrays_for_energy(
    bonds: np.ndarray,
    angles: np.ndarray,
    dihedrals: np.ndarray,
    types: np.ndarray,
    ff: FFParams,
    torsion_tmax: int = 4,
):
    """Emit per-instance bonded arrays for energy.py"""
    k_r, r0 = build_bond_instance_params(bonds, types, ff.bonds)
    k_th, th0 = build_angle_instance_params(angles, types, ff.angles)
    n, k_n, delta, active_mask = build_torsion_fourier_arrays(
        dihedrals, types, ff.torsions, torsion_tmax
    )
    return (k_r, r0), (k_th, th0), (n, k_n, delta, active_mask)


def prepare_nonbonded_scaling(
    n_atoms: int,
    bonds: np.ndarray,
    dihedrals: np.ndarray,
    ff: FFParams,
):
    """Return (exclude_mask, scale14_vdw_mat, scale14_elec_mat) for neighbor code."""
    return build_exclusions_and_14_matrices(
        n_atoms, bonds, dihedrals, ff.scale14_vdw, ff.scale14_elec
    )
