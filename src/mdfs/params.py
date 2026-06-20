"""Force-field parameter extraction from OpenMM.

Rather than reverse-engineering ``openmm.app.ForceField`` internals (which are
private and version-fragile), ``mdfs`` lets OpenMM do parameter assignment via
the public :meth:`ForceField.createSystem` and then reads fully-resolved,
per-instance parameters off the resulting :class:`openmm.System` forces:

- :class:`HarmonicBondForce` -> per-bond ``(i, j, r0, k)``
- :class:`HarmonicAngleForce` -> per-angle ``(i, j, k, theta0, k)``
- :class:`PeriodicTorsionForce` -> per-torsion ``(i, j, k, l, periodicity, phase, k)``
  (this includes both proper and improper torsions)
- :class:`NonbondedForce` -> per-particle ``(charge, sigma, epsilon)`` and the
  exception list (1-2/1-3 exclusions and scaled 1-4 interactions)

OpenMM resolves atom typing, wildcard torsion matching, and 1-4 scaling for us.
``CMAPTorsionForce`` (ff19SB's backbone correction) is intentionally **not**
extracted -- see the project README for this documented limitation.

Everything is returned in mdfs units (nm, ps, amu, kJ/mol, e); see
:mod:`mdfs.constants`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from openmm import (
    HarmonicAngleForce,
    HarmonicBondForce,
    NonbondedForce,
    PeriodicTorsionForce,
    System,
    unit,
)
from openmm.app import ForceField, Modeller, NoCutoff, PDBFile, Topology

from mdfs.constants import ONE_4PI_EPS0
from mdfs.energy import BondedSet, DSFParams, LJMixParams, NonbondedSet
from mdfs.paths import DEFAULT_FFXML
from mdfs.types import StrPath

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SystemParams:
    """Fully-resolved force-field parameters for a single molecular system.

    All arrays are NumPy; indices are zero-based atom indices into ``positions``.
    """

    positions: np.ndarray  # (N, 3) float64, nm
    masses: np.ndarray  # (N,) float64, amu
    box_lengths: np.ndarray | None  # (3,) float64, nm (orthorhombic) or None (vacuum)

    # Bonded terms (per instance).
    bonds: np.ndarray  # (Nb, 2) int32
    bond_k: np.ndarray  # (Nb,) kJ/mol/nm^2
    bond_r0: np.ndarray  # (Nb,) nm
    angles: np.ndarray  # (Na, 3) int32
    angle_k: np.ndarray  # (Na,) kJ/mol/rad^2
    angle_theta0: np.ndarray  # (Na,) rad
    torsions: np.ndarray  # (Nd, 4) int32 (propers + impropers)
    torsion_periodicity: np.ndarray  # (Nd,) float64
    torsion_phase: np.ndarray  # (Nd,) rad
    torsion_k: np.ndarray  # (Nd,) kJ/mol

    # Nonbonded (per particle).
    charges: np.ndarray  # (N,) e
    sigma: np.ndarray  # (N,) nm
    epsilon: np.ndarray  # (N,) kJ/mol

    # Nonbonded special pairs (exceptions): 1-2/1-3 exclusions and scaled 1-4.
    # ``exclude_pairs`` removes these from the main pair loop; ``exc_*`` re-adds the
    # exception-specific interaction (pure exclusions carry qq = eps = 0 -> contribute 0).
    exclude_pairs: np.ndarray  # (Ne, 2) int32
    exc_qq: np.ndarray  # (Ne,) e^2 (chargeProd)
    exc_sigma: np.ndarray  # (Ne,) nm
    exc_eps: np.ndarray  # (Ne,) kJ/mol

    # Reference metadata.
    atom_names: list[str]
    res_names: list[str]

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the system."""
        return self.positions.shape[0]

    def exclude_mask(self) -> np.ndarray:
        """Build the dense ``(N, N)`` boolean exclusion mask from exception pairs."""
        n = self.n_atoms
        mask = np.zeros((n, n), dtype=bool)
        if self.exclude_pairs.size:
            i = self.exclude_pairs[:, 0]
            j = self.exclude_pairs[:, 1]
            mask[i, j] = True
            mask[j, i] = True
        return mask


def build_forcefield(ffxml: StrPath | None = None) -> ForceField:
    """Construct an OpenMM ``ForceField`` (defaults to the bundled Amber ff19SB)."""
    path = str(ffxml) if ffxml is not None else str(DEFAULT_FFXML)
    return ForceField(path)


def prepare_topology(
    pdb_path: StrPath,
    ffxml: StrPath | None = None,
    *,
    add_hydrogens: bool = True,
    ph: float = 7.0,
) -> tuple[Topology, unit.Quantity, ForceField]:
    """Load a PDB, optionally add hydrogens, and return ``(topology, positions, ff)``."""
    pdb = PDBFile(str(pdb_path))
    forcefield = build_forcefield(ffxml)
    modeller = Modeller(pdb.topology, pdb.positions)
    if add_hydrogens:
        modeller.addHydrogens(forcefield, pH=ph)
    return modeller.topology, modeller.positions, forcefield


def extract_system_params(
    system: System,
    topology: Topology,
    positions: unit.Quantity,
) -> SystemParams:
    """Read fully-resolved per-instance parameters from an OpenMM ``System``."""
    n = system.getNumParticles()

    pos = np.array(positions.value_in_unit(unit.nanometer), dtype=np.float64)
    masses = np.array(
        [system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(n)],
        dtype=np.float64,
    )

    box_lengths: np.ndarray | None = None
    box_vecs = topology.getPeriodicBoxVectors()
    if box_vecs is not None:
        vecs = np.array(box_vecs.value_in_unit(unit.nanometer), dtype=np.float64)
        off_diagonal = float(np.max(np.abs(vecs - np.diag(np.diag(vecs)))))
        if off_diagonal > 1e-6:
            logger.warning(
                "Triclinic box detected (max off-diagonal %.3g nm); mdfs supports only "
                "orthorhombic PBC, so off-diagonal components are ignored.",
                off_diagonal,
            )
        box_lengths = np.array([vecs[0, 0], vecs[1, 1], vecs[2, 2]], dtype=np.float64)

    forces = {f.__class__.__name__: f for f in system.getForces()}
    if "CMAPTorsionForce" in forces:
        logger.warning(
            "CMAPTorsionForce present in the OpenMM system but not extracted; "
            "mdfs energetics omit the ff19SB backbone CMAP correction (documented limitation)."
        )

    bonds, bond_k, bond_r0 = _extract_bonds(forces.get("HarmonicBondForce"))
    angles, angle_k, angle_theta0 = _extract_angles(forces.get("HarmonicAngleForce"))
    torsions, periodicity, phase, torsion_k = _extract_torsions(forces.get("PeriodicTorsionForce"))
    charges, sigma, epsilon, exclude_pairs, exc_qq, exc_sigma, exc_eps = _extract_nonbonded(
        forces.get("NonbondedForce"), n
    )

    atoms = list(topology.atoms())
    atom_names = [a.name for a in atoms]
    res_names = [a.residue.name for a in atoms]

    return SystemParams(
        positions=pos,
        masses=masses,
        box_lengths=box_lengths,
        bonds=bonds,
        bond_k=bond_k,
        bond_r0=bond_r0,
        angles=angles,
        angle_k=angle_k,
        angle_theta0=angle_theta0,
        torsions=torsions,
        torsion_periodicity=periodicity,
        torsion_phase=phase,
        torsion_k=torsion_k,
        charges=charges,
        sigma=sigma,
        epsilon=epsilon,
        exclude_pairs=exclude_pairs,
        exc_qq=exc_qq,
        exc_sigma=exc_sigma,
        exc_eps=exc_eps,
        atom_names=atom_names,
        res_names=res_names,
    )


def _extract_bonds(
    force: HarmonicBondForce | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if force is None:
        return _empty_int(2), _empty_float(), _empty_float()
    idx, r0, k = [], [], []
    for b in range(force.getNumBonds()):
        i, j, length, kk = force.getBondParameters(b)
        idx.append((i, j))
        r0.append(length.value_in_unit(unit.nanometer))
        k.append(kk.value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2))
    return (
        np.array(idx, dtype=np.int32),
        np.array(k, dtype=np.float64),
        np.array(r0, dtype=np.float64),
    )


def _extract_angles(
    force: HarmonicAngleForce | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if force is None:
        return _empty_int(3), _empty_float(), _empty_float()
    idx, theta0, k = [], [], []
    for a in range(force.getNumAngles()):
        i, j, m, angle, kk = force.getAngleParameters(a)
        idx.append((i, j, m))
        theta0.append(angle.value_in_unit(unit.radian))
        k.append(kk.value_in_unit(unit.kilojoule_per_mole / unit.radian**2))
    return (
        np.array(idx, dtype=np.int32),
        np.array(k, dtype=np.float64),
        np.array(theta0, dtype=np.float64),
    )


def _extract_torsions(
    force: PeriodicTorsionForce | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if force is None:
        return _empty_int(4), _empty_float(), _empty_float(), _empty_float()
    idx, period, phase, k = [], [], [], []
    for t in range(force.getNumTorsions()):
        i, j, m, n, per, ph, kk = force.getTorsionParameters(t)
        idx.append((i, j, m, n))
        period.append(float(per))
        phase.append(ph.value_in_unit(unit.radian))
        k.append(kk.value_in_unit(unit.kilojoule_per_mole))
    return (
        np.array(idx, dtype=np.int32),
        np.array(period, dtype=np.float64),
        np.array(phase, dtype=np.float64),
        np.array(k, dtype=np.float64),
    )


def _extract_nonbonded(
    force: NonbondedForce | None,
    n: int,
) -> tuple[np.ndarray, ...]:
    if force is None:
        raise ValueError("System has no NonbondedForce; cannot build nonbonded parameters.")
    charges = np.zeros(n, dtype=np.float64)
    sigma = np.zeros(n, dtype=np.float64)
    epsilon = np.zeros(n, dtype=np.float64)
    for i in range(force.getNumParticles()):
        q, s, e = force.getParticleParameters(i)
        charges[i] = q.value_in_unit(unit.elementary_charge)
        sigma[i] = s.value_in_unit(unit.nanometer)
        epsilon[i] = e.value_in_unit(unit.kilojoule_per_mole)

    pairs, qq, sig, eps = [], [], [], []
    for x in range(force.getNumExceptions()):
        i, j, charge_prod, s, e = force.getExceptionParameters(x)
        pairs.append((i, j))
        qq.append(charge_prod.value_in_unit(unit.elementary_charge**2))
        sig.append(s.value_in_unit(unit.nanometer))
        eps.append(e.value_in_unit(unit.kilojoule_per_mole))

    exclude_pairs = np.array(pairs, dtype=np.int32) if pairs else _empty_int(2)
    return (
        charges,
        sigma,
        epsilon,
        exclude_pairs,
        np.array(qq, dtype=np.float64),
        np.array(sig, dtype=np.float64),
        np.array(eps, dtype=np.float64),
    )


def _empty_int(width: int) -> np.ndarray:
    return np.zeros((0, width), dtype=np.int32)


def _empty_float() -> np.ndarray:
    return np.zeros((0,), dtype=np.float64)


def system_params_from_pdb(
    pdb_path: StrPath,
    ffxml: StrPath | None = None,
    *,
    add_hydrogens: bool = True,
    ph: float = 7.0,
    nonbonded_method: type = NoCutoff,
    constraints: object | None = None,
    remove_cm_motion: bool = False,
) -> tuple[SystemParams, Topology]:
    """Convenience: PDB -> (optional addH) -> ``createSystem`` -> :class:`SystemParams`.

    Returns the extracted parameters and the (protonated) OpenMM ``Topology`` for
    downstream trajectory writing. Defaults build a vacuum, unconstrained system.
    """
    topology, positions, forcefield = prepare_topology(
        pdb_path, ffxml, add_hydrogens=add_hydrogens, ph=ph
    )
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=nonbonded_method,
        constraints=constraints,
        removeCMMotion=remove_cm_motion,
    )
    return extract_system_params(system, topology, positions), topology


def to_bonded_set(sp: SystemParams) -> BondedSet:
    """Convert extracted parameters into a JAX :class:`~mdfs.energy.BondedSet`."""
    return BondedSet(
        bonds=jnp.asarray(sp.bonds),
        k_r=jnp.asarray(sp.bond_k),
        r0=jnp.asarray(sp.bond_r0),
        angles=jnp.asarray(sp.angles),
        k_theta=jnp.asarray(sp.angle_k),
        theta0=jnp.asarray(sp.angle_theta0),
        torsions=jnp.asarray(sp.torsions),
        periodicity=jnp.asarray(sp.torsion_periodicity),
        torsion_k=jnp.asarray(sp.torsion_k),
        phase=jnp.asarray(sp.torsion_phase),
    )


def to_nonbonded_set(
    sp: SystemParams,
    pairs: np.ndarray | jax.Array,
    *,
    k_e: float = ONE_4PI_EPS0,
    r_cut_lj: float | None = None,
    dsf: DSFParams | None = None,
    shift_lj: bool = False,
) -> NonbondedSet:
    """Convert extracted parameters into a JAX :class:`~mdfs.energy.NonbondedSet`.

    ``pairs`` is the main-loop pair list (e.g. :func:`mdfs.partition.all_pairs`).
    Defaults give the plain, no-cutoff vacuum form that matches OpenMM ``NoCutoff``.
    """
    return NonbondedSet(
        pairs=jnp.asarray(pairs),
        types=jnp.arange(sp.n_atoms),
        q=jnp.asarray(sp.charges),
        lj_params=LJMixParams(eps_type=jnp.asarray(sp.epsilon), sig_type=jnp.asarray(sp.sigma)),
        exclude_mask=jnp.asarray(sp.exclude_mask()),
        exc_pairs=jnp.asarray(sp.exclude_pairs),
        exc_qq=jnp.asarray(sp.exc_qq),
        exc_sigma=jnp.asarray(sp.exc_sigma),
        exc_eps=jnp.asarray(sp.exc_eps),
        k_e=k_e,
        r_cut_lj=r_cut_lj,
        dsf=dsf,
        shift_lj=shift_lj,
    )
