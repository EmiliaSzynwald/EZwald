"""Geometry and lattice utilities for periodic systems.

This module provides functions for reciprocal vectors, cell volumes,
Bravais lattice generation, minimum-image convention (PBC displacement),
and k-space sampling. Used by the Ewald and lattice modules.
"""

from typing import Sequence
import jax
import jax.numpy as jnp


def calc_recvec(latvec):
    """Compute reciprocal lattice vectors from direct lattice vectors.

    Args:
        latvec: Array of shape (ndim, ndim). Direct lattice vectors (rows).

    Returns:
        Array of shape (ndim, ndim). Reciprocal lattice vectors (rows);
        b_i = 2*pi * (a_j x a_k) / omega for cyclic i,j,k.
    """
    return 2*jnp.pi*jnp.linalg.inv(latvec).T


def calc_volume(latvec):
    """Compute cell volume (or area in 2D) from lattice vectors.

    Args:
        latvec: Array of shape (ndim, ndim). Lattice vectors (rows).

    Returns:
        float. Absolute value of the determinant (cell volume).
    """
    return jnp.abs(jnp.linalg.det(latvec))


def gen_ticks(mesh: Sequence[int], kspace: bool = True):
    """Generate index ticks to discretize direct or reciprocal space.

    Args:
        mesh: Sequence of int. Number of points along each dimension.
        kspace: bool. If True, use FFT-style (centered) ticks for k-space;
            if False, use 0..nx-1 for real space. Default True.

    Returns:
        List of arrays. One array per dimension; integer indices.
    """
    if kspace:
        ticks = [jnp.around(jnp.fft.fftfreq(nx)*nx).astype(int) for nx in mesh]
    else:
        ticks = [jnp.arange(nx) for nx in mesh]
    return ticks


def gen_indices(ticks, positive: bool = False):
    """Generate all Miller index combinations from ticks.

    Args:
        ticks: List of 1D arrays; index values per dimension.
        positive: bool. If True, restrict to half-space (e.g. k_z >= 0)
            to avoid double-counting. Default False.

    Returns:
        Array of shape (prod(mesh), ndim). Integer Miller indices.
    """
    ndim = len(ticks)
    idx = jnp.stack(
        jnp.meshgrid(*ticks, indexing='ij'), axis=-1
    ).reshape(-1, ndim)
    if positive:
        sel = idx[:, 0] >= 0
        for l in range(1, ndim):
            sel = sel & ~((idx[:, l-1]==0) & (idx[:, l] < 0))
        idx = idx[sel]
    return idx


def gen_kvecs(recvec, mesh: Sequence[int]):
    """Generate reciprocal-space grid from reciprocal vectors and mesh.

    Args:
        recvec: Array of shape (ndim, ndim). Reciprocal lattice vectors (rows).
        mesh: Sequence of int. Number of k-points per dimension.

    Returns:
        Array of shape (prod(mesh), ndim). k-vectors in Cartesian coordinates.
    """
    ticks = gen_ticks(mesh)
    return gen_indices(ticks) @ recvec


def gen_lattice(latvec, mesh: Sequence[int], kspace: bool = True, positive: bool = False):
    """Generate Bravais lattice points from lattice vectors and mesh.

    Args:
        latvec: Array of shape (ndim, ndim). Lattice vectors (rows).
        mesh: Sequence of int. Number of points per dimension.
        kspace: bool. If True, use FFT-style ticks (for k-space); if False,
            use 0..n-1 (for real-space supercells). Default True.
        positive: bool. If True, restrict to half-space indices. Default False.

    Returns:
        Array of shape (prod(mesh), ndim). Lattice point coordinates.
    """
    ticks = gen_ticks(mesh, kspace=kspace)
    return gen_indices(ticks, positive=positive) @ latvec


def calc_rwsc(latvec, nx: int = 3):
    """Calculate the inscribing radius of the Wigner-Seitz cell.

    Uses a (2*nx+1)^ndim grid of lattice points and returns half the
    minimum distance to a neighbor (inscribing radius).

    Args:
        latvec: Array of shape (ndim, ndim). Lattice vectors (rows).
        nx: int. Half-width of the index range per dimension. Default 3.

    Returns:
        float. Inscribing radius (half of minimum image distance).
    """
    mesh = (2*nx+1,)*len(latvec)
    pos = gen_lattice(latvec, mesh)
    rmin = jnp.linalg.norm(pos, axis=-1)[1:].min()
    return rmin/2


def guess_kmesh(recvec, kcut: float) -> Sequence[int]:
    """Estimate k-mesh size so that the first shell reaches kcut.

    Args:
        recvec: Array of shape (ndim, ndim). Reciprocal lattice vectors.
        kcut: float. Desired reciprocal-space cutoff magnitude.

    Returns:
        Tuple of int. Suggested mesh (2*nmax,) per dimension.
    """
    ndim = len(recvec)
    # first shell of neighbors in reciprocal space
    kpts = gen_kvecs(recvec, (3,) * ndim)
    # determine maximum number of shells needed to reach kcut
    kmags = jnp.linalg.norm(kpts[1:], axis=-1)
    nmax = jnp.ceil(kcut / kmags).astype(int).max()
    kmesh = (2 * nmax,) * ndim
    return kmesh


def tile(pos, mesht, cell):
    """Tile particle positions into a supercell via lattice translations.

    Args:
        pos: Array of shape (npart, ndim). Positions in the unit cell.
        mesht: Sequence of int. Supercell multiplicity per dimension.
        cell: Array of shape (ndim, ndim). Lattice vectors (rows).

    Returns:
        Array of shape (npart * prod(mesht), ndim). All image positions.
    """
    ndim = len(cell)
    rvecs = gen_lattice(cell, mesht, kspace=False)
    all_pos = rvecs[:, None] + pos[None, :]
    pos1 = all_pos.reshape(-1, ndim)
    return pos1


def determine_cell_type(latvec, ortho_tol=1e-10) -> str:
    """Classify cell as diagonal, orthogonal, or general for PBC dispatch.

    Args:
        latvec: Array of shape (ndim, ndim). Lattice vectors (rows).
        ortho_tol: float. Tolerance for zero off-diagonal elements. Default 1e-10.

    Returns:
        str. One of "diagonal", "orthogonal", "general".
    """
    is_diagonal = jnp.all(jnp.abs(latvec - jnp.diag(jnp.diag(latvec))) < ortho_tol)
    if is_diagonal:
        return "diagonal"
    is_orthogonal = jnp.all(jnp.abs(jnp.triu(latvec @ latvec.T, k=1)) < ortho_tol)
    if is_orthogonal:
        return "orthogonal"
    return "general"


def gen_pbc_disp_fn(latvec, mode="auto"):
    """Build a minimum-image displacement function for the given cell.

    Args:
        latvec: Array of shape (ndim, ndim). Lattice vectors (rows).
        mode: str. "auto" (detect from latvec), "diagonal", "orthogonal", or
            "general". Default "auto".

    Returns:
        Callable (xa, xb) -> disp. Returns the minimum-image vector xa - xb
        (or equivalent for general cells). Inputs/outputs are (ndim,) arrays.

    Raises:
        ValueError: If mode is not recognized.
    """
    latvec = jnp.asarray(latvec)
    mode = mode.lower()
    if mode == "auto":
        ortho_tol = 1e-10
        mode = determine_cell_type(latvec, ortho_tol=ortho_tol)
    # diagonal cell
    if mode.startswith("diag"):
        latdiag = jnp.diagonal(latvec)
        def diagonal_disp(xa, xb):
            frac_disp = xa / latdiag - xb / latdiag
            shifted_frac_disp = frac_disp - jnp.rint(frac_disp)
            return shifted_frac_disp * latdiag
        return diagonal_disp
    # orthogonal cell
    if mode.startswith("orth"):
        invvec = jnp.linalg.inv(latvec)
        def orthogonal_disp(xa, xb):
            frac_disp = xa @ invvec - xb @ invvec
            shifted_frac_disp = frac_disp - jnp.rint(frac_disp)
            return shifted_frac_disp @ latvec
        return orthogonal_disp
    # general cell
    if mode.startswith("gen"):
        n_lat = 1
        mesh = (2*n_lat+1,)*len(latvec)
        images = gen_lattice(latvec, mesh)
        invvec = jnp.linalg.inv(latvec)
        def xpbc(x):  # wrap position into simulation cell
            f = x @ invvec
            return (f % 1) @ latvec
        def monoclinic_disp(xa, xb):
            disps = (xpbc(xa) - xpbc(xb))[None] + images
            dists = jnp.linalg.norm(disps, axis=-1)
            idx = jnp.argmin(dists)
            return disps[idx]
        return monoclinic_disp
    # fail to recognize mode
    raise ValueError(f"unknown mode for gen_pbc_disp_fn: {mode}")


def displace_matrix(xa, xb, disp_fn=None):
    """Pairwise displacement matrix with optional PBC minimum image.

    Args:
        xa: Array of shape (na, ndim). First set of positions.
        xb: Array of shape (nb, ndim). Second set of positions.
        disp_fn: Optional callable (x1, x2) -> disp. If None, use plain
            difference xa - xb (no PBC).

    Returns:
        Array of shape (na, nb, ndim). Displacement xa[i] - xb[j] (or
        minimum-image equivalent if disp_fn is set).
    """
    if disp_fn is None:
        return jnp.expand_dims(xa, -2) - jnp.expand_dims(xb, -3)
    else:
        return jax.vmap(jax.vmap(disp_fn, (None, 0)), (0, None))(xa, xb)


def get_nvecs(axes, pos, atol=1e-10):
    """Find integer lattice indices corresponding to given positions.

    Computes n such that pos ≈ n @ axes (rounded to nearest integer).

    Args:
        axes: Array of shape (ndim, ndim). Lattice vectors (rows).
        pos: Array of shape (..., ndim). Positions in Cartesian coordinates.
        atol: float. Not used; kept for API compatibility. Default 1e-10.

    Returns:
        Array of integers, same shape as pos. Miller indices.
    """
    inv_axes = jnp.linalg.inv(axes)
    ncands = pos @ inv_axes
    nvecs = jnp.rint(ncands).astype(int)
    # Optional check omitted for JAX performance, relying on logic
    return nvecs


def pos_in_axes(axes, pos, ztol=1e-10):
    """Fold positions into the unit cell (fractional coords in [0,1), then back).

    Positions on the boundary (within ztol of 1) are mapped to 0.

    Args:
        axes: Array of shape (ndim, ndim). Lattice vectors (rows).
        pos: Array of shape (..., ndim). Positions in Cartesian coordinates.
        ztol: float. Tolerance for treating fractional coord 1 as 0. Default 1e-10.

    Returns:
        Array, same shape as pos. Positions folded into [0,1) @ axes.
    """
    upos = pos @ jnp.linalg.inv(axes)
    u = upos % 1
    d = jnp.abs(u - 1)
    u = jnp.where(d < ztol, 0, u)
    pos0 = u @ axes
    return pos0


def get_ksphere(raxes, kc, margin=0.2, twist=None):
    """Generate k-vectors with |k| < kc on the reciprocal lattice (+ optional twist).

    Args:
        raxes: Array of shape (ndim, ndim). Reciprocal lattice vectors (rows).
        kc: float. Cutoff magnitude; only k with |k| < kc are returned.
        margin: float. Internal mesh uses (1+margin)*kc to ensure coverage. Default 0.2.
        twist: Optional array of shape (ndim,). Twist vector in fractional
            reciprocal coordinates; added to all k. Default None (no twist).

    Returns:
        Array of shape (nvec, ndim). k-vectors satisfying |k| < kc.
    """
    ndim = raxes.shape[0]
    kmesh = guess_kmesh(raxes, (1+margin)*kc)
    qvec = jnp.zeros(ndim)
    if twist is not None:
        qvec = jnp.dot(twist, raxes)
    kvecs = gen_lattice(raxes, kmesh, kspace=True) + qvec
    kmags = jnp.linalg.norm(kvecs, axis=-1)
    ksel = kmags < kc
    return kvecs[ksel]
