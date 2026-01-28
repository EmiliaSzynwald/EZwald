"""Geometric utilities for lattice operations and periodic boundary conditions.

This module provides functions for computing reciprocal vectors, generating
lattice grids, handling periodic boundary conditions, and computing distances.
"""
from typing import Sequence
import jax
import jax.numpy as jnp


##### Basic Lattice Utilities #####
def calc_recvec(latvec):
    """Calculate reciprocal lattice vectors.

    Args:
        latvec (jnp.array): Direct lattice vectors in row-major format.
            Shape (ndim, ndim).

    Returns:
        jnp.array: Reciprocal lattice vectors in row-major format.
            Shape (ndim, ndim). Computed as 2*pi times the inverse transpose.
    """
    return 2*jnp.pi*jnp.linalg.inv(latvec).T


def calc_volume(latvec):
    """Calculate the volume (or area in 2D) of a unit cell.

    Args:
        latvec (jnp.array): Lattice vectors in row-major format.
            Shape (ndim, ndim).

    Returns:
        float: Volume (or area) of the unit cell.
    """
    return jnp.abs(jnp.linalg.det(latvec))


def gen_ticks(mesh: Sequence[int], kspace:bool=True):
    """Generate ticks to discretize space.

    Args:
        mesh (Sequence[int]): Number of grid points in each dimension.
        kspace (bool, optional): If True, generate k-space ticks centered around
            zero using FFT frequencies. If False, generate real-space ticks
            starting from zero. Default is True.

    Returns:
        list[jnp.array]: List of tick arrays, one for each dimension.
    """
    if kspace:
        ticks = [jnp.around(jnp.fft.fftfreq(nx)*nx).astype(int) for nx in mesh]
    else:
        ticks = [jnp.arange(nx) for nx in mesh]
    return ticks


def gen_indices(ticks, positive:bool=False):
    """Generate Miller indices from tick arrays.

    Creates all combinations of indices from the tick arrays, optionally
    filtering to only positive indices (first non-zero component must be positive).

    Args:
        ticks (list[jnp.array]): List of tick arrays for each dimension.
        positive (bool, optional): If True, only return indices in the positive
            quadrant. Default is False.

    Returns:
        jnp.array: Miller indices. Shape (M, ndim) where M is the number
            of combinations.
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
    """Generate reciprocal lattice vectors from reciprocal lattice vectors.

    Args:
        recvec (jnp.array): Reciprocal lattice basis vectors in row-major format.
            Shape (ndim, ndim).
        mesh (Sequence[int]): Number of grid points in each dimension.

    Returns:
        jnp.array: Reciprocal lattice vectors. Shape (M, ndim) where
            M is the product of mesh values.
    """
    ticks = gen_ticks(mesh)
    return gen_indices(ticks) @ recvec


def gen_lattice(latvec, mesh: Sequence[int], kspace:bool=True, positive:bool=False):
    """Generate Bravais lattice points from lattice vectors.

    Args:
        latvec (jnp.array): Direct lattice vectors in row-major format.
            Shape (ndim, ndim).
        mesh (Sequence[int]): Number of grid points in each dimension.
        kspace (bool, optional): If True, generate k-space lattice centered
            around zero. If False, generate real-space lattice starting from
            zero. Default is True.
        positive (bool, optional): If True, only return points in the positive
            quadrant. Default is False.

    Returns:
        jnp.array: Lattice points. Shape (M, ndim) where M is the number
            of generated points.
    """
    ticks = gen_ticks(mesh, kspace=kspace)
    return gen_indices(ticks, positive=positive) @ latvec


def calc_rwsc(latvec, nx:int=3):
    """Calculate the inscribing radius of the Wigner-Seitz cell.

    The Wigner-Seitz cell is the Voronoi cell of the lattice. This function
    approximates the radius of the largest sphere that can be inscribed.

    Args:
        latvec (jnp.array): Lattice vectors in row-major format.
            Shape (ndim, ndim).
        nx (int, optional): Number of neighboring cells to search in each
            direction. Default is 3.

    Returns:
        float: Radius of the inscribed sphere in the Wigner-Seitz cell.
    """
    mesh = (2*nx+1,)*len(latvec)
    pos = gen_lattice(latvec, mesh)
    rmin = jnp.linalg.norm(pos, axis=-1)[1:].min()
    return rmin/2


def guess_kmesh(recvec, kcut: float) -> Sequence[int]:
    """Guess an appropriate k-space mesh size for a given cutoff.

    Determines the mesh size needed to include all k-vectors within the
    specified cutoff by examining the first shell of reciprocal lattice neighbors.

    Args:
        recvec (jnp.array): Reciprocal lattice vectors in row-major format.
            Shape (ndim, ndim).
        kcut (float): Reciprocal space cutoff magnitude.

    Returns:
        Sequence[int]: Suggested mesh size for each dimension.
    """
    ndim = len(recvec)
    # first shell of neighbors in reciprocal space
    kpts = gen_kvecs(recvec, (3,) * ndim)
    # determine maximum number of shells needed to reach kcut
    kmags = jnp.linalg.norm(kpts[1:], axis=-1)
    nmax = jnp.ceil(kcut / kmags).astype(int).max()
    kmesh = (2 * nmax,) * ndim
    return kmesh


def gen_ksphere(
    cell,
    k_cut: float,
    twist: Sequence[float] = None,
    margin: float = 0.2,
    positive: bool = False,
):
    """Generate k-vectors within a sphere in reciprocal space.

    Args:
        cell (jnp.array): Direct lattice vectors in row-major format.
            Shape (ndim, ndim).
        k_cut (float): Cutoff radius for k-vectors.
        twist (Sequence[float], optional): Twist vector to shift the k-sphere
            center. Default is None (center at origin).
        margin (float, optional): Margin to extend the mesh size beyond the
            cutoff. Default is 0.2.
        positive (bool, optional): If True, only return k-vectors in the
            positive quadrant. Default is False.

    Returns:
        jnp.array: k-vectors within the sphere. Shape (M, ndim) where
            M is the number of k-vectors inside the sphere.
    """
    recvec = calc_recvec(cell)
    qvec = jnp.zeros(len(cell))
    if twist is not None:
        twist = (jnp.asarray(twist) + 0.5) % 1. - 0.5
        qvec = twist @ recvec
    mesh = guess_kmesh(recvec, (1 + margin) * k_cut)
    kvecs = qvec + gen_kvecs(recvec, mesh)
    kmags = jnp.linalg.norm(kvecs, axis=-1)
    sel = kmags < k_cut
    if positive:
        sel = sel & (kvecs[:, 0] >= 0)
        ndim = len(cell)
        for l in range(1, ndim):
            sel = sel & ~((kvecs[:, l-1]==0) & (kvecs[:, l] < 0))
    return kvecs[sel]

def tile(pos, mesht, cell):
    """Tile positions to create a supercell.

    Replicates positions across periodic images to create a larger supercell.

    Args:
        pos (jnp.array): Positions in the unit cell. Shape (N, ndim).
        mesht (Sequence[int]): Tiling factors for each dimension.
        cell (jnp.array): Lattice vectors in row-major format.
            Shape (ndim, ndim).

    Returns:
        jnp.array: Tiled positions. Shape (M, ndim) where M = N * product(mesht).
    """
    ndim = len(cell)
    rvecs = gen_lattice(cell, mesht, kspace=False)
    all_pos = rvecs[:, None] + pos[None, :]
    pos1 = all_pos.reshape(-1, ndim)
    return pos1


##### Minimum Image Convention #####


def determine_cell_type(latvec, ortho_tol=1e-10) -> str:
    """Determine the type of unit cell for optimized PBC handling.

    Classifies the cell as diagonal, orthogonal, or general based on the
    structure of the lattice vectors.

    Args:
        latvec (jnp.array): Lattice vectors in row-major format.
            Shape (ndim, ndim).
        ortho_tol (float, optional): Tolerance for determining orthogonality.
            Default is 1e-10.

    Returns:
        str: Cell type, one of 'diagonal', 'orthogonal', or 'general'.
    """
    is_diagonal = jnp.all(jnp.abs(latvec - jnp.diag(jnp.diag(latvec))) < ortho_tol)
    if is_diagonal:
        return "diagonal"
    is_orthogonal = jnp.all(jnp.abs(jnp.triu(latvec @ latvec.T, k=1)) < ortho_tol)
    if is_orthogonal:
        return "orthogonal"
    return "general"


def gen_pbc_disp_fn(latvec, mode="auto"):
    """Generate a periodic boundary condition displacement function.

    Creates an optimized displacement function based on the cell type. The
    function computes the minimum image displacement between two positions.

    Args:
        latvec (jnp.array): Lattice vectors in row-major format.
            Shape (ndim, ndim).
        mode (str, optional): Mode for PBC handling. Options:
            - 'auto': Automatically determine cell type
            - 'diagonal': Use optimized diagonal cell algorithm
            - 'orthogonal': Use optimized orthogonal cell algorithm
            - 'general': Use general algorithm for arbitrary cells
            Default is 'auto'.

    Returns:
        callable: Displacement function that takes two positions and returns
            the minimum image displacement. Signature: disp_fn(xa, xb) -> disp.

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
    """Compute displacement matrix between two sets of positions.

    Computes pairwise displacements between all positions in xa and xb,
    optionally applying periodic boundary conditions.

    Args:
        xa (jnp.array): First set of positions. Shape (N_a, ndim).
        xb (jnp.array): Second set of positions. Shape (N_b, ndim).
        disp_fn (callable, optional): Periodic boundary condition displacement
            function. If None, uses simple difference. Default is None.

    Returns:
        jnp.array: Displacement matrix. Shape (N_a, N_b, ndim).
    """
    if disp_fn is None:
        return jnp.expand_dims(xa, -2) - jnp.expand_dims(xb, -3)
    else:
        return jax.vmap(jax.vmap(disp_fn, (None, 0)), (0, None))(xa, xb)

def pdist(x, disp_fn=None):
    """Compute pairwise distances between positions.

    Computes the distance matrix for a set of positions, optionally applying
    periodic boundary conditions. Excludes self-distances (diagonal elements).

    Args:
        x (jnp.array): Positions. Shape (..., n, ndim).
        disp_fn (callable, optional): Periodic boundary condition displacement
            function. If None, uses simple difference. Default is None.

    Returns:
        jnp.array: Pairwise distance matrix. Shape (..., n, n) with zeros
            on the diagonal.
    """
    n = x.shape[-2]
    disp = displace_matrix(x, x, disp_fn)
    disp_padded = disp + jnp.eye(n)[..., None]
    dist = jnp.linalg.norm(disp_padded, axis=-1) * (1 - jnp.eye(n))
    return dist

def cdist(xa, xb, disp_fn=None):
    """Compute distances between two sets of positions.

    Computes the distance matrix between all positions in xa and xb,
    optionally applying periodic boundary conditions.

    Args:
        xa (jnp.array): First set of positions. Shape (N_a, ndim).
        xb (jnp.array): Second set of positions. Shape (N_b, ndim).
        disp_fn (callable, optional): Periodic boundary condition displacement
            function. If None, uses simple difference. Default is None.

    Returns:
        jnp.array: Distance matrix. Shape (N_a, N_b).
    """
    disp = displace_matrix(xa, xb, disp_fn)
    dist = jnp.linalg.norm(disp, axis=-1)
    return dist
