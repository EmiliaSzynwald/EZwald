from typing import Sequence
import jax
import jax.numpy as jnp


##### Basic Lattice Utilities #####
def calc_recvec(latvec):
    return 2*jnp.pi*jnp.linalg.inv(latvec).T


def calc_volume(latvec):
    return jnp.abs(jnp.linalg.det(latvec))


def gen_ticks(mesh: Sequence[int], kspace:bool=True):
    """Generate ticks to discretize space"""
    if kspace:
        ticks = [jnp.around(jnp.fft.fftfreq(nx)*nx).astype(int) for nx in mesh]
    else:
        ticks = [jnp.arange(nx) for nx in mesh]
    return ticks


def gen_indices(ticks, positive:bool=False):
    """Generate Miller indices"""
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
    """Generate reciprocal lattice from rec. latt. vectors"""
    ticks = gen_ticks(mesh)
    return gen_indices(ticks) @ recvec


def gen_lattice(latvec, mesh: Sequence[int], kspace:bool=True, positive:bool=False):
    """Generate Bravais lattice from lattice vectors"""
    ticks = gen_ticks(mesh, kspace=kspace)
    return gen_indices(ticks, positive=positive) @ latvec


def calc_rwsc(latvec, nx:int=3):
    """Calculate the inscribing radius of the Wigner-Seitz cell"""
    mesh = (2*nx+1,)*len(latvec)
    pos = gen_lattice(latvec, mesh)
    rmin = jnp.linalg.norm(pos, axis=-1)[1:].min()
    return rmin/2


def guess_kmesh(recvec, kcut: float) -> Sequence[int]:
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
    ndim = len(cell)
    #cell1 = np.diag(mesht) @ cell
    rvecs = gen_lattice(cell, mesht, kspace=False)
    all_pos = rvecs[:, None] + pos[None, :]
    pos1 = all_pos.reshape(-1, ndim)
    return pos1

##### Minimum Image Convention #####


def determine_cell_type(latvec, ortho_tol=1e-10) -> str:
    is_diagonal = jnp.all(jnp.abs(latvec - jnp.diag(jnp.diag(latvec))) < ortho_tol)
    if is_diagonal:
        return "diagonal"
    is_orthogonal = jnp.all(jnp.abs(jnp.triu(latvec @ latvec.T, k=1)) < ortho_tol)
    if is_orthogonal:
        return "orthogonal"
    return "general"


def gen_pbc_disp_fn(latvec, mode="auto"):
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
    if disp_fn is None:
        return jnp.expand_dims(xa, -2) - jnp.expand_dims(xb, -3)
    else:
        return jax.vmap(jax.vmap(disp_fn, (None, 0)), (0, None))(xa, xb)

def pdist(x, disp_fn=None):
    # x is assumed to have dimension [..., n, 3]
    n = x.shape[-2]
    disp = displace_matrix(x, x, disp_fn)
    disp_padded = disp + jnp.eye(n)[..., None]
    dist = jnp.linalg.norm(disp_padded, axis=-1) * (1 - jnp.eye(n))
    return dist

def cdist(xa, xb, disp_fn=None):
    disp = displace_matrix(xa, xb, disp_fn)
    dist = jnp.linalg.norm(disp, axis=-1)
    return dist
