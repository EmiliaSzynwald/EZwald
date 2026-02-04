"""Quasi-2D Ewald summation for bilayer (slab) Coulomb interactions.

This module implements Ewald summation for two parallel layers with
periodic boundary conditions in-plane and open boundaries out-of-plane.
Used for bilayer materials (e.g. twisted bilayers) and slab geometries.
"""

import jax
import jax.numpy as jnp
from . import geometry as geo
from . import sofk


def gen_positive_gpoints(recvec, g_max):
    """Generate reciprocal G-vectors in the positive half-space for slab Ewald.

    Builds a set of G-vectors with non-negative indices (half-space) to
    avoid double-counting in the reciprocal sum.

    Args:
        recvec: Array of shape (ndim, ndim). Reciprocal lattice vectors (rows).
        g_max: int. Maximum |G| index per dimension.

    Returns:
        Array of shape (nvec, ndim). G-vectors in Cartesian coordinates.
    """
    n_d = recvec.shape[0]  # number of spatial dimension
    zero = jnp.asarray([0])
    half = jnp.arange(1, g_max + 1)
    full = jnp.arange(-g_max, g_max + 1)
    gpts_list = [
        jnp.meshgrid(*([zero] * ii + [half] + [full] * (n_d - ii - 1)), indexing="ij")
        for ii in range(n_d)
    ]
    gpts = jnp.concatenate(
        [jnp.stack(g, axis=-1).reshape(-1, n_d) for g in gpts_list], axis=0
    )
    gpoints = gpts @ recvec
    return gpoints


def calc_gweight(gpoints, cellvolume, alpha):
    """Reciprocal-space weight for intralayer (2D) Ewald kernel.

    Args:
        gpoints: Array of shape (ng, ndim). Reciprocal G-vectors.
        cellvolume: float. In-plane cell area (2D) or volume (3D).
        alpha: float. Ewald splitting parameter.

    Returns:
        Array of shape (ng,). Weight for each G in the intralayer reciprocal sum.
    """
    if gpoints.shape[-1] == 2:
        gnorm = jnp.linalg.norm(gpoints, axis=-1)
        gweight = 2 * jnp.pi / (cellvolume * gnorm) * jax.lax.erfc(gnorm / (2 * alpha))
    else:  # 3d case
        gsquared = (gpoints**2).sum(-1)
        gweight = (
            4 * jnp.pi / (cellvolume * gsquared) * jnp.exp(-gsquared / (4 * alpha**2))
        )
    return gweight


def calc_gweight_interlayer(gpoints, cellvolume, alpha, hz):
    """Reciprocal-space weight for interlayer Ewald kernel (slab).

    Args:
        gpoints: Array of shape (ng, ndim). Reciprocal G-vectors.
        cellvolume: float. In-plane cell area.
        alpha: float. Ewald splitting parameter.
        hz: float. Interlayer separation (out-of-plane distance).

    Returns:
        Array of shape (ng,). Weight for each G in the interlayer reciprocal sum.
    """
    if gpoints.shape[-1] == 2:
        gnorm = jnp.linalg.norm(gpoints, axis=-1)
        gweight = (
            2
            * jnp.pi
            / (cellvolume * gnorm)
            * (
                jax.lax.exp(gnorm * hz) * jax.lax.erfc(gnorm / (2 * alpha) + hz * alpha)
                + jax.lax.exp(-gnorm * hz)
                * jax.lax.erfc(gnorm / (2 * alpha) - hz * alpha)
            )
        ) / 2.0
    return gweight


class EwaldSumSlab:
    """Quasi-2D Ewald summation for bilayer (slab) Coulomb interaction.

    Attributes:
        latvec: Array. In-plane lattice vectors (rows).
        hz: float. Interlayer separation (positive).
        chargefactor: float. +1 or -1 for repulsive/attractive interlayer.
        recvec: Array. Reciprocal lattice vectors.
        cellvolume: float. In-plane cell area.
        alpha: float. Ewald splitting parameter.
        disp_fn: Callable. Minimum-image displacement for in-plane PBC.
        lattice_displacements: Array. Real-space lattice vectors for sum.
        gpoints: Array. Reciprocal G-vectors used in the sum.
        gweight: Array. Intralayer reciprocal weights.
        gweight_interlayer: Array. Interlayer reciprocal weights.
    """

    def __init__(
        self,
        latvec,
        hz,
        n_up=0,
        n_down=0,
        attractive=False,
        n_lat=1,
        g_max=200,
        g_threshold=1e-12,
        alpha=None,
        disp_fn_mode='auto',
        gpoints=None,
    ):
        """Initialize the slab Ewald class: PBC displacement, lattice displacements, and reciporcal G-points.

        Args:
            latvec: Array of shape (ndim, ndim). In-plane lattice vectors (rows).
            hz: float. Interlayer separation (must be positive).
            attractive: bool. If True, interlayer interaction is attractive
                (opposite charges). Default False (repulsive).
            n_lat: int. Number of real-space lattice shells. Default 1.
            g_max: int. Maximum G-index for reciprocal sum. Default 200.
            g_threshold: float. Drop G-points with weight below this. Default 1e-12.
            alpha: Optional float. Ewald parameter; if None, guessed from cell. Default None.
            disp_fn_mode: str. Passed to gen_pbc_disp_fn ("auto", "diagonal", etc.). Default "auto".
            gpoints: Optional array. Precomputed G-vectors; if None, built from g_max. Default None.

        Raises:
            ValueError: If hz < 0.
        """
        self.latvec = jnp.asarray(latvec)
        if hz < 0:
            raise ValueError("hz must be positive")
        self.hz = hz
        self.chargefactor = 1.0
        if attractive:
            self.chargefactor = -1.0

        self.recvec = geo.calc_recvec(latvec)
        self.cellvolume = geo.calc_volume(latvec)
        # determine alpha
        self.alpha = self._guess_alpha(n_lat) if alpha is None else alpha
        # minimal image displacement function
        self.disp_fn = geo.gen_pbc_disp_fn(latvec, mode=disp_fn_mode)
        # lattice displacement to be added to disp in real space sum
        self.lattice_displacements, self.simg_const = self._prepare_lattice(n_lat)
        # g points to be used in reciprocal sum
        if gpoints is None:
            self.gpoints, self.gweight, self.gweight_interlayer = self._prepare_gpoints(
                g_max, g_threshold
            )
        else:
            self.gpoints = gpoints
            self.gweight = calc_gweight(gpoints, self.cellvolume, self.alpha)
            self.gweight_interlayer = calc_gweight_interlayer(gpoints, self.cellvolume, self.alpha, self.hz)

    def _guess_alpha(self, n_lat):
        """Guess Ewald alpha from cell and n_lat (internal use)."""
        # The smallest height of the cell, from reciprocal vectors
        smallest_height = jnp.min(2 * jnp.pi / jnp.linalg.norm(self.recvec, axis=1))
        # rescale accrording to n_lat
        smallest_height *= (2 * n_lat + 1) / 3  # devide by 3 here to keep default
        return 5.0 / smallest_height

    def _prepare_lattice(self, n_lat):
        """Build real-space lattice displacements and self-image constant (internal use)."""
        lattice_displacements = geo.gen_lattice(self.latvec, (2*n_lat+1,)*len(self.latvec))
        lat_norm = jnp.linalg.norm(lattice_displacements[1:], axis=-1)  # skip 0
        simg_const = jnp.sum(jax.lax.erfc(self.alpha * lat_norm) / lat_norm)
        return lattice_displacements, simg_const

    def _prepare_gpoints(self, g_max, g_threshold):
        """Build G-points and weights, filtering by g_threshold (internal use)."""
        raw_gpoints = gen_positive_gpoints(self.recvec, g_max)
        raw_gweight = calc_gweight(raw_gpoints, self.cellvolume, self.alpha)
        raw_gweight_interlayer = calc_gweight_interlayer(
            raw_gpoints, self.cellvolume, self.alpha, self.hz
        )
        selected_gidx = jnp.logical_or(
            (raw_gweight > g_threshold), (raw_gweight_interlayer > g_threshold)
        )
        gpoints = raw_gpoints[selected_gidx]
        gweight = raw_gweight[selected_gidx]
        gweight_interlayer = raw_gweight_interlayer[selected_gidx]
        return gpoints, gweight, gweight_interlayer

    def const_part(self, charge):
        """Constant (self + k=0) contribution for the slab Ewald sum.

        Args:
            charge: Array of shape (npart,). All particle charges (top then bottom).

        Returns:
            Tuple (e_self, e_charged_k0). Self-energy and k=0 background terms.
        """
        dm1 = self.latvec.shape[-1] - 1
        q2_sum = jnp.sum(charge**2)
        charge_t, charge_b = jnp.split(charge, [n_up])
        e_self = -self.alpha / jnp.sqrt(jnp.pi) * q2_sum
        denom = dm1 * self.cellvolume * self.alpha**dm1
        e_charged_k0 = 0.0
        # double counting; self energy is just 1/2 canceled by self energy defined also comes with 1/2
        e_charged_k0 = (
            -(
                2
                * jnp.pi ** (dm1 / 2.0)
                / denom
                * (jnp.sum(charge_t) ** 2 + jnp.sum(charge_b) ** 2)
            )
            / 2
        )
        # zero K term of long-range part
        e_charged_k0 += -(
            2 * jnp.exp(-((self.alpha * self.hz) ** 2)) * jnp.pi ** (dm1 / 2.0) / denom
            - 2
            * jnp.pi
            * self.hz
            * jax.lax.erfc(self.alpha * self.hz)
            / self.cellvolume
        ) * (jnp.sum(charge_t) * jnp.sum(charge_b))
        return e_self, e_charged_k0

    def intralayer_real_part(self, charge, pos):
        """Real-space intralayer contribution for one layer.

        Args:
            charge: Array of shape (n,). Charges of particles in this layer.
            pos: Array of shape (n, ndim). Positions in this layer.

        Returns:
            float. Real-space intralayer energy for this layer.
        """
        disp = geo.displace_matrix(pos, pos, disp_fn=self.disp_fn)
        rvec = disp[None, :, :, :] + self.lattice_displacements[:, None, None, :]
        r = jnp.linalg.norm(rvec + jnp.eye(pos.shape[0])[..., None], axis=-1)
        charge_ij = charge[:, None] * charge[None, :]
        e_real = jnp.sum(jnp.triu(charge_ij * jax.lax.erfc(self.alpha * r) / r, k=1))
        e_real += 0.5 * jnp.sum(charge**2) * self.simg_const  # self image
        return e_real

    def intralayer_recip_part(self, charge, pos):
        """Reciprocal-space intralayer contribution for one layer.

        Args:
            charge: Array of shape (n,). Charges of particles in this layer.
            pos: Array of shape (n, ndim). Positions in this layer.

        Returns:
            float. Reciprocal-space intralayer energy for this layer.
        """
        sofk_val = sofk.structure_factor(self.gpoints, pos, charge)
        e_recip = self.gweight @ sofk_val
        return e_recip

    def interlayer_real_part(self, charge_t, pos_t, charge_b, pos_b):
        """Real-space interlayer contribution between top and bottom layers.

        Args:
            charge_t: Array of shape (n_t,). Top layer charges.
            pos_t: Array of shape (n_t, ndim). Top layer positions.
            charge_b: Array of shape (n_b,). Bottom layer charges.
            pos_b: Array of shape (n_b, ndim). Bottom layer positions.

        Returns:
            float. Real-space interlayer energy.
        """
        disp = geo.displace_matrix(pos_t, pos_b, disp_fn=self.disp_fn)
        rvec = disp[None, :, :, :] + self.lattice_displacements[:, None, None, :]
        r = (jnp.linalg.norm(rvec, axis=-1) ** 2 + self.hz**2) ** 0.5
        charge_ij = charge_t[:, None] * charge_b[None, :]
        e_real = jnp.sum(charge_ij * jax.lax.erfc(self.alpha * r) / r)
        return e_real

    def interlayer_recip_part(self, charge_t, pos_t, charge_b, pos_b):
        """Reciprocal-space interlayer contribution between top and bottom layers.

        Args:
            charge_t: Array of shape (n_t,). Top layer charges.
            pos_t: Array of shape (n_t, ndim). Top layer positions.
            charge_b: Array of shape (n_b,). Bottom layer charges.
            pos_b: Array of shape (n_b, ndim). Bottom layer positions.

        Returns:
            float. Reciprocal-space interlayer energy.
        """
        g_dot_r_t = self.gpoints @ pos_t.T  # [n_gpoints, n_particle]
        rhok_t = jnp.exp(1j * g_dot_r_t) @ charge_t  # [n_gpoints,]
        g_dot_r_b = self.gpoints @ pos_b.T  # [n_gpoints, n_particle]
        rhok_b = jnp.exp(1j * g_dot_r_b) @ charge_b  # [n_gpoints,]
        sofk = (rhok_t * rhok_b.conj()).real
        # factor 2: no double counting for interlayer
        e_recip = 2 * self.gweight_interlayer @ sofk
        return e_recip

    def energy(self, charge, posn, np = None, nd = None):
        """Total Coulomb energy for the bilayer (all terms).

        Args:
            charge: Array of shape (npart,). Charges (first half = top, second = bottom).
            pos: Array of shape (npart, ndim). Positions (first half = top, second = bottom).

        Returns:
            float. Total slab Ewald Coulomb energy.
        """
        if np is None:
             np = len(posn) // 2
        if nd is None:
            nd = len(posn) - np
        n_up = np
        n_down = nd

        charge_t, charge_b = jnp.split(charge, [n_up])
        pos_t, pos_b = jnp.split(pos, [n_up])
        return (
            sum(self.const_part(charge))
            + self.intralayer_real_part(charge_t, pos_t)
            + self.intralayer_real_part(charge_b, pos_b)
            + self.intralayer_recip_part(charge_t, pos_t)
            + self.intralayer_recip_part(charge_b, pos_b)
            + self.interlayer_real_part(charge_t, pos_t, charge_b, pos_b)
            + self.interlayer_recip_part(charge_t, pos_t, charge_b, pos_b)
        )

    def calc_pe(self, elems, r, x):
        """Warpped interface for potential energy from nuclei and electrons"""
        assert elems.shape[0] == r.shape[0]
        assert elems.ndim == 1 and r.ndim == x.ndim == 2
        assert (x.shape[0] % 2) == 0
        charge = jnp.concatenate(
            [
                elems,
                self.chargefactor * jnp.ones(x.shape[0] // 2),
                jnp.ones(x.shape[0] // 2),
            ],
            axis=0,
        )
        pos = jnp.concatenate([r, x], axis=0)
        return self.energy(charge, pos)
