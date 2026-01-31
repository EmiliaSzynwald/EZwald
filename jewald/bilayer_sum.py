import os

import jax

import jax.numpy as jnp

from . import geometry as geo
from . import sofk

def gen_positive_gpoints(recvec, g_max):
    # Determine G points to include in reciprocal Ewald sum
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
    """
    Quasi-2D Ewald summation to calculate bi-layer Coulumb interaction
    """

    def __init__(
        self,
        latvec,
        hz,
        attractive=False,
        n_lat=1,
        g_max=200,
        g_threshold=1e-12,
        alpha=None,
        disp_fn_mode='auto',
        gpoints=None,
    ):
        """
        Initilization of the Ewald summation class by preparing
        pbc displace function, lattice displacements, and reciporcal g points

        Args:
            latvec (Array): 3x3 matrix with each row a lattice vector
            hz: positive float value of layer displacement
            attractive (bool): True means interlayer interaction is attractive, default is False.
            n_lat (int): How far to take real-space sum; probably never needs to be changed.
            g_max (int): How far to take reciprocal sum; probably never needs to be changed.
            g_threshold (float): ignore g points below this value. Following DeepSolid value.
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
        # The smallest height of the cell, from reciprocal vectors
        smallest_height = jnp.min(2 * jnp.pi / jnp.linalg.norm(self.recvec, axis=1))
        # rescale accrording to n_lat
        smallest_height *= (2 * n_lat + 1) / 3  # devide by 3 here to keep default
        return 5.0 / smallest_height

    def _prepare_lattice(self, n_lat):

        lattice_displacements = geo.gen_lattice(self.latvec, (2*n_lat+1,)*len(self.latvec))
        lat_norm = jnp.linalg.norm(lattice_displacements[1:], axis=-1)  # skip 0
        simg_const = jnp.sum(jax.lax.erfc(self.alpha * lat_norm) / lat_norm)
        return lattice_displacements, simg_const

    def _prepare_gpoints(self, g_max, g_threshold):  # can be overriden by subclass
        # genetate g points to be used in reciprocal sum. Keep only large gweights
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
        dm1 = self.latvec.shape[-1] - 1
        q2_sum = jnp.sum(charge**2)
        charge_t, charge_b = jnp.array_split(charge, 2)
        e_self = -self.alpha / jnp.sqrt(jnp.pi) * q2_sum
        denom = dm1 * self.cellvolume * self.alpha**dm1
        e_charged_k0 = 0.0
        """double counting; self energy is just 1/2 canceled by self energy defined also comes with 1/2"""
        e_charged_k0 = (
            -(
                2
                * jnp.pi ** (dm1 / 2.0)
                / denom
                * (jnp.sum(charge_t) ** 2 + jnp.sum(charge_b) ** 2)
            )
            / 2
        )
        """here might be some convention difference between paul and the note, here it is the zero K term of Long range part"""
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
        disp = geo.displace_matrix(pos, pos, disp_fn=self.disp_fn)
        rvec = disp[None, :, :, :] + self.lattice_displacements[:, None, None, :]
        r = jnp.linalg.norm(rvec + jnp.eye(pos.shape[0])[..., None], axis=-1)
        charge_ij = charge[:, None] * charge[None, :]
        e_real = jnp.sum(jnp.triu(charge_ij * jax.lax.erfc(self.alpha * r) / r, k=1))
        e_real += 0.5 * jnp.sum(charge**2) * self.simg_const  # self image
        return e_real

    def intralayer_recip_part(self, charge, pos):
        sofk_val = sofk.structure_factor(self.gpoints, pos, charge)
        e_recip = self.gweight @ sofk_val
        return e_recip

    def interlayer_real_part(self, charge_t, pos_t, charge_b, pos_b):
        disp = geo.displace_matrix(pos_t, pos_b, disp_fn=self.disp_fn)
        rvec = disp[None, :, :, :] + self.lattice_displacements[:, None, None, :]
        r = (jnp.linalg.norm(rvec, axis=-1) ** 2 + self.hz**2) ** 0.5
        charge_ij = charge_t[:, None] * charge_b[None, :]
        e_real = jnp.sum(charge_ij * jax.lax.erfc(self.alpha * r) / r)
        return e_real

    def interlayer_recip_part(self, charge_t, pos_t, charge_b, pos_b):
        g_dot_r_t = self.gpoints @ pos_t.T  # [n_gpoints, n_particle]
        rhok_t = jnp.exp(1j * g_dot_r_t) @ charge_t  # [n_gpoints,]
        g_dot_r_b = self.gpoints @ pos_b.T  # [n_gpoints, n_particle]
        rhok_b = jnp.exp(1j * g_dot_r_b) @ charge_b  # [n_gpoints,]
        sofk = (rhok_t * rhok_b.conj()).real
        """here 2* because there is no double counting"""
        e_recip = 2 * self.gweight_interlayer @ sofk
        return e_recip

    def energy(self, charge, pos):
        charge_t, charge_b = jnp.array_split(charge, 2)
        pos_t, pos_b = jnp.array_split(pos, 2)
        """Calculation the Coulomb energy from point charges and their positions"""
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
