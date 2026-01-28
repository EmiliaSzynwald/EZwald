"""Ewald summation for bilayer (quasi-2D) systems.

This module implements quasi-2D Ewald summation for computing Coulomb interactions
in bilayer systems with periodic boundary conditions in the plane and open boundary
conditions in the perpendicular direction.
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .geometry import displace_matrix, gen_pbc_disp_fn, gen_lattice


def gen_lattice_displacements(latvec, n_lat):
    """Generate lattice displacement vectors for real-space Ewald summation.

    Creates a grid of lattice displacement vectors by generating all combinations
    of integer multiples of lattice vectors up to n_lat in each direction.

    Args:
        latvec (jnp.array): Lattice vectors in row-major format. Shape (ndim, ndim).
        n_lat (int): Maximum number of lattice cells to include in each direction.
            Generates displacements from -n_lat to n_lat in each dimension.

    Returns:
        jnp.array: Lattice displacement vectors.
    """
    n_d = latvec.shape[0]  # number of spatial dimension
    XYZ = jnp.meshgrid(*[jnp.arange(-n_lat, n_lat + 1)] * n_d, indexing="ij")
    xyz = jnp.stack(XYZ, axis=-1).reshape((-1, n_d))
    return jnp.dot(xyz, latvec)


def gen_positive_gpoints(recvec, g_max):
    """Generate positive reciprocal lattice points for Ewald summation.

    Determines G points to include in the reciprocal Ewald sum by generating
    points in the positive quadrant of reciprocal space.

    Args:
        recvec (jnp.array): Reciprocal lattice vectors in row-major format.
            Shape (ndim, ndim).
        g_max (int): Maximum magnitude of reciprocal lattice indices to include.

    Returns:
        jnp.array: Reciprocal lattice points (G vectors). Shape (M, ndim) where
            M is the number of generated G points.
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
    gpoints = 2 * jnp.pi * gpts @ recvec
    return gpoints


def calc_gweight(gpoints, cellvolume, alpha):
    """Calculate the Ewald weight for reciprocal space points.

    Computes the weight factor for each reciprocal space point in the Ewald sum.
    The formula differs for 2D and 3D systems.

    Args:
        gpoints (jnp.array): Reciprocal space points (G vectors). Shape (M, ndim).
        cellvolume (float): Volume (or area in 2D) of the unit cell.
        alpha (float): Ewald splitting parameter.

    Returns:
        jnp.array: Weight factors for each G point.
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
    """Calculate the Ewald weight for interlayer interactions in reciprocal space.

    Computes the weight factor for interlayer interactions between two layers
    separated by distance hz.

    Args:
        gpoints (jnp.array): Reciprocal space points (G vectors). Shape (M, ndim).
        cellvolume (float): Volume (or area in 2D) of the unit cell.
        alpha (float): Ewald splitting parameter.
        hz (float): Perpendicular separation between layers (must be positive).

    Returns:
        jnp.array: Weight factors for interlayer interactions.
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
    """Quasi-2D Ewald summation for calculating bilayer Coulomb interactions.

    This class implements Ewald summation for bilayer systems.
    It handles both intralayer and interlayer Coulomb interactions.

    Attributes:
        latvec (jnp.array): Lattice vectors in row-major format.
        hz (float): Perpendicular separation between layers.
        attractive (bool): Whether interlayer interaction is attractive (True) or repulsive (False).
        chargefactor (float): Charge factor for interlayer interactions.
        recvec (jnp.array): Reciprocal lattice vectors.
        cellvolume (float): Volume (or area) of the unit cell.
        alpha (float): Ewald splitting parameter.
        disp_fn (callable): Periodic boundary condition displacement function.
        lattice_displacements (jnp.array): Lattice displacement vectors for real-space sum.
        simg_const (float): Self-image constant for real-space sum.
        gpoints (jnp.array): Reciprocal space points.
        gweight (jnp.array): Weight factors for intralayer reciprocal sum.
        gweight_interlayer (jnp.array): Weight factors for interlayer reciprocal sum.
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
        """Initialize the Ewald summation class.

        Prepares the periodic boundary condition displacement function, lattice
        displacements, and reciprocal G points for efficient Ewald summation.

        Args:
            latvec (jnp.array): Lattice vectors in row-major format. Shape (ndim, ndim).
            hz (float): Positive float value of layer separation perpendicular to the plane.
            attractive (bool, optional): If True, interlayer interaction is attractive;
                default is False (repulsive).
            n_lat (int, optional): How far to take real-space sum in each direction.
                Default is 1. Probably never needs to be changed.
            g_max (int, optional): Maximum magnitude of reciprocal lattice indices to include.
                Default is 200. Probably never needs to be changed.
            g_threshold (float, optional): Ignore G points with weight below this value.
                Following DeepSolid convention. Default is 1e-12.
            alpha (float, optional): Ewald splitting parameter. If None, automatically
                determined from lattice parameters. Default is None.
            disp_fn_mode (str, optional): Mode for periodic boundary condition function.
                Options: 'auto', 'diagonal', 'orthogonal', 'general'. Default is 'auto'.
            gpoints (jnp.array, optional): Pre-computed reciprocal space points.
                If None, will be generated automatically. Default is None.

        Raises:
            ValueError: If hz is negative.
        """
        self.latvec = jnp.asarray(latvec)
        if hz < 0:
            raise ValueError("hz must be positive")
        self.hz = hz
        self.attractive = attractive
        self.chargefactor = 1.0
        if attractive:
            self.chargefactor = -1.0

        self.recvec = jnp.linalg.inv(latvec).T
        self.cellvolume = jnp.abs(jnp.linalg.det(latvec))
        # determine alpha
        self.alpha = self._guess_alpha(n_lat) if alpha is None else alpha
        # minimal image displacement function
        self.disp_fn = gen_pbc_disp_fn(latvec, mode=disp_fn_mode)
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
        """Guess an appropriate Ewald splitting parameter.

        Determines alpha based on the smallest height of the cell (from reciprocal
        vectors) and the number of lattice cells included in the real-space sum.

        Args:
            n_lat (int): Number of lattice cells included in real-space sum.

        Returns:
            float: Suggested Ewald splitting parameter alpha.
        """
        # The smallest height of the cell, from reciprocal vectors
        smallest_height = jnp.min(1 / jnp.linalg.norm(self.recvec, axis=1))
        # rescale accrording to n_lat
        smallest_height *= (2 * n_lat + 1) / 3  # devide by 3 here to keep default
        return 5.0 / smallest_height

    def _prepare_lattice(self, n_lat):
        """Prepare lattice displacements and self-image constant for real-space sum.

        Generates lattice displacement vectors and computes the self-image constant
        used in the real-space Ewald summation.

        Args:
            n_lat (int): Number of lattice cells to include in each direction.

        Returns:
            tuple: A tuple containing:
                - lattice_displacements (jnp.array): Lattice displacement vectors.
                - simg_const (float): Self-image constant for real-space sum.
        """
        #lattice_displacements = gen_lattice_displacements(self.latvec, n_lat)
        lattice_displacements = gen_lattice(self.latvec, (2*n_lat+1,)*len(self.latvec))
        lat_norm = jnp.linalg.norm(lattice_displacements[1:], axis=-1)  # skip 0
        simg_const = jnp.sum(jax.lax.erfc(self.alpha * lat_norm) / lat_norm)
        return lattice_displacements, simg_const

    def _prepare_gpoints(self, g_max, g_threshold):
        """Prepare reciprocal space points and weights for Ewald summation.

        Generates G points to be used in the reciprocal sum and filters them
        based on weight thresholds. Keeps only points with significant weights
        for either intralayer or interlayer interactions. Can be overridden by subclass.

        Args:
            g_max (int): Maximum magnitude of reciprocal lattice indices.
            g_threshold (float): Minimum weight threshold for keeping G points.

        Returns:
            tuple: A tuple containing:
                - gpoints (jnp.array): Selected reciprocal space points.
                - gweight (jnp.array): Weight factors for intralayer interactions.
                - gweight_interlayer (jnp.array): Weight factors for interlayer
                    interactions.
        """
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
        """Calculate the constant part of the Ewald energy.

        Computes the self-energy and charged k=0 terms that are independent of
        particle positions. This includes the self-energy correction and the
        neutralizing background terms.

        Args:
            charge (jnp.array): Charge values for all particles.

        Returns:
            tuple: A tuple containing:
                - e_self (float): Self-energy term.
                - e_charged_k0 (float): Charged k=0 term including interlayer contribution.
        """
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
        """Calculate the real-space intralayer Ewald energy.

        Computes the short-range (real-space) contribution to the intralayer
        Coulomb energy using the complementary error function.

        Args:
            charge (jnp.array): Charge values for particles in the layer.
            pos (jnp.array): Positions of particles in the layer.
+
        Returns:
            float: Real-space intralayer energy contribution.
        """
        # if charge.shape[0] < 2:
        #     return 0
        disp = displace_matrix(pos, pos, disp_fn=self.disp_fn)
        rvec = disp[None, :, :, :] + self.lattice_displacements[:, None, None, :]
        r = jnp.linalg.norm(rvec + jnp.eye(pos.shape[0])[..., None], axis=-1)
        charge_ij = charge[:, None] * charge[None, :]
        e_real = jnp.sum(jnp.triu(charge_ij * jax.lax.erfc(self.alpha * r) / r, k=1))
        e_real += 0.5 * jnp.sum(charge**2) * self.simg_const  # self image
        return e_real

    def intralayer_recip_part(self, charge, pos):
        """Calculate the reciprocal-space intralayer Ewald energy.

        Computes the long-range (reciprocal-space) contribution to the intralayer
        Coulomb energy using the structure factor.

        Args:
            charge (jnp.array): Charge values for particles in the layer. 
            pos (jnp.array): Positions of particles in the layer. 

        Returns:
            float: Reciprocal-space intralayer energy contribution.
        """
        g_dot_r = self.gpoints @ pos.T  # [n_gpoints, n_particle]
        rhok = jnp.exp(1j * g_dot_r) @ charge  # [n_gpoints,]
        sofk = (rhok * rhok.conj()).real
        e_recip = self.gweight @ sofk
        return e_recip

    def interlayer_real_part(self, charge_t, pos_t, charge_b, pos_b):
        """Calculate the real-space interlayer Ewald energy.

        Computes the short-range (real-space) contribution to the interlayer
        Coulomb energy between top and bottom layers.

        Args:
            charge_t (jnp.array): Charge values for particles in the top layer.
            pos_t (jnp.array): Positions of particles in the top layer.
            charge_b (jnp.array): Charge values for particles in the bottom layer.
            pos_b (jnp.array): Positions of particles in the bottom layer.

        Returns:
            float: Real-space interlayer energy contribution.
        """
        # if charge_t.shape[0] < 2:
        #     return 0
        # if charge_b.shape[0] < 2:
        #     return 0
        disp = displace_matrix(pos_t, pos_b, disp_fn=self.disp_fn)
        rvec = disp[None, :, :, :] + self.lattice_displacements[:, None, None, :]
        r = (jnp.linalg.norm(rvec, axis=-1) ** 2 + self.hz**2) ** 0.5
        charge_ij = charge_t[:, None] * charge_b[None, :]
        e_real = jnp.sum(charge_ij * jax.lax.erfc(self.alpha * r) / r)
        return e_real

    def interlayer_recip_part(self, charge_t, pos_t, charge_b, pos_b):
        """Calculate the reciprocal-space interlayer Ewald energy.

        Computes the long-range (reciprocal-space) contribution to the interlayer
        Coulomb energy between top and bottom layers using the structure factor.

        Args:
            charge_t (jnp.array): Charge values for particles in the top layer.
            pos_t (jnp.ndarray): Positions of particles in the top layer.
            charge_b (jnp.ndarray): Charge values for particles in the bottom layer.
            pos_b (jnp.ndarray): Positions of particles in the bottom layer.

        Returns:
            float: Reciprocal-space interlayer energy contribution.
        """
        g_dot_r_t = self.gpoints @ pos_t.T  # [n_gpoints, n_particle]
        rhok_t = jnp.exp(1j * g_dot_r_t) @ charge_t  # [n_gpoints,]
        g_dot_r_b = self.gpoints @ pos_b.T  # [n_gpoints, n_particle]
        rhok_b = jnp.exp(1j * g_dot_r_b) @ charge_b  # [n_gpoints,]
        sofk = (rhok_t * rhok_b.conj()).real
        """here 2* because there is no double counting"""
        e_recip = 2 * self.gweight_interlayer @ sofk
        return e_recip

    def energy(self, charge, pos):
        """Calculate the total Coulomb energy for a bilayer system.

        Computes the complete Ewald sum including constant terms, intralayer
        (real and reciprocal space), and interlayer (real and reciprocal space)
        contributions. Assumes charges and positions are ordered with top layer
        first, then bottom layer.

        Args:
            charge (jnp.array): Charge values for all particles.
            pos (jnp.array): Positions of all particles.

        Returns:
            float: Total Coulomb energy of the bilayer system.
        """
        charge_t, charge_b = jnp.array_split(charge, 2)
        pos_t, pos_b = jnp.array_split(pos, 2)
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
        """Calculate potential energy from nuclei and electrons.

        Wrapped interface that constructs charges from nuclear charges (elems)
        and electron positions (x), then computes the total Coulomb energy.
        Assumes electrons are split equally between top and bottom layers.

        Args:
            elems (jnp.array): Nuclear charges. 
            r (jnp.array): Nuclear positions. 
            x (jnp.array): Electron positions. 

        Returns:
            float: Total potential energy from nuclei and electrons.

        Raises:
            AssertionError: If dimensions don't match or electron count is odd.
        """
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
