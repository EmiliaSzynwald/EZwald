"""Ewald summation for periodic Coulomb interactions in 2D and 3D.

This module implements the standard Ewald decomposition of the Coulomb sum
into real-space (short-range), reciprocal-space (long-range), and constant
(self + background) terms. Supports both 2D and 3D periodic boundary
conditions.
"""

import jax.numpy as jnp
from jax.scipy.special import erfc
from jewald import lattice, sofk, geometry as geo


def make_ewald(axes, rckc=30.0):
  """Build Ewald object and lattice vectors for a given cell.

  Args:
    axes: Array of shape (ndim, ndim). Lattice vectors in row-major form
        (each row is a lattice vector).
    rckc: float. Product of real-space cutoff and reciprocal-space cutoff;
        controls Ewald convergence (default 30.0).

  Returns:
    ew: Ewald instance configured for the cell.
    rvecs: Real-space lattice displacement vectors for the Ewald sum.
    kvecs: Reciprocal-space vectors (excluding k=0) for the Ewald sum.
  """
  rvecs, kvecs = lattice.make_lattice(axes, rckc)
  rc, kc = lattice.compute_cutoffs(axes, rckc)
  a = alpha(rc, kc)
  ndim = len(axes)
  omega = geo.calc_volume(axes)
  ew = Ewald(a, ndim, omega)
  return ew, rvecs, kvecs


def alpha(rc, kc):
  """Ewald splitting parameter from real- and reciprocal-space cutoffs.

  Args:
    rc: float. Real-space cutoff distance.
    kc: float. Reciprocal-space cutoff magnitude.

  Returns:
    float. Splitting parameter alpha = sqrt(kc / (2*rc)).
  """
  return jnp.sqrt(kc/(2*rc))


class Ewald:
  """Ewald summation for Coulomb energy in a periodic cell.

  Attributes:
    alpha: float. Ewald splitting parameter.
    ndim: int. Number of spatial dimensions (2 or 3).
    omega: float. Cell volume (or area in 2D).
  """

  def __init__(self, alpha, ndim, omega):
    self.alpha = alpha
    self.ndim = ndim
    self.omega = omega

  def sum(self, pos, charge, rvecs, kvecs):
    """Total Coulomb energy: constant + real-space + reciprocal-space terms.

    Args:
      pos: Array of shape (npart, ndim). Particle positions in Cartesian
          coordinates.
      charge: Array of shape (npart,). Particle charges.
      rvecs: Real-space lattice displacement vectors.
      kvecs: Reciprocal-space vectors (excluding k=0).

    Returns:
      float. Total Ewald Coulomb energy.
    """
    vconst = self.constant(charge)
    vsr = self.sum_sr(pos, charge, rvecs)
    vlr = self.sum_lr(pos, charge, kvecs)
    return vconst + vsr + vlr

  def vsr_k0(self):
    """Reciprocal-space k=0 coefficient for the short-range (neutralizing) term.

    Returns:
      float. Coefficient for the charge-squared sum in the constant term.
    """
    dm1 = self.ndim-1
    denom = dm1*self.alpha**dm1*self.omega
    return 2*jnp.pi**(dm1/2.)/denom

  def vlr_r0(self):
    """Real-space r=0 coefficient for the long-range (self) term.

    Returns:
      float. Coefficient for the self-energy in the constant term.
    """
    return 2*self.alpha/jnp.pi**0.5

  def constant(self, charge):
    """Constant term: self-energy and neutralizing background.

    Args:
      charge: Array of shape (npart,). Particle charges.

    Returns:
      float. Constant (k=0, r=0) contribution to the Ewald energy.
    """
    vsr_k0 = self.vsr_k0()
    vlr_r0 = self.vlr_r0()
    q2_sum = jnp.sum(charge**2)
    q_sum_sq = jnp.sum(charge)**2
    ebg = - 0.5 * q_sum_sq * vsr_k0
    vconst = -0.5 * q2_sum * vlr_r0 + ebg
    return vconst

  def fvsr_r(self, r):
    """Short-range (real-space) pair potential: erfc(alpha*r)/r.

    Args:
      r: Array of pair distances.

    Returns:
      Array. Same shape as r; potential values.
    """
    return erfc(self.alpha*r)/r

  def sum_sr(self, pos, charge, rvecs):
    """Real-space (short-range) sum over all pairs and lattice images.

    Args:
      pos: Array of shape (npart, ndim). Particle positions.
      charge: Array of shape (npart,). Particle charges.
      rvecs: Real-space lattice displacement vectors.

    Returns:
      float. Real-space contribution to the Ewald energy.
    """
    esr = 0.0
    npart = len(pos)
    if npart == 1:
      return esr
    idx = jnp.triu_indices(npart, k=1)
    q_pairs = charge[idx[0]] * charge[idx[1]]
    shifts = rvecs
    esrl = []
    for shift in shifts:
      drij = (pos[:, None] - (pos+shift)[None, :])[idx]
      rij = jnp.linalg.norm(drij, axis=-1)
      esr1 = (self.fvsr_r(rij) * q_pairs).sum()
      esrl.append(esr1)
    esr = jnp.sum(jnp.array(esrl))
    return esr

  def fvlr_k2d(self, k):
    """Long-range (reciprocal-space) potential kernel for 2D.

    Args:
      k: Array of reciprocal-space magnitudes.

    Returns:
      Array. Same shape as k; 2D reciprocal-space kernel.
    """
    dm1 = self.ndim-1
    ak = k/(2*self.alpha)
    vk = 2*dm1*jnp.pi/k**dm1/self.omega
    return vk*erfc(ak)

  def fvlr_k3d(self, k):
    """Long-range (reciprocal-space) potential kernel for 3D.

    Args:
      k: Array of reciprocal-space magnitudes.

    Returns:
      Array. Same shape as k; 3D reciprocal-space kernel.
    """
    dm1 = self.ndim-1
    ak = k/(2*self.alpha)
    vk = 2*dm1*jnp.pi/k**dm1/self.omega
    return vk*jnp.exp(-ak**2)

  def sum_lr(self, pos, charge, kvecs):
    """Reciprocal-space (long-range) sum via structure factor.

    Args:
      pos: Array of shape (npart, ndim). Particle positions.
      charge: Array of shape (npart,). Particle charges.
      kvecs: Reciprocal-space vectors (excluding k=0).

    Returns:
      float. Reciprocal-space contribution to the Ewald energy.
    """
    kmags = jnp.linalg.norm(kvecs, axis=-1)
    vlr_k = self.fvlr_k2d(kmags) if self.ndim == 2 else self.fvlr_k3d(kmags)
    sk = sofk.structure_factor(kvecs, pos, charge)
    elr = 0.5*jnp.dot(sk, vlr_k)
    return elr
