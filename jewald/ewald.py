"""Ewald summation for periodic systems.

This module implements the Ewald summation method for computing Coulomb interactions
in periodic systems with full periodic boundary conditions.
"""
import jax.numpy as jnp
from jax.scipy.special import erfc
from jewald import axes_pos, lattice, sofk

def make_ewald(axes, rckc=30.0):
  """Create an Ewald summation object and generate lattice vectors.

  Args:
    axes (jnp.array): Lattice vectors in row-major format. 
    rckc (float, optional): Product of real-space and reciprocal-space cutoffs.
        Default is 30.0.

  Returns:
    tuple: A tuple containing:
        - ew (Ewald): Initialized Ewald summation object.
        - rvecs (jnp.array): Real-space lattice vectors.
        - kvecs (jnp.array): Reciprocal-space lattice vectors.
  """
  rvecs, kvecs = lattice.make_lattice(axes, rckc)       #lattice vectors in real and reciprocal space
  rc, kc = lattice.compute_cutoffs(axes, rckc)          #real-space and reciprocal-space cutoff distances
  a = alpha(rc, kc)
  ndim = len(axes)                                      #number of dimensions
  omega = axes_pos.volume(axes)
  ew = Ewald(a, ndim, omega)                            #ewald object
  return ew, rvecs, kvecs

def alpha(rc, kc):
  """Compute the Ewald splitting parameter.

  The splitting parameter alpha balances the convergence of real-space and
  reciprocal-space sums. Larger alpha means faster convergence in real space
  but slower in reciprocal space.

  Args:
    rc (float): Real-space cutoff distance.
    kc (float): Reciprocal-space cutoff distance.

  Returns:
    float: Ewald splitting parameter alpha.
  """
  return jnp.sqrt(kc/(2*rc))

class Ewald:
  """Ewald summation for computing Coulomb energy in periodic systems.

  This class implements the Ewald summation method, which splits the Coulomb
  interaction into short-range (real-space) and long-range (reciprocal-space)
  contributions for efficient computation in periodic systems.

  Attributes:
    alpha (float): Ewald splitting parameter.
    ndim (int): Number of spatial dimensions (2 for 2D, 3 for 3D).
    omega (float): Volume (or area in 2D) of the unit cell.
  """

  def __init__(self, alpha, ndim, omega):
    """Initialize the Ewald summation object.

    Args:
      alpha (float): Ewald splitting parameter.
      ndim (int): Number of spatial dimensions (2 for 2D, 3 for 3D).
      omega (float): Volume (or area in 2D) of the unit cell.
    """
    self.alpha = alpha
    self.ndim = ndim #number of row space dimensions (if = 2 then 2d)
    self.omega = omega #cell volume 

  def sum(self, pos, rvecs, kvecs):
    """Compute the total Ewald energy.

    Calculates the complete Ewald sum including constant terms, real-space
    (short-range) contributions, and reciprocal-space (long-range) contributions.

    Args:
      pos (jnp.array): Particle positions. Shape (N, ndim).
      rvecs (jnp.array): Real-space lattice vectors for periodic images.
          Shape (M_r, ndim).
      kvecs (jnp.array): Reciprocal-space lattice vectors. Shape (M_k, ndim).

    Returns:
      float: Total Ewald energy (constant + short-range + long-range).
    """
    vconst = self.constant(len(pos)) 
    vsr = self.sum_sr(pos, rvecs) #real-space sum
    vlr = self.sum_lr(pos, kvecs) #reciprocal-space sum
    return vconst + vsr + vlr #total Ewald energy: constant + short-range + long-range

  def vsr_k0(self):
    """Compute the k=0 term for the short-range (real-space) potential.

    This is the neutralizing background term for the real-space sum.

    Returns:
      float: k=0 contribution to short-range potential.
    """
    dm1 = self.ndim-1
    denom = dm1*self.alpha**dm1*self.omega
    return 2*jnp.pi**(dm1/2.)/denom

  def vlr_r0(self):
    """Compute the r=0 term for the long-range (reciprocal-space) potential.

    This is the self-energy correction term for the reciprocal-space sum.

    Returns:
      float: r=0 contribution to long-range potential.
    """
    return 2*self.alpha/jnp.pi**0.5

  def constant(self, npart):
    """Compute the constant part of the Ewald energy.

    Includes self-energy corrections and neutralizing background terms that
    are independent of particle positions.

    Args:
      npart (int): Number of particles in the system.

    Returns:
      float: Constant energy contribution (self-energy + background).
    """
    vsr_k0 = self.vsr_k0()
    vlr_r0 = self.vlr_r0()
    ebg = -0.5*npart*(npart-1)*vsr_k0
    vconst = -0.5*npart*(vlr_r0+vsr_k0)+ebg
    return vconst

  def fvsr_r(self, r):
    """Short-range (real-space) pair potential.

    Uses the complementary error function to smoothly cut off the Coulomb
    interaction at short distances.

    Args:
      r (jnp.array): Pairwise distances. Shape (M,).

    Returns:
      jnp.array: Short-range potential values. Shape (M,).
    """
    return erfc(self.alpha*r)/r

  def sum_sr(self, pos, rvecs):
    """Compute the real-space (short-range) Ewald sum.

    Sums over all unique particle pairs and periodic lattice shifts using
    the short-range potential.

    Args:
      pos (jnp.array): Particle positions. Shape (N, ndim).
      rvecs (jnp.array): Real-space lattice vectors for periodic images.
          Shape (M_r, ndim).

    Returns:
      float: Real-space (short-range) energy contribution.
    """
    esr = 0.0
    npart = len(pos)
    if npart == 1:
      return esr
    idx = jnp.triu_indices(npart, k=1)
    shifts = rvecs
    esrl = []
    for shift in shifts:
      drij = (pos[:, None] - (pos+shift)[None, :])[idx]
      rij = jnp.linalg.norm(drij, axis=-1)
      esr1 = self.fvsr_r(rij).sum()
      esrl.append(esr1)
    esr = jnp.sum(jnp.array(esrl))
    return esr

  def fvlr_k2d(self, k):
    """Long-range (reciprocal-space) potential for 2D systems.

    Args:
      k (jnp.array): Magnitudes of reciprocal space vectors. Shape (M,).

    Returns:
      jnp.array: Long-range potential values for 2D. Shape (M,).
    """
    dm1 = self.ndim-1
    ak = k/(2*self.alpha)
    vk = 2*dm1*jnp.pi/k**dm1/self.omega
    return vk*erfc(ak)

  def fvlr_k3d(self, k):
    """Long-range (reciprocal-space) potential for 3D systems.

    Args:
      k (jnp.array): Magnitudes of reciprocal space vectors. Shape (M,).

    Returns:
      jnp.array: Long-range potential values for 3D. Shape (M,).
    """
    dm1 = self.ndim-1
    ak = k/(2*self.alpha)
    vk = 2*dm1*jnp.pi/k**dm1/self.omega
    return vk*jnp.exp(-ak**2)

  def sum_lr(self, pos, kvecs):
    """Compute the reciprocal-space (long-range) Ewald sum.

    Uses the structure factor to efficiently compute the long-range contribution
    in reciprocal space.

    Args:
      pos (jnp.array): Particle positions. Shape (N, ndim).
      kvecs (jnp.array): Reciprocal-space lattice vectors. Shape (M_k, ndim).

    Returns:
      float: Reciprocal-space (long-range) energy contribution.
    """
    kmags = jnp.linalg.norm(kvecs, axis=-1)
    vlr_k = self.fvlr_k2d(kmags) if self.ndim == 2 else self.fvlr_k3d(kmags)
    sk = sofk.structure_factor(kvecs, pos)
    elr = 0.5*jnp.dot(sk, vlr_k)
    return elr
