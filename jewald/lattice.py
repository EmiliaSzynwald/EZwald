 """Lattice generation utilities for Ewald summation.

This module provides functions for generating lattice vectors in both direct
and reciprocal space, computing cutoffs, and transforming between Miller
indices and coordinate representations.
"""
import jax.numpy as jnp
from jewald import axes_pos

def make_lattice(axes, rckc):
  """Create crystal lattice in direct and reciprocal spaces.

  Generates lattice vectors for both real-space and reciprocal-space Ewald
  summation based on the unit cell and cutoff parameters.

  Args:
    axes (jnp.array): Lattice vectors in row-major format. Shape (ndim, ndim).
    rckc (float): Product of real-space and reciprocal-space cutoffs.

  Returns:
    tuple: A tuple containing:
        - rvecs (jnp.array): Direct-space lattice vectors. Shape (M_r, ndim).
        - kvecs (jnp.array): Reciprocal-space lattice vectors. Shape (M_k, ndim).
  """
  rc, kc = compute_cutoffs(axes, rckc)
  latidx = lattice_indices(axes, rc, kc)
  rvecs, kvecs = transform_lattice(latidx, axes)
  return rvecs, kvecs

def compute_cutoffs(axes, rckc):
  """Compute periodic image cutoffs for Ewald summation.

  Determines appropriate real-space and reciprocal-space cutoffs based on
  the Wigner-Seitz cell radius and the product rckc.

  Args:
    axes (jnp.array): Lattice vectors in row-major format. Shape (ndim, ndim).
    rckc (float): Product of real-space and reciprocal-space cutoffs.

  Returns:
    tuple: A tuple containing:
        - rc (float): Real-space cutoff distance.
        - kc (float): Reciprocal-space cutoff magnitude.
  """
  rc = axes_pos.rwsc(axes)
  kc = rckc/rc
  return rc, kc

def lattice_indices(axes, rc, kc):
  """Generate Miller indices for real-space and reciprocal-space lattices.

  Creates integer lattice indices (Miller indices) for both direct and
  reciprocal space that fall within the specified cutoffs.

  Args:
    axes (jnp.array): Lattice vectors in row-major format. Shape (ndim, ndim).
    rc (float): Real-space cutoff distance.
    kc (float): Reciprocal-space cutoff magnitude.

  Returns:
    tuple: A tuple containing:
        - lvecs (jnp.array): Real-space lattice indices (Miller indices).
            Shape (M_r, ndim).
        - gvecs (jnp.array): Reciprocal-space lattice indices (Miller indices).
            Shape (M_k, ndim).
  """
  # real-space Miller indices
  rmax = rc + 2*rc  # box size is ~ 2*rc
  rvecs = axes_pos.get_ksphere(axes, rmax)
  lvecs = axes_pos.get_nvecs(axes, rvecs)
  # reciprocal-space Miller indices
  raxes = axes_pos.raxes(axes)
  kvecs = axes_pos.get_ksphere(raxes, kc)[1:]
  gvecs = axes_pos.get_nvecs(raxes, kvecs)
  return lvecs, gvecs

def transform_lattice(latidx, axes):
  """Transform integer Miller indices to coordinate vectors.

  Converts Miller indices (integer lattice coordinates) to actual coordinate
  vectors in both direct and reciprocal space.

  Args:
    latidx (tuple): A tuple containing:
        - lvecs (jnp.array): Real-space Miller indices. Shape (M_r, ndim).
        - gvecs (jnp.array): Reciprocal-space Miller indices. Shape (M_k, ndim).
    axes (jnp.array): Direct lattice vectors in row-major format.
        Shape (ndim, ndim).

  Returns:
    tuple: A tuple containing:
        - rvecs (jnp.array): Direct-space lattice vectors. Shape (M_r, ndim).
        - kvecs (jnp.array): Reciprocal-space lattice vectors. Shape (M_k, ndim).
  """
  lvecs, gvecs = latidx
  raxes = 2*jnp.pi*jnp.linalg.inv(axes).T
  rvecs = jnp.dot(lvecs, axes)
  kvecs = jnp.dot(gvecs, raxes)
  return rvecs, kvecs
