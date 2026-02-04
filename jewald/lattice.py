"""Lattice generation for Ewald summation.

This module builds real-space and reciprocal-space lattice vectors
from cell axes and cutoffs, used by the Ewald module to perform
the real- and reciprocal-space sums.
"""

import jax.numpy as jnp
from jewald import geometry as geo


def make_lattice(axes, rckc):
  """Create crystal lattice in direct and reciprocal spaces.

  Args:
    axes: Array of shape (ndim, ndim). Lattice vectors in row-major form.
    rckc: float. Product of real-space and reciprocal-space cutoffs.

  Returns:
    rvecs: Array. Direct-space lattice displacement vectors.
    kvecs: Array. Reciprocal-space vectors (excluding k=0).
  """
  rc, kc = compute_cutoffs(axes, rckc)
  latidx = lattice_indices(axes, rc, kc)
  rvecs, kvecs = transform_lattice(latidx, axes)
  return rvecs, kvecs

def compute_cutoffs(axes, rckc):
  """Compute real- and reciprocal-space cutoffs from rckc and cell.

  Args:
    axes: Array of shape (ndim, ndim). Lattice vectors in row-major form.
    rckc: float. Product rcut * kcut (controls Ewald convergence).

  Returns:
    rc: float. Real-space cutoff (inscribing radius of Wigner-Seitz cell).
    kc: float. Reciprocal-space cutoff magnitude.
  """
  rc = geo.calc_rwsc(axes)
  kc = rckc/rc
  return rc, kc

def lattice_indices(axes, rc, kc):
  """Build integer lattice indices for real- and reciprocal-space sums.

  Args:
    axes: Array of shape (ndim, ndim). Lattice vectors in row-major form.
    rc: float. Real-space cutoff.
    kc: float. Reciprocal-space cutoff magnitude.

  Returns:
    lvecs: Array. Integer indices for direct-space lattice points.
    gvecs: Array. Integer indices for reciprocal-space vectors (excluding 0).
  """
  # real-space Miller indices
  rmax = rc + 2*rc  # box size is ~ 2*rc
  rvecs = geo.get_ksphere(axes, rmax)
  lvecs = geo.get_nvecs(axes, rvecs)
  # reciprocal-space Miller indices
  raxes = geo.calc_recvec(axes)
  kvecs = geo.get_ksphere(raxes, kc)[1:]
  gvecs = geo.get_nvecs(raxes, kvecs)
  return lvecs, gvecs

def transform_lattice(latidx, axes):
  """Transform integer lattice indices to Cartesian coordinates.

  Args:
    latidx: Tuple (lvecs, gvecs). Integer Miller indices for real and
        reciprocal space.
    axes: Array of shape (ndim, ndim). Lattice vectors in row-major form.

  Returns:
    rvecs: Array. Direct-space lattice vectors in Cartesian coordinates.
    kvecs: Array. Reciprocal-space vectors in Cartesian coordinates.
  """
  lvecs, gvecs = latidx
  raxes = geo.calc_recvec(axes)
  rvecs = jnp.dot(lvecs, axes)
  kvecs = jnp.dot(gvecs, raxes)
  return rvecs, kvecs
