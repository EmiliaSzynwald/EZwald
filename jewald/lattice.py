import jax.numpy as jnp
from jewald import axes_pos, geometry

def make_lattice(axes, rckc):
  """Create crystal lattice in direct and reciprocal spaces.

  Args:
    axes (np.array): lattice vectors in row-major
    rckc (float): rcut*kcut
  Return:
    np.array: rvecs, direct-space lattice
    np.array: kvecs, reciprocal-space lattice
  """
  rc, kc = compute_cutoffs(axes, rckc)
  latidx = lattice_indices(axes, rc, kc)
  rvecs, kvecs = transform_lattice(latidx, axes)
  return rvecs, kvecs

def compute_cutoffs(axes, rckc):
  """Compute periodic image cutoffs.

  Args:
    axes (np.array): lattice vectors in row-major
    rckc (float): rcut*kcut
  Return:
    float: rcut, real-space cutoff
    float: kcut, reciprocal-space cutoff
  """
  rc = geometry.calc_rwsc(axes)
  kc = rckc/rc
  return rc, kc

def lattice_indices(axes, rc, kc):
  """Initialize Miller indices.

  Args:
    axes (np.array): lattice vectors in row-major
    rc (float): real-space cutoff
    kc (float): reciprocal-space cutoff
  Return:
    np.array: lvecs, real-space lattice indices
    np.array: gvecs, reciprocal-space lattice indices
  """
  # real-space Miller indices
  rmax = rc + 2*rc  # box size is ~ 2*rc
  rvecs = axes_pos.get_ksphere(axes, rmax)
  lvecs = axes_pos.get_nvecs(axes, rvecs)
  # reciprocal-space Miller indices
  raxes = geometry.calc_recvec(axes)
  kvecs = axes_pos.get_ksphere(raxes, kc)[1:]
  gvecs = axes_pos.get_nvecs(raxes, kvecs)
  return lvecs, gvecs

def transform_lattice(latidx, axes):
  """Transform integer indices to coordinates

  Args:
    latidx (tuple): lattice indices (lvecs, gvecs)
    axes (jnp.array): lattice vectors in row-major
  Return:
    jnp.array: rvecs, direct-space lattice
    jnp.array: kvecs, reciprocal-space lattice
  """
  lvecs, gvecs = latidx
  raxes = 2*jnp.pi*jnp.linalg.inv(axes).T
  rvecs = jnp.dot(lvecs, axes)
  kvecs = jnp.dot(gvecs, raxes)
  return rvecs, kvecs
