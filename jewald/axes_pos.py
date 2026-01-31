# use no jax in this file
import numpy as np

def rwsc(axes, dn=1):
  """ radius of the inscribed sphere inside the real-space
  Wigner-Seitz cell of the given cell

  Args:
    axes (np.array): lattice vectors in row-major
    dn (int,optional): number of image cells to search in each
     dimension, default dn=1 searches 26 images in 3D. Increasing `dn` increases accuracy but also computational cost.
  Returns:
    float: Wigner-Seitz cell radius. Approximated by finding the minimum distance to neighboring image cells and halving it.
  """
  ndim = len(axes)
  from itertools import product
  r2imgl  = []  # keep a list of distance^2 to all neighboring images
  images = product(range(-dn, dn+1), repeat=ndim)
  for ushift in images:
    if sum(ushift) == 0:
      continue  # ignore self
    shift = np.dot(ushift, axes)
    r2imgl.append(np.dot(shift, shift))
  rimg = np.sqrt(min(r2imgl))
  return rimg/2.

def cubic_pos(spaces):
  """
    Generates a grid of points based on the provided spaces for each dimension.

    Args:
        spaces (list of np.array): A list of numpy arrays, where each array represents the coordinates 
                                    along a specific dimension.  For example, if ndim=3, spaces might be
                                    [np.array([0,1,2]), np.array([0,1]), np.array([0,1,2,3])].

    Returns:
        np.array: A numpy array of shape (N, ndim), where N is the total number of grid points and
                  ndim is the number of dimensions. Each row represents the coordinates of a grid point.
                  The grid is generated using `np.meshgrid` with indexing set to 'ij'.
    """
  ndim = len(spaces)
  gvecs = np.stack(
    np.meshgrid(*spaces, indexing='ij'), axis=-1
  ).reshape(-1, ndim)
  return gvecs

def get_rvecs(axes, mesh):
  """Regular grid in positive quadrant
  Args:
        axes (np.array): Lattice vectors in row-major format.
        mesh (tuple or list): Number of grid points along each lattice vector.
                             For example, (nx, ny, nz) specifies the grid density.

    Returns:
        np.array: A numpy array containing the real-space vectors of the grid points.
    """
  spaces = [np.arange(nx) for nx in mesh]
  gvecs = cubic_pos(spaces)
  fracs = axes/np.array(mesh)[:, np.newaxis]  # axes is row-major
  return np.dot(gvecs, fracs)

def get_kvecs(raxes, mesh):
  """Regular grid centered around 0
    Args:
        raxes (np.array): Reciprocal lattice vectors in row-major format.
        mesh (tuple or list): Number of grid points along each reciprocal lattice vector.

    Returns:
        np.array: A numpy array containing the k-space vectors of the grid points.
                   Uses `np.fft.fftfreq` to generate frequencies centered around zero,
                   suitable for FFT-based calculations.
    """
  spaces = [np.fft.fftfreq(nx)*nx for nx in mesh]
  gvecs = cubic_pos(spaces)
  return np.dot(gvecs, raxes)

def get_ksphere(raxes, kc, margin=0.2, twist=None):
  """
    Generates a set of k-vectors within a sphere of radius kc in reciprocal space.

    Args:
        raxes (np.array): Reciprocal lattice vectors in row-major format.
        kc (float): Radius of the k-sphere.
        margin (float, optional): Margin to extend the k-sphere radius when determining the mesh size. Defaults to 0.2.
        twist (np.array, optional): A twist vector to shift the k-sphere center. Defaults to None (center at the origin).

    Returns:
        tuple: A tuple containing:
            - kvecs (np.array): k-vectors within the k-sphere.
            - ksel (np.array): A boolean array indicating which k-vectors are inside the sphere.
    """
  from itertools import product
  ndim = len(raxes)
  qvec = np.zeros(ndim)
  if twist is not None:
    qvec = np.dot(twist, raxes)
  nmax = 0
  for direction in product(range(-1, 1+1), repeat=ndim):
    vec = np.dot(direction, raxes)
    vmag = np.linalg.norm(vec)
    if vmag < 1e-8:
      continue
    n1 = int(np.ceil((1+margin)*kc/vmag))
    nmax = max(nmax, n1)
  mesh = (2*nmax,)*ndim
  kvecs = get_kvecs(raxes, mesh)+qvec
  kmags = np.linalg.norm(kvecs, axis=-1)
  ksel = kmags<kc
  return kvecs[ksel]

def get_nvecs(axes, pos, atol=1e-10):
  """ find integer vectors of lattice positions from unit cell

  Args:
    axes (np.array): lattice vectors in row-major
    pos (np.array): lattice sites
  Return:
    np.array: nvecs, integer vectors that label the lattice sites
  Example:
    >>> nvecs = get_nvecs(axes, pos)
  """
  ncands = np.dot(pos, np.linalg.inv(axes))  # candidates
  nvecs = np.around(ncands).astype(int)  # convert to integer
  success = np.allclose(np.dot(nvecs, axes), pos, atol=atol)  # check
  if not success:
    raise RuntimeError('problem in get_nvecs')
  return nvecs

def pos_in_axes(axes, pos, ztol=1e-10):
  """ particle position(s) in cell

  Args:
    axes (np.array): crystal lattice vectors
    pos (np.array): particle position(s)
  Returns:
    pos0(np.array): particle position(s) inside the cell
  """
  upos = np.dot(pos, np.linalg.inv(axes))
  zsel = abs(upos % 1-1) < ztol
  upos[zsel] = 0
  pos0 = np.dot(upos % 1, axes)
  return pos0
