import jax.numpy as jnp
from jax.scipy.special import erfc
from jewald import axes_pos, lattice, sofk

def make_ewald(axes, rckc=30.0):
  rvecs, kvecs = lattice.make_lattice(axes, rckc)       #lattice vectors in real and reciprocal space
  rc, kc = lattice.compute_cutoffs(axes, rckc)          #real-space and reciprocal-space cutoff distances
  a = alpha(rc, kc)
  ndim = len(axes)                                      #number of dimensions
  omega = axes_pos.volume(axes)
  ew = Ewald(a, ndim, omega)                            #ewald object
  return ew, rvecs, kvecs

def alpha(rc, kc):
  return jnp.sqrt(kc/(2*rc))

class Ewald:
  def __init__(self, alpha, ndim, omega):
    self.alpha = alpha
    self.ndim = ndim
    self.omega = omega

  ### purpose in life
  def sum(self, pos, rvecs, kvecs):
    vconst = self.constant(len(pos))
    vsr = self.sum_sr(pos, rvecs) #real-space sum
    vlr = self.sum_lr(pos, kvecs) #reciprocal-space sum
    return vconst + vsr + vlr #total Ewald energy: constant + short-range + long-range

  ### neutralizing background
  def vsr_k0(self): #short range
    dm1 = self.ndim-1
    denom = dm1*self.alpha**dm1*self.omega
    return 2*jnp.pi**(dm1/2.)/denom

  def vlr_r0(self): #long range
    return 2*self.alpha/jnp.pi**0.5

  def constant(self, npart):
    vsr_k0 = self.vsr_k0()
    vlr_r0 = self.vlr_r0()
    ebg = -0.5*npart*(npart-1)*vsr_k0
    vconst = -0.5*npart*(vlr_r0+vsr_k0)+ebg
    return vconst

  ### direct-space sum
  def fvsr_r(self, r): #Short-range (real-space) pair potential
    return erfc(self.alpha*r)/r

  def sum_sr(self, pos, rvecs): #Compute the real-space (short-range) sum over all unique particle pairs and lattice shifts
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

  ### reciprocal-space sum
  def fvlr_k2d(self, k): #Long-range (reciprocal-space) potential for 2D
    dm1 = self.ndim-1
    ak = k/(2*self.alpha)
    vk = 2*dm1*jnp.pi/k**dm1/self.omega
    return vk*erfc(ak)

  def fvlr_k3d(self, k): #Long-range (reciprocal-space) potential for 3D
    dm1 = self.ndim-1
    ak = k/(2*self.alpha)
    vk = 2*dm1*jnp.pi/k**dm1/self.omega
    return vk*jnp.exp(-ak**2)

  def sum_lr(self, pos, kvecs): #Compute the reciprocal-space (long-range) sum using the structure factor
    kmags = jnp.linalg.norm(kvecs, axis=-1)
    vlr_k = self.fvlr_k2d(kmags) if self.ndim == 2 else self.fvlr_k3d(kmags)
    sk = sofk.structure_factor(kvecs, pos)
    elr = 0.5*jnp.dot(sk, vlr_k)
    return elr
