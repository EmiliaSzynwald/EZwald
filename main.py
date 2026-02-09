#!/usr/bin/env python3
"""Demo: Ewald geometry optimization for a 2D Wigner crystal.

Minimizes the Coulomb energy of point charges in a 2D periodic cell with
respect to the cell angle (and optionally positions). Uses JAX for
differentiation and optax for optimization. Run with --verbose for
timing; --gpu to use GPU if available.
"""

import numpy as np
from time import time


def main():
  """Parse arguments, build Ewald setup, and run geometry optimization.

  Creates a 2D triangular lattice, builds Ewald + lattice, and minimizes
  total Coulomb energy w.r.t. cell angle (and positions if randomized).
  Saves axes and positions before/after to axes0.dat, pos0.dat, axes1.dat, pos1.dat.

  Returns:
    None.
  """
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument('--seed', type=int, default=52)
  parser.add_argument('--nx', '-n', type=int, default=3)
  parser.add_argument('--rckc', '-c', type=float, default=30.)
  parser.add_argument('--lr', type=float, default=0.2)
  parser.add_argument('--gradient_clip', '-clip', type=float, default=1.0)
  parser.add_argument('--decay_steps', '-ds', type=int, default=1000)
  parser.add_argument('--decay_rate', '-dr', type=float, default=0.5)
  parser.add_argument('--nstep', '-r', type=int, default=1000)
  parser.add_argument('--verbose', '-v', action='store_true')
  parser.add_argument('--gpu',action='store_true')
  args = parser.parse_args()
  nx = args.nx
  import os
  if not args.gpu:
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'
  import jax
  import jax.numpy as jnp
  from jewald import ewald, lattice, geometry as geo
  print(jax.devices())
  rng = np.random.default_rng(args.seed)

  ndim = 2
  axes = nx*np.eye(ndim)
  rc, kc = lattice.compute_cutoffs(axes, args.rckc)
  latidx = lattice.lattice_indices(axes, rc, kc)
  mesh = (nx,)*ndim
  latvec = axes / np.array(mesh)[:, None]
  pos = geo.gen_lattice(latvec, mesh, kspace=False)
  print('N=',len(pos))
  charge = -jnp.ones(len(pos))

  area = geo.calc_volume(axes)

  def make_cell(theta, area):
    """Build 2D lattice vectors from cell angle and fixed area (rhombus).

    Args:
      theta (float): Angle between lattice vectors (radians).
      area (float): Cell area (held constant).

    Returns:
      array: Lattice vectors in row-major (rows are a, b), shape (2, 2).
    """
    a = (area/jnp.sin(theta))**0.5
    axes = a*jnp.array([
      [1, 0],
      [jnp.cos(theta), jnp.sin(theta)],
    ])
    return axes

  ew = ewald.Ewald(ewald.alpha(rc, kc), ndim, area)

  @jax.jit
  def loss(params):
    theta, pos = params
    axes = make_cell(theta, area)
    rvecs, kvecs = lattice.transform_lattice(latidx, axes)
    return ew.sum(pos, charge, rvecs, kvecs)

  grad_fn = jax.jit(jax.value_and_grad(loss))

  # ideal triangular crystal
  theta = 120./180*np.pi
  axes = make_cell(theta, area)
  mesh = (nx,)*ndim
  latvec = axes / np.array(mesh)[:, None]
  pos = geo.gen_lattice(latvec, mesh, kspace=False)
  params = (theta, pos)

  if args.verbose:
    # timing tests
    tick = time()
    grad_fn(params)
    tock = time()
    dt = tock-tick
    print('before JIT', dt, 's per call')

    dt = 0
    nrun = 5
    for i in range(nrun):
      pos = rng.random(pos.shape)
      tick = time()
      params = (theta, pos)
      grad_fn(params)
      tock = time()
      dt += tock-tick
    print('after JIT', dt/nrun, 's per call')
    assert 0

  # geomtry optimization
  import optax
  schedule = optax.exponential_decay(
    args.lr,
    args.decay_steps,
    args.decay_rate,
  )
  optimizer = optax.chain(
    optax.clip(args.gradient_clip),
    optax.adam(schedule),
  )

  def fit(params, optimizer):
    """Run gradient-based optimization to minimize loss w.r.t. params.

    Args:
      params (tuple): Initial (theta, pos). Cell angle and particle positions.
      optimizer (optax.GradientTransformation): Optimizer (e.g. Adam).

    Returns:
      tuple: Optimized (theta, pos) after nstep steps.
    """
    @jax.jit
    def step(params, opt_state):
      loss_value, grads = grad_fn(params)
      updates, opt_state = optimizer.update(grads, opt_state, params)
      params = optax.apply_updates(params, updates)
      return params, opt_state, loss_value
    opt_state = optimizer.init(params)
    for i in range(args.nstep):
      params, opt_state, loss_value = step(params, opt_state)
      if i % 100 == 0:
        print(f'step {i}, loss: {loss_value}')
    return params

  ref = loss(params)
  print('expect: E = %f, theta = %d' % (ref, theta/np.pi*180))

  # initialize random particles in a square box
  theta0 = 90./180*np.pi
  axes0 = make_cell(theta0, area)
  fracs = rng.random(pos.shape)
  pos0 = np.dot(fracs, axes0)
  np.savetxt('axes0.dat', axes0)
  np.savetxt('pos0.dat', pos0)
  params0 = (theta0, pos0)

  params1 = fit(params0, optimizer)

  # save optimized geometry
  theta1, pos1 = params1
  axes1 = make_cell(theta1, area)
  np.savetxt('axes1.dat', axes1)
  np.savetxt('pos1.dat', geo.pos_in_axes(axes1, pos1))
  print('optimized theta = %.1f' % (theta1/np.pi*180))

if __name__ == '__main__':
  main()
