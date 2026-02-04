"""Structure factor and Fourier-space utilities for Ewald sums.

This module computes exp(i k·r) and the charge structure factor |rho(k)|^2
used in the reciprocal-space part of the Ewald summation.
"""

import jax.numpy as jnp


def calc_eikr(kvecs, pos):
  """Compute exp(i k·r) for each k-vector and particle position.

  Args:
    kvecs: Array of shape (..., ndim). Reciprocal-space vectors.
    pos: Array of shape (npart, ndim). Particle positions.

  Returns:
    Array of shape (..., npart). Complex exponentials exp(i k·r_j).
  """
  kdotr = jnp.einsum('...i,ri->...r', kvecs, pos)
  eikr = jnp.exp(1j*kdotr)
  return eikr


def structure_factor(kvecs, pos, charge):
  """Compute the real-valued structure factor |rho(k)|^2 for each k.

  rho(k) = sum_j charge_j * exp(-i k·r_j); structure factor S(k) = |rho(k)|^2.

  Args:
    kvecs: Array of shape (nk, ndim). Reciprocal-space vectors.
    pos: Array of shape (npart, ndim). Particle positions.
    charge: Array of shape (npart,). Particle charges.

  Returns:
    Array of shape (nk,). Real-valued structure factor for each k-vector.
  """
  rhok = jnp.dot(calc_eikr(kvecs, pos), charge)
  sk = (rhok.conj()*rhok).real
  return sk
