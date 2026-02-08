"""Structure factor and Fourier-space utilities for Ewald sums.

This module computes exp(i k·r) and the charge structure factor |rho(k)|^2
used in the reciprocal-space part of the Ewald summation.
"""

import jax.numpy as jnp


def calc_eikr(kvecs, pos):
  """Compute the dot product (exp(i k·r)) of each k-vector with each particle position

  Args:
    kvecs (array): Reciprocal-space vectors, shape (..., ndim).
    pos (array): Particle positions, shape (npart, ndim).

  Returns:
    array: Complex exponentials exp(i k·r_j) which represents a plane wave for each k-vector and particle position, 
      shape (..., npart).
  """
  kdotr = jnp.einsum('...i,ri->...r', kvecs, pos)
  eikr = jnp.exp(1j*kdotr)
  return eikr


def structure_factor(kvecs, pos, charge):
  """Compute the real-valued structure factor |rho(k)|^2 for each k.

  rho(k) = sum_j charge_j * exp(-i k·r_j); structure factor S(k) = |rho(k)|^2.

  Args:
    kvecs (array): Reciprocal-space vectors, shape (nk, ndim).
    pos (array): Particle positions, shape (npart, ndim).
    charge (array): Particle charges, shape (npart,).

  Returns:
    array: Real-valued structure factor for each k-vector, shape (nk,).
  """
  rhok = jnp.dot(calc_eikr(kvecs, pos), charge)
  sk = (rhok.conj()*rhok).real
  return sk
