"""Structure factor calculations for Ewald summation.

This module provides functions for computing structure factors, which are
essential for efficient reciprocal-space Ewald summation.
"""
import jax.numpy as jnp

def calc_eikr(kvecs, pos):
  """Calculate exp(i*k*r) for all k-vectors and positions.

  Computes the dot product of each k-vector with each particle position
  and returns the complex exponentials exp(i*k*r), which represent plane
  waves for each k-vector and particle position.

  Args:
    kvecs (jnp.array): Reciprocal space vectors. Shape (M_k, ndim).
    pos (jnp.array): Particle positions. Shape (N, ndim).

  Returns:
    jnp.array: Complex exponentials exp(i*k*r). Shape (M_k, N) or
        (..., M_k, N) depending on input shapes.
  """
  kdotr = jnp.einsum('...i,ri->...r', kvecs, pos)
  eikr = jnp.exp(1j*kdotr) #Represents a plane wave for each k-vector and particle position.
  return eikr #Return the array of complex exponentials.

def structure_factor(kvecs, pos):
  """Compute the structure factor for given k-vectors and positions.

  Calculates the Fourier component of the density at each k-vector. The
  structure factor is the square magnitude of the Fourier transform of
  the particle density.

  Args:
    kvecs (jnp.array): Reciprocal space vectors. Shape (M_k, ndim).
    pos (jnp.array): Particle positions. Shape (N, ndim).

  Returns:
    jnp.array: Real-valued structure factor for each k-vector. Shape (M_k,).
  """
  rhok = calc_eikr(kvecs, pos).sum(axis=-1)
  sk = (rhok.conj()*rhok).real
  return sk #Return the real-valued structure factor for each k-vector.
