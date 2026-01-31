import jax.numpy as jnp

def calc_eikr(kvecs, pos): #Calculate the dot product of each k-vector with each particle position.
  kdotr = jnp.einsum('...i,ri->...r', kvecs, pos)
  eikr = jnp.exp(1j*kdotr) #Represents a plane wave for each k-vector and particle position.
  return eikr #Return the array of complex exponentials.

def structure_factor(kvecs, pos, charge):
  rhok = jnp.dot(calc_eikr(kvecs, pos), charge)
  sk = (rhok.conj()*rhok).real
  return sk #Return the real-valued structure factor for each k-vector.
