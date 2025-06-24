import jax.numpy as jnp

def calc_eikr(kvecs, pos):
  kdotr = jnp.einsum('...i,ri->...r', kvecs, pos)
  eikr = jnp.exp(1j*kdotr)
  return eikr

def structure_factor(kvecs, pos):
  rhok = calc_eikr(kvecs, pos).sum(axis=-1)
  sk = (rhok.conj()*rhok).real
  return sk
