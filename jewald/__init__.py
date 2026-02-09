"""JAX-based Ewald summation for periodic Coulomb interactions.

This package provides Ewald summation implementations for 2D and 3D periodic
systems, including monolayer (bulk) and bilayer (slab) geometries. All
computations are JAX-differentiable for use in optimization and machine
learning workflows.
"""

import jax
jax.config.update("jax_enable_x64", True)
