#reconstructing test cases
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from jewald import axes_pos, ewald, lattice
# from . import axes_pos, lattice, ewald
import matplotlib.pyplot as plt


def compute_energy_per_particle(axes, pos):
    area = axes_pos.volume(axes)
    rc, kc = lattice.compute_cutoffs(axes, 15.0)
    latidx = lattice.lattice_indices(axes, rc, kc)
    ew = ewald.Ewald(ewald.alpha(rc, kc), 2, area)
    rvecs, kvecs = lattice.transform_lattice(latidx, axes)
    total_energy = ew.sum(pos, rvecs, kvecs)
    return total_energy / len(pos)

###Square
ndim = 2 #ndim is the number of dimensions
nx = 2
axes = nx * np.eye(ndim)
pos = axes_pos.get_rvecs(axes, (nx,) * ndim)
esq = compute_energy_per_particle(axes, pos)




###Rectangle 
nx = 2
a = 3.0
b = 1.5
axes = np.array([
    [a, 0],
    [0, b]
])
pos = axes_pos.get_rvecs(axes, (nx, nx))
energy = compute_energy_per_particle(axes, pos)
print(energy)