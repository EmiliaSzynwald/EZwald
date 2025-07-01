import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# import jewald
from jewald import axes_pos, ewald, lattice
# pytest -v .

def compute_energy_per_particle(axes, pos):
    area = axes_pos.volume(axes)
    rc, kc = lattice.compute_cutoffs(axes, 15.0)
    latidx = lattice.lattice_indices(axes, rc, kc)
    ew = ewald.Ewald(ewald.alpha(rc, kc), 2, area)
    rvecs, kvecs = lattice.transform_lattice(latidx, axes)
    total_energy = ew.sum(pos, rvecs, kvecs)
    return total_energy / len(pos)

def test_square():
    ndim = 2
    nx = 2
    rckc = 10.6567587
    axes = nx * np.eye(ndim)
    pos = axes_pos.get_rvecs(axes, (nx,) * ndim)
    esq = compute_energy_per_particle(axes, pos)
    assert np.isclose(-1.9501325, esq, atol=1e-5) #tolerance of minimum of 6, preferred of 7 or more


def test_rectangle():
    nx = 2
    a = 3.0
    b = 1.5
    axes = np.array([
        [a, 0],
        [0, b]
    ])
    pos = axes_pos.get_rvecs(axes, (nx, nx))
    expected = -1.68092485 #expected value is different if its monolayer vs bilayer??
    energy = compute_energy_per_particle(axes, pos)
    assert np.isclose(energy, expected, atol=1e-6) 

def test_oblique():
    nx = 2
    a = 4.5
    b = 8.0
    theta_deg = 70
    theta = np.deg2rad(theta_deg)
    axes = np.array([
        [a, 0],
        [b * np.cos(theta), b * np.sin(theta)]
    ])
    pos = axes_pos.get_rvecs(axes, (nx, nx))
    expected = -0.64032870
    energy = compute_energy_per_particle(axes, pos)
    assert np.isclose(energy, expected, atol=1e-6)

def test_hexagon():
    nx = 2
    a = 3.0
    axes = np.array([
        [a, 0],
        [a/2, a * np.sqrt(3)/2]
    ])
    pos = axes_pos.get_rvecs(axes, (nx, nx))
    expected = -1.40447531
    energy = compute_energy_per_particle(axes, pos)
    assert np.isclose(energy, expected, atol=1e-6)

def test_triangle(): #hexagon and triangle are the same 
    nx =3
    axes = np.array([[1, 0], [-0.5, (np.sqrt(3)) / 2]]) * 3  # Stretches the lattice by factor of 3
    pos = np.array([[0, 0]])
    expected = -0.7022378
    energy = compute_energy_per_particle(axes, pos)
    assert np.isclose(energy, expected, atol=1e-6) #should i increase tolerance here to

def test_honeycomb(): #2 overlapping triangles
    a =1
    axes = np.array([[1,0],[-0.5,np.sqrt(3)/2]])
    pos0 = np.array([[0, 0], [0, 1/np.sqrt(3)]])
    expected = -2.0349255
    energy = compute_energy_per_particle(axes, pos0)
    assert np.isclose(energy, expected, atol=1e-6) #big issue
