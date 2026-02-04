"""Bilayer (slab) Ewald tests: staggered square, rectangular, and hexagonal cells.

Regression and tiling tests for EwaldSumSlab: energy per particle vs. reference
values for different layer separations and cell types.
"""

import numpy as np
import pytest
import os

import jax.numpy as jnp
import sys
from jewald import bilayer_sum
from jewald import geometry as geo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

test_cases_staggered_square = [
    (0.2404040404040404, -1.575227519156419),
    (0.33939393939393936, -1.497248680039713),
    (0.4808080808080808, -1.433649116760696),
    (0.6787878787878787, -1.396372425213264),
]
@pytest.mark.parametrize("d, expected", test_cases_staggered_square)
def test_staggered_square(d, expected):
    """Test Ewald sum energy per particle for staggered square bilayer supercells.

    This test verifies that the slab Ewald sum computed for a staggered square
    bilayer (two layers with in-plane square lattice) matches the reference
    rescaled energy E/sqrt(n) for the given interlayer separation.

    Args:
        d (float): Interlayer separation.
        expected (float): Reference rescaled energy (E/sqrt(n)) for comparison.

    The function compares the computed result to the reference value within an absolute
    tolerance of 1e-3.
    """
    alat = 1
    cell = 2 * np.array([[alat, 0], [0, alat]])
    disp=np.array([[0,0],[0.5, 0.5]]) /2
    x0 = np.array([[0,0],[0.5,0],[0,0.5],[0.5,0.5]])
    x = np.concatenate([_disp + x0 for _disp in disp])
    pos = x@cell
    charge = -1*jnp.ones(len(pos))
    bew = bilayer_sum.EwaldSumSlab(cell, d)
    E = bew.energy(charge, pos)/len(pos)
    n = (len(pos))/abs(np.linalg.det(cell))
    energy_rescaling = E/(np.sqrt(n))
    assert np.isclose(energy_rescaling, expected, atol=1e-3)

@pytest.mark.parametrize("d, expected", test_cases_staggered_square)
def test_tile_staggered_square(d, expected, nx=2):
    """Test that staggered square bilayer energy is invariant under nx x nx tiling.

    This test verifies that the slab Ewald sum for a tiled staggered square
    bilayer (nx x nx supercell) gives the same rescaled energy per particle
    as the unit-cell regression test for the same interlayer separation.

    Args:
        d (float): Interlayer separation.
        expected (float): Reference rescaled energy (E/sqrt(n)) for comparison.
        nx (int): Tiling factor along x and y directions (default is 2).

    The function compares the computed result to the reference value within an absolute
    tolerance of 1e-3.
    """
    alat = 1
    reg_cell = 2 * np.array([[alat, 0], [0, alat]])
    disp=np.array([[0,0],[0.5, 0.5]]) /2
    x0 = np.array([[0,0],[0.5,0],[0,0.5],[0.5,0.5]])
    x = np.concatenate([_disp + x0 for _disp in disp])
    reg_pos = x@reg_cell

    n_up = int(len(reg_pos)/2)
    n_down = int(len(reg_pos) - n_up)

    reg_pos_t, reg_pos_b = np.split(reg_pos, [n_up])

    cell = np.diag( (nx, nx) ) @ reg_cell
    pos_t = geo.tile(reg_pos_t, (nx, nx), reg_cell) #pos of nx x nx supercell
    pos_b = geo.tile(reg_pos_b, (nx, nx), reg_cell)

    pos = np.concatenate([pos_t, pos_b], axis=0)

    charge = -1*jnp.ones(len(pos))
    bew = bilayer_sum.EwaldSumSlab(cell, d)
    E = bew.energy(charge, pos)/len(pos)
    n = (len(pos))/abs(np.linalg.det(cell))
    energy_rescaling = E/(np.sqrt(n))
    assert np.isclose(energy_rescaling, expected, atol=1e-3)

test_cases_rectangular = [
    (0.0, -1.9605157893210492),
    (0.20555555555555557, -1.6755085482438778),
    (0.3550505050505051, -1.5389573331071371),
    (0.5045454545454546, -1.4487808635472104),
]
@pytest.mark.parametrize("d, expected", test_cases_rectangular)
def test_rectangular(d, expected):
    """Test Ewald sum energy per particle for rectangular bilayer supercells.

    This test verifies that the slab Ewald sum computed for a rectangular
    bilayer (in-plane rectangular lattice) matches the reference rescaled
    energy E/sqrt(n) for the given interlayer separation.

    Args:
        d (float): Interlayer separation.
        expected (float): Reference rescaled energy (E/sqrt(n)) for comparison.

    The function compares the computed result to the reference value within an absolute
    tolerance of 1e-3.
    """
    alat = 1
    cell = 2 * np.array([
        [alat, 0],
        [0, np.sqrt(3) * alat]
    ])
    disp = np.array([[0,0],[0.5, 0.5]]) /2
    x0 = np.array([[0,0],[0.5,0],[0,0.5],[0.5,0.5]])
    x = np.concatenate([_disp + x0 for _disp in disp])
    pos = x@cell
    charge = -1*jnp.ones(len(pos)) #This is e
    bew = bilayer_sum.EwaldSumSlab(cell, d)
    E = bew.energy(charge, pos)/len(pos)
    n = (len(pos))/abs(np.linalg.det(cell))
    energy_rescaling = E/(np.sqrt(n))
    assert np.isclose(energy_rescaling, expected, atol=1e-3)

@pytest.mark.parametrize("d, expected", test_cases_rectangular)
def test_tile_rectangular(d, expected, nx=2):
    """Test that rectangular bilayer energy is invariant under nx x nx tiling.

    This test verifies that the slab Ewald sum for a tiled rectangular
    bilayer (nx x nx supercell) gives the same rescaled energy per particle
    as the unit-cell regression test for the same interlayer separation.

    Args:
        d (float): Interlayer separation.
        expected (float): Reference rescaled energy (E/sqrt(n)) for comparison.
        nx (int): Tiling factor along x and y directions (default is 2).

    The function compares the computed result to the reference value within an absolute
    tolerance of 1e-3.
    """
    alat = 1
    reg_cell = 2 * np.array([
        [alat, 0],
        [0, np.sqrt(3) * alat]
    ])
    disp = np.array([[0,0],[0.5, 0.5]]) /2
    x0 = np.array([[0,0],[0.5,0],[0,0.5],[0.5,0.5]])
    x = np.concatenate([_disp + x0 for _disp in disp])
    reg_pos = x@reg_cell

    n_up = int(len(reg_pos)/2)
    n_down = int(len(reg_pos) - n_up)

    reg_pos_t, reg_pos_b = np.split(reg_pos, [n_up])

    cell = np.diag( (nx, nx) ) @ reg_cell
    pos_t = geo.tile(reg_pos_t, (nx, nx), reg_cell) #pos of nx x nx supercell
    pos_b = geo.tile(reg_pos_b, (nx, nx), reg_cell)

    pos = np.concatenate([pos_t, pos_b], axis=0)

    charge = -1*jnp.ones(len(pos))
    bew = bilayer_sum.EwaldSumSlab(cell, d, n_up, n_down)
    E = bew.energy(charge, pos, n_up, n_down)/len(pos)
    n = (len(pos))/abs(np.linalg.det(cell))
    energy_rescaling = E/(np.sqrt(n))
    assert np.isclose(energy_rescaling, expected, atol=1e-3)

test_cases_staggered_hexagonal = [
    (0.6060606060606061, -1.469355465180093),
    (0.7575757575757576, -1.435466086946966),
    (0.9090909090909091, -1.41493610327513),
    (1.0606060606060606, -1.4027859902277606),
]
@pytest.mark.parametrize("d, expected", test_cases_staggered_hexagonal)
def test_staggered_hexagonal(d, expected):
    """Test Ewald sum energy per particle for staggered hexagonal bilayer supercells.

    This test verifies that the slab Ewald sum computed for a staggered
    hexagonal bilayer (in-plane hexagonal lattice) matches the reference
    rescaled energy E/sqrt(n) for the given interlayer separation.

    Args:
        d (float): Interlayer separation.
        expected (float): Reference rescaled energy (E/sqrt(n)) for comparison.

    The function compares the computed result to the reference value within an absolute
    tolerance of 1e-3.
    """
    alat = np.sqrt(2*np.pi/np.sqrt(3))
    cell = 2 * np.array([[alat,0],[-0.5*alat,np.sqrt(3)/2*alat]])
    disp=np.array([[0,0],[2./3,1./3]]) /2
    x0 = np.array([[0,0],[0.5,0],[0,0.5],[0.5,0.5]])
    x = np.concatenate([_disp + x0 for _disp in disp])
    pos = x@cell
    charge = -1*jnp.ones(len(pos)) #This is e
    bew = bilayer_sum.EwaldSumSlab(cell, d)
    E = bew.energy(charge, pos)/len(pos)
    n = (len(pos))/abs(np.linalg.det(cell))
    energy_rescaling = E/(np.sqrt(n))
    assert np.isclose(energy_rescaling, expected, atol=1e-3)

@pytest.mark.parametrize("d, expected", test_cases_staggered_hexagonal)
def test_tile_staggered_hexagonal(d, expected, nx=2):
    """Test that staggered hexagonal bilayer energy is invariant under nx x nx tiling.

    This test verifies that the slab Ewald sum for a tiled staggered hexagonal
    bilayer (nx x nx supercell) gives the same rescaled energy per particle
    as the unit-cell regression test for the same interlayer separation.

    Args:
        d (float): Interlayer separation.
        expected (float): Reference rescaled energy (E/sqrt(n)) for comparison.
        nx (int): Tiling factor along x and y directions (default is 2).

    The function compares the computed result to the reference value within an absolute
    tolerance of 1e-3.
    """
    alat = np.sqrt(2*np.pi/np.sqrt(3))
    reg_cell = 2 * np.array([[alat,0],[-0.5*alat,np.sqrt(3)/2*alat]])
    disp=np.array([[0,0],[2./3,1./3]]) /2
    x0 = np.array([[0,0],[0.5,0],[0,0.5],[0.5,0.5]])
    x = np.concatenate([_disp + x0 for _disp in disp])
    reg_pos = x@reg_cell

    n_up = int(len(reg_pos)/2)
    n_down = int(len(reg_pos) - n_up)

    reg_pos_t, reg_pos_b = np.split(reg_pos, [n_up])

    cell = np.diag( (nx, nx) ) @ reg_cell
    pos_t = geo.tile(reg_pos_t, (nx, nx), reg_cell) #pos of nx x nx supercell
    pos_b = geo.tile(reg_pos_b, (nx, nx), reg_cell)

    pos = np.concatenate([pos_t, pos_b], axis=0)

    charge = -1*jnp.ones(len(pos))
    bew = bilayer_sum.EwaldSumSlab(cell, d, n_up, n_down)
    E = bew.energy(charge, pos, n_up, n_down)/len(pos)
    n = (len(pos))/abs(np.linalg.det(cell))
    energy_rescaling = E/(np.sqrt(n))
    assert np.isclose(energy_rescaling, expected, atol=1e-3)