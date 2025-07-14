import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# import jewald
from jewald import axes_pos, ewald, lattice
# pytest -v .

def test_square():
    #calculating from functions
    nx = 2 
    axes = nx * np.eye(2) # 2x2 scaled by nx
    pos = axes_pos.get_rvecs(axes, (nx, nx))
    ew, rvecs, kvecs = ewald.make_ewald(axes)
    square_sum = ew.sum(pos, rvecs, kvecs) / len(pos)

    #expected from paper [Some static and dynamical properties of a two-dimensional Wigner crystal]
    ac = nx**2
    sumsq = -3.900265/ac**0.5

    assert np.isclose(square_sum, sumsq, atol=1e-5)


def test_PrimitiveRectangle():
    #calculating from functions
    lam = 0.95
    a2 = 1
    a1 = lam*a2
    pos_PR = np.array([
        [0, 0],
    ])
    axes_PR = np.diag([a1, a2])
    ew, rvecs, kvecs = ewald.make_ewald(axes_PR)
    sum_PR = ew.sum(pos_PR, rvecs, kvecs)

    #expected from paper [Some static and dynamical properties of a two-dimensional Wigner crystal]
    ac = a1*a2
    PRsum = -3.898597/2/ac**0.5

    assert np.isclose(sum_PR, PRsum, atol=1e-6) 

def test_CenteredRectangle():
    #calculating from functions
    lam = 0.95
    a2 = 1
    a1 = lam*a2
    pos_CR = np.array([
        [0, 0],
    ])
    axes_CR = np.array([
        [a1,    0],
        [a1/2,  a2/2]
    ])
    ew, rvecs, kvecs = ewald.make_ewald(axes_CR)
    sum_CR = ew.sum(pos_CR, rvecs, kvecs)

    #expected from paper [Some static and dynamical properties of a two-dimensional Wigner crystal]
    ac = 0.5*a1*a2
    CRsum =  -3.900647/2/ac**0.5

    assert np.isclose(sum_CR, CRsum, atol=1e-6) 

# def test_oblique(): #need to adjust
#     nx = 2
#     a = 4.5
#     b = 8.0
#     theta_deg = 70
#     theta = np.deg2rad(theta_deg)
#     axes = np.array([
#         [a, 0],
#         [b * np.cos(theta), b * np.sin(theta)]
#     ])
#     pos = axes_pos.get_rvecs(axes, (nx, nx))
#     expected = -0.64032870
#     energy = compute_energy_per_particle(axes, pos)
#     assert np.isclose(energy, expected, atol=1e-6)

def test_hexagon(): #also known as triangular
    #calculating from functions
    nx = 3
    axes_hex = np.array([[1, 0], [-0.5, (np.sqrt(3)) / 2]]) * 3  # Stretches the lattice by factor of 3
    pos_hex = np.array([[0, 0]])
    ew_hex, rvecs_hex, kvecs_hex = ewald.make_ewald(axes_hex)
    hex_sum = ew_hex.sum(pos_hex, rvecs_hex, kvecs_hex) / len(pos_hex)

    #expected from paper [Some static and dynamical properties of a two-dimensional Wigner crystal]
    ac = ((np.sqrt(3)) / 2) * (nx**2)
    sumhex =  -3.921034/2/ac**0.5

    assert np.isclose(hex_sum, sumhex, atol=1e-6)

# def test_honeycomb(): #2 overlapping triangles
#     a =1
#     axes = np.array([[1,0],[-0.5,np.sqrt(3)/2]])
#     pos0 = np.array([[0, 0], [0, 1/np.sqrt(3)]])
#     expected = -1.51865 #previous issue
#     energy = compute_energy_per_particle(axes, pos0)
#     assert np.isclose(energy, expected, atol=1e-6)
