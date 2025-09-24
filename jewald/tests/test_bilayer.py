import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from jewald import bilayer_sum

test_cases_staggered_square = [
    (0.2404040404040404, -1.575227519156419),
    (0.33939393939393936, -1.497248680039713),
    (0.4808080808080808, -1.433649116760696),
    (0.6787878787878787, -1.396372425213264),
]
@pytest.mark.parametrize("d, expected", test_cases_staggered_square)
def test_staggered_square(d, expected):
    #Regression test
    alat = 1
    cell = 2 * np.array([[alat, 0], [0, alat]])
    disp=np.array([[0,0],[0.5, 0.5]]) /2
    x0 = np.array([[0,0],[0.5,0],[0,0.5],[0.5,0.5]])
    x = np.concatenate([_disp + x0 for _disp in disp])
    pos = x@cell
    charge = -1*jnp.ones(len(pos))
    bew = EwaldSumSlab(cell, d)
    E = bew.energy(charge, pos)/len(pos)
    n = (len(pos))/abs(np.linalg.det(cell))
    rs = 1 / (np.sqrt(np.pi*n))
    eta = d/ (np.sqrt(2*np.pi* (rs**2)))
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
    #Regression test
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
    bew = EwaldSumSlab(cell, d)
    E = bew.energy(charge, pos)/len(pos)
    n = (len(pos))/abs(np.linalg.det(cell))
    rs = 1 / (np.sqrt(np.pi*n))
    eta = d/ (np.sqrt(2*np.pi* (rs**2)))
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
    #Regression test
    alat = np.sqrt(2*np.pi/np.sqrt(3))
    cell = 2 * np.array([[alat,0],[-0.5*alat,np.sqrt(3)/2*alat]])
    disp=np.array([[0,0],[2./3,1./3]]) /2
    x0 = np.array([[0,0],[0.5,0],[0,0.5],[0.5,0.5]])
    x = np.concatenate([_disp + x0 for _disp in disp])
    pos = x@cell
    charge = -1*jnp.ones(len(pos)) #This is e
    bew = EwaldSumSlab(cell, d)
    E = bew.energy(charge, pos)/len(pos)
    n = (len(pos))/abs(np.linalg.det(cell))
    rs = 1 / (np.sqrt(np.pi*n))
    eta = d/ (np.sqrt(2*np.pi* (rs**2)))
    energy_rescaling = E/(np.sqrt(n))

    assert np.isclose(energy_rescaling, expected, atol=1e-3)




