import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from jewald import axes_pos, ewald, lattice

def test_Square(nx=2):
    """Test Ewald sum energy per particle for nx x nx tiled square supercells.

    This test verifies that the Ewald sum computed for a nx x nx square lattice
    matches the analytical result reported in:
    'Some static and dynamical properties of a two-dimensional Wigner crystal'.

    Args:
        nx (int): Tiling factor along x and y directions (default is 2).
    
    The function compares the computed result to the reference value within an absolute
    tolerance of 1e-5.
    """
    # Primitive cell: square lattice
    axes_primitive = np.eye(2)

    # Supercell axes (scaled)
    axes = nx * axes_primitive

    # Generate grid of lattice sites for supercell
    pos = axes_pos.get_rvecs(axes, (nx, nx))

    # Build Ewald sum objects
    ew, rvecs, kvecs = ewald.make_ewald(axes)
    total_energy = ew.sum(pos, rvecs, kvecs)
    energy_per_particle = total_energy / len(pos)

    # Analytical result from reference (for comparison)
    ac = axes_pos.volume(axes_primitive)
    expected = -3.900265 / ac**0.5

    # Assert energy per particle remains consistent
    assert np.isclose(energy_per_particle, expected, atol=1e-5)


def test_PrimitiveRectangle(nx=2):
    """Test Ewald sum energy per particle for nx x nx tiled rectangular supercells.

    This test verifies that the Ewald sum computed for an nx x nx rectangular lattice
    matches the analytical result reported in:
    'Some static and dynamical properties of a two-dimensional Wigner crystal'.

    Args:
        nx (int): Tiling factor along x and y directions (default is 2).

    The function compares the computed result to the reference value within an absolute
    tolerance of 1e-6.
    """
    # Primitive cell: rectangular lattice (aspect ratio lambda < 1)
    lam = 0.95
    a2 = 1.0
    a1 = lam * a2
    axes_primitive = np.diag([a1, a2])

    # Supercell axes (scaled)
    axes = nx * axes_primitive

    # Generate grid of lattice sites for supercell
    pos = axes_pos.get_rvecs(axes, (nx, nx))

    # Build Ewald sum objects
    ew, rvecs, kvecs = ewald.make_ewald(axes)
    total_energy = ew.sum(pos, rvecs, kvecs)
    energy_per_particle = total_energy / len(pos)

    # Analytical result from reference (for comparison)
    ac = axes_pos.volume(axes_primitive)
    expected = -3.898597 / (2 * (a1 * a2) ** 0.5)

    # Assert energy per particle remains consistent
    assert np.isclose(energy_per_particle, expected, atol=1e-6)


def test_CenteredRectangle(nx=2):
    """Test Ewald sum energy per particle for nx x nx tiled centered-rectangle supercells.

    This test verifies that the Ewald sum computed for an nx x nx centered rectangular lattice
    matches the analytical result reported in:
    'Some static and dynamical properties of a two-dimensional Wigner crystal'.

    Args:
        nx (int): Tiling factor along x and y directions (default is 2).

    The function compares the computed result to the reference value within an absolute
    tolerance of 1e-6.
    """
    # Primitive cell: centered rectangular lattice (aspect ratio lambda < 1)
    lam = 0.95
    a2 = 1.0
    a1 = lam * a2
    axes_primitive = np.array([
        [a1,     0.0],
        [a1 / 2, a2 / 2]
    ])

    # Supercell axes (scaled)
    axes = nx * axes_primitive

    # Generate grid of lattice sites for supercell
    pos = axes_pos.get_rvecs(axes, (nx, nx))

    # Build Ewald sum objects
    ew, rvecs, kvecs = ewald.make_ewald(axes)
    total_energy = ew.sum(pos, rvecs, kvecs)
    energy_per_particle = total_energy / len(pos)

    # Analytical result from reference (for comparison)
    ac = axes_pos.volume(axes_primitive)
    expected = -3.900647 / (2 * ac ** 0.5)

    # Assert energy per particle remains consistent
    assert np.isclose(energy_per_particle, expected, atol=1e-6)


def test_Hexagon(nx=2):
    """Test Ewald sum energy per particle for nx x nx tiled hexagonal (triangular) supercells.

    This test verifies that the Ewald sum computed for an nx x nx triangular/hexagonal lattice
    matches the analytical result reported in:
    'Some static and dynamical properties of a two-dimensional Wigner crystal'.

    Args:
        nx (int): Tiling factor along x and y directions (default is 3).
    
    The function compares the computed result to the reference value within an absolute
    tolerance of 1e-6.
    """
    # Primitive cell: hexagonal (triangular) lattice
    axes_primitive = np.array([
        [1.0, 0.0],
        [-0.5, np.sqrt(3) / 2]
    ])

    # Supercell axes (scaled)
    axes = nx * axes_primitive

    # Generate grid of lattice sites for supercell
    pos = axes_pos.get_rvecs(axes, (nx, nx))

    # Build Ewald sum objects
    ew, rvecs, kvecs = ewald.make_ewald(axes)
    total_energy = ew.sum(pos, rvecs, kvecs)
    energy_per_particle = total_energy / len(pos)

    # Analytical result from reference (for comparison)
    ac = axes_pos.volume(axes_primitive)
    expected = -3.921034 / (2 * ac**0.5)

    # Assert energy per particle remains consistent
    assert np.isclose(energy_per_particle, expected, atol=1e-6)


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

# def test_honeycomb(): #2 overlapping triangles
#     a =1
#     axes = np.array([[1,0],[-0.5,np.sqrt(3)/2]])
#     pos0 = np.array([[0, 0], [0, 1/np.sqrt(3)]])
#     expected = -1.51865 #previous issue
#     energy = compute_energy_per_particle(axes, pos0)
#     assert np.isclose(energy, expected, atol=1e-6)
