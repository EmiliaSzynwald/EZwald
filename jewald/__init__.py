"""JAX-based Ewald summation library for periodic systems.

This package provides efficient implementations of Ewald summation for computing
Coulomb interactions in periodic systems, including:
- Full periodic boundary conditions (2D and 3D)
- Quasi-2D bilayer systems
- Optimized periodic boundary condition handling
- Structure factor calculations

Main modules:
    - ewald: Full periodic Ewald summation
    - bilayer_sum: Quasi-2D bilayer Ewald summation
    - geometry: Lattice utilities and PBC functions
    - lattice: Lattice generation and transformation
    - axes_pos: Position and lattice vector utilities
    - sofk: Structure factor calculations
"""
