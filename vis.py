"""Visualization utilities for crystal structures.

This module provides functions for visualizing crystal structures, including
drawing unit cells and plotting atomic positions.
"""
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def set_default_cell_styles(kwargs):
  """Set default styling options for cell visualization.

  Modifies the kwargs dictionary in-place to add default values for color,
  alpha, and linewidth if not already specified.

  Args:
    kwargs (dict): Dictionary of matplotlib plotting keyword arguments.
        Modified in-place.
  """
  if not (('c' in kwargs) or ('color' in kwargs)):
    kwargs['c'] = 'gray'
  if ('alpha' not in kwargs):
    kwargs['alpha'] = 0.6
  if not (('lw' in kwargs) or ('linewidth' in kwargs)):
    kwargs['lw'] = 2

def draw_cell(ax, axes, corner=None, enclose=True, **kwargs):
  """ draw cell on ax
  see example in draw_crystal

  Args:
    ax (plt.Axes): matplotlib Axes object, must have projection='3d'
    axes (np.array): lattice vectors in row-major 3x3 array
    corner (np.array,optional): lower left corner of the lattice
      ,use (0,0,0) by default
    enclose (bool): enclose the cell with lattice vectors
      ,default is True. If False, then draw lattice vectors only
    kwargs (dict,optional): keyword arguments passed to plt.plot
  Returns:
    list: a list of plt.Line3D or Line, one for each lattice vector
  Example:
    >>> # draw 2D rectangular box, centered around (0, 0)
    >>> box = np.array([3.0, 1.5])
    >>> axes = np.diag(box)
    >>> fig, ax = plt.subplots(1, 1)
    >>> lines = draw_cell(ax, axes, corner=-box/2)
    >>> plt.show()
  """
  ndim = len(axes)
  if ndim not in [2, 3]:
    raise RuntimeError('ndim = %d is not supported' % ndim)
  cell = []
  if corner is None:
    corner = np.zeros(ndim)

  set_default_cell_styles(kwargs)

  # a,b,c lattice vectors
  for iax in range(ndim):
    start = corner
    end   = start + axes[iax]
    line = ax.plot(*zip(start, end), **kwargs)
    cell.append(line)

  if enclose:
    # counter a,b,c vectors
    for iax in range(ndim):
      start = corner+axes.sum(axis=0)
      end   = start - axes[iax]
      line = ax.plot(*zip(start, end), **kwargs)
      cell.append(line)

    if ndim > 2:
      # remaining vectors needed to enclose cell
      for iax in range(ndim):
        start = corner+axes[iax]
        for jax in range(ndim):
          if jax == iax:
            continue
          end = start + axes[jax]
          line = ax.plot(*zip(start, end), **kwargs)
          cell.append(line)
  return cell

def main():
  """Main visualization function.

  Loads and displays crystal structures before and after optimization,
    reading from 'axes0.dat', 'pos0.dat', 'axes1.dat', and 'pos1.dat'.
  """
  fig = plt.figure()

  axl = []
  for i in range(2):
    axes = np.loadtxt('axes%d.dat' % i)
    pos = np.loadtxt('pos%d.dat' % i)
    ax = fig.add_subplot(1, 2, i+1, aspect=1)
    draw_cell(ax, axes)
    ax.plot(*pos.T, ls='', marker='.')
    axl.append(ax)

  axl[0].set_title('before')
  axl[1].set_title('after')
  fig.tight_layout()
  plt.show()

if __name__ == '__main__':
  main()  # set no global variable
