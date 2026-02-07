#!/usr/bin/env python3
"""Visualization: draw periodic cells and particle positions.

Loads axes and positions from axes0.dat, pos0.dat, axes1.dat, pos1.dat
(typically produced by main.py) and draws the unit cell and particles
before and after optimization.
"""

import numpy as np
import matplotlib.pyplot as plt


def set_default_cell_styles(kwargs):
  """Set default line style for cell edges (gray, semi-transparent, linewidth 2).

  Modifies kwargs in place; only sets keys that are not already present.

  Args:
    kwargs: dict. Keyword arguments for plotting (e.g. for plt.plot).
  """
  if not (('c' in kwargs) or ('color' in kwargs)):
    kwargs['c'] = 'gray'
  if ('alpha' not in kwargs):
    kwargs['alpha'] = 0.6
  if not (('lw' in kwargs) or ('linewidth' in kwargs)):
    kwargs['lw'] = 2


def draw_cell(ax, axes, corner=None, enclose=True, **kwargs):
  """Draw the unit cell (lattice vectors and optionally full enclosure) on ax.

  Args:
    ax: matplotlib Axes. Must support 2D or 3D (e.g. projection='3d' for 3D).
    axes: Array of shape (ndim, ndim). Lattice vectors in row-major (rows = a, b, c).
    corner: Optional array of shape (ndim,). Origin for drawing; default (0,...,0).
    enclose: bool. If True, draw all edges to enclose the cell; if False, only
        the lattice vectors from corner. Default True.
    **kwargs: Keyword arguments passed to plt.plot (color, linewidth, etc.).

  Returns:
    list. List of plot artists (e.g. Line2D/Line3D) for each drawn segment.

  Raises:
    RuntimeError: If ndim is not 2 or 3.

  Example:
    Draw a 2D rectangular box centered at (0, 0):
      box = np.array([3.0, 1.5])
      axes = np.diag(box)
      fig, ax = plt.subplots(1, 1)
      lines = draw_cell(ax, axes, corner=-box/2)
      plt.show()
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
  """Load axes/pos from axes0.dat, pos0.dat, axes1.dat, pos1.dat and plot before/after."""
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
