#reconstructing test cases
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from jewald import axes_pos, ewald, lattice
# from . import axes_pos, lattice, ewald
import matplotlib.pyplot as plt


def compute_energy_per_particle(axes, pos):
    ew, rvecs, kvecs = ewald.make_ewald(axes)
    total_energy = ew.sum(pos, rvecs, kvecs)
    return total_energy / len(pos)

# From axes you get: ew, rvecs, kvecs Using: make_ewald(axes)
# From pos, rvecs, kvecs you get: sum Using: ew.sum(pos, rvecs, kvecs)
# need axes and pos to compute sum

###Square: CORRECT
nx_square = 2 #any integer
axes_square = nx_square * np.eye(2) # 2x2 scaled by nx
pos_square = axes_pos.get_rvecs(axes_square, (nx_square, nx_square))

ew_square, rvecs_square, kvecs_square = ewald.make_ewald(axes_square)
square_sum = ew_square.sum(pos_square, rvecs_square, kvecs_square) / len(pos_square)
print("axes_square: ", axes_square)
print("pos_square: ", pos_square)

ac = 2**2
sumsq = -3.900265/ac**0.5
print("Square calculated energy: ", square_sum, "and expected", sumsq)
###Square END


###Primitive Rectangle: CORRECT
lam = 0.95
a2 = 1
a1 = lam*a2
pos_PR = np.array([
  [0, 0],
])
axes_PR = np.diag([a1, a2])
ew, rvecs, kvecs = ewald.make_ewald(axes_PR)
sum_PR = ew.sum(pos_PR, rvecs, kvecs)

ac = a1*a2
PRsum = -3.898597/2/ac**0.5
print("axes_PR: ", axes_PR)
print("pos_PR: ", pos_PR)
print("PR calculated energy: ", sum_PR, "and expected", PRsum)
###Primitive Rectangle END


###Centered Rectangle: off
lam = 0.95
a2 = 1
a1 = lam*a2
pos_CR = np.array([
  [0, 0],
])

axes_CR = np.array([
    [a1,    0],
    [a1/2,  a2]
])

ew, rvecs, kvecs = ewald.make_ewald(axes_CR)
sum_CR = ew.sum(pos_CR, rvecs, kvecs)

ac = 0.5*a1*a2
CRsum =  -3.900647/2/ac**0.5
print("axes_CR: ", axes_CR)
print("pos_CR: ", pos_CR)
print("CR calculated energy: ", sum_CR, "and expected", CRsum)
###Centered Rectangle END


###Oblique:

###Oblique END


###Hexagonal (Triangular): CORRECT
nx =3
axes_hex = np.array([[1, 0], [-0.5, (np.sqrt(3)) / 2]]) * 3  # Stretches the lattice by factor of 3
pos_hex = np.array([[0, 0]])


ew_hex, rvecs_hex, kvecs_hex = ewald.make_ewald(axes_hex)
hex_sum = ew_hex.sum(pos_hex, rvecs_hex, kvecs_hex) / len(pos_hex)

ac = ((np.sqrt(3)) / 2) * (nx**2)
sumhex =  -3.921034/2/ac**0.5

print("axes_hex: ", axes_hex)
print("pos_hex: ", pos_hex)
print("Hex calculated energy: ", hex_sum, "and expected", sumhex)
###Hexagonal END



### Visualizing
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, aspect=1)
# draw_cell(ax, AXES/cell)
# ax.plot(*pos.T, ls='', marker='.')



#### draw cell

def set_default_cell_styles(kwargs):
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

#### draw cell

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, aspect=1)
# draw_cell(ax, axes_CR)
# ax.plot(*pos_CR.T, ls='', marker='.')