# EZwald

[![pytest](https://github.com/EmiliaSzynwald/EZwald/actions/workflows/pytest.yml/badge.svg)](https://github.com/EmiliaSzynwald/EZwald/actions/workflows/pytest.yml)

EZwald is an easy-to-use Ewald summation code for quasi-two-dimensional charged systems, built on JAX. It runs on GPUs and optimizes both the particle positions and the shape of the simulation cell, including systems with different charges per layer and point defects such as vacancies and interstitials. As the layers are pulled apart, the optimizer finds the stable crystal phases and the transition phases between them on its own. The code is modular, so it can serve as the energy calculator in a molecular dynamics or Monte Carlo program, or be extended to model new experiments.

- **Homepage:** https://dancing-electrons.github.io/research/students/
- **Tutorials:** [`tutorials/`](tutorials/)

## Features

- `ezwald.ewald`: Ewald summation for a single 2D layer
- `ezwald.bilayer_sum`: Ewald summation for a bilayer with adjustable layer spacing
- `ezwald.lattice`: simulation cells, reciprocal lattices, and real- and reciprocal-space grids

## Installation and Usage

Install EZwald from PyPI:

```bash
pip install ezwald
```

The tutorials also use `matplotlib` and `pandas`:

```bash
pip install matplotlib pandas
```

To run the tests, clone the repository and run:

```bash
pip install -r requirements.txt
python -m pytest -v
```

## Getting Started

If you are new to EZwald, start with [`quickstart.ipynb`](tutorials/quickstart.ipynb), which walks through building a crystal, computing its energy, tiling it, and relaxing it with optimization, for both monolayers and bilayers.

The [`tutorials/`](tutorials/) folder also contains ready-to-run examples that reproduce published results and are intended to be modified to fit your needs:

| Tutorial | Description |
|---|---|
| [`monolayer_energy_validation.ipynb`](tutorials/monolayer_energy_validation.ipynb) | Compares EZwald energies for four 2D lattices with the values of Bonsall & Maradudin (1977) |
| [`bilayer_optimization.ipynb`](tutorials/bilayer_optimization.ipynb) | Finds the ground-state bilayer lattice by optimization and compares it with Goldoni & Peeters (1996) |

## License

EZwald is released under the MIT License. See [LICENSE](LICENSE) for details.
