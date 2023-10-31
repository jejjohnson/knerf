# Neural Fields with Keras-Core
[![codecov](https://codecov.io/gh/jejjohnson/fieldx/branch/main/graph/badge.svg?token=YGPQQEAK91)](https://codecov.io/gh/jejjohnson/fieldx)

> This package implements some minimal functionality to train neural fields using keras-core library.
> It will also feature some extra functionality when working with spatiotemporal datasets.
> I would eventually like it to have an efficient implementation for introducing modulation within each of the models.



---
## Key Algorithms

**Fourier Features**. The canonical method for Fourier Features.

**SIREN**. The canonical method that uses sinusoidal activation functions with a special initialization procedure.

**FourierNet** (TODO)

**GaborNet** (TODO)

**Spherical Harmonics** (TODO)

**Modulation** (TODO)

---
## Installation

We can install it directly through pip

```bash
pip install git+https://github.com/jejjohnson/knerf
```

We also use poetry for the development environment.

```bash
git clone https://github.com/jejjohnson/knerf.git
cd knerf
conda create -n knerf python=3.11 poetry
poetry install
```



---
## References

**Software**

* [jaxdf](https://github.com/ucl-bug/jaxdf/tree/main) - Arbitrary Discretizations
* [diffrax (example)](https://docs.kidger.site/diffrax/examples/nonlinear_heat_pde/) - example of diffrax (and equinox) with an finite difference discretization