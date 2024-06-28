# PyFstat

This is a python package providing an interface to perform F-statistic based
continuous gravitational wave (CW) searches,
built on top of the [LALSuite library](https://doi.org/10.7935/GT1W-FZ16).

Getting started:
* This README provides information on
[installing](#installation),
[contributing](#contributors) to
and [citing](#citing-this-work) PyFstat.
* PyFstat usage and its API are documented at [pyfstat.readthedocs.io](https://pyfstat.readthedocs.io/).
* We also have a number of [tutorials](https://github.com/PyFstat/PyFstat/tree/master/examples/tutorials) and
[examples](https://github.com/PyFstat/PyFstat/tree/master/examples), demonstrating different use cases.
You can run them locally, or online as jupyter notebooks with
[binder](https://mybinder.org/v2/gh/PyFstat/PyFstat/master).
* The [project wiki](https://github.com/PyFstat/PyFstat/wiki) is mainly used for developer information.
* A [changelog](https://github.com/PyFstat/PyFstat/blob/master/CHANGELOG.md)
is also available.
* PyFstat >=2.0.0 requires a recent LALSuite (>=7.13) / lalpulsar (>=6.0)
including the [SFTv3 specification](https://dcc.ligo.org/T040164-v2/public).
If you need to work with older versions, the last PyFstat release supporting those was `1.19.1`.

[![PyPI version](https://badge.fury.io/py/PyFstat.svg)](https://badge.fury.io/py/PyFstat)
[![Conda version](https://anaconda.org/conda-forge/pyfstat/badges/version.svg)](https://anaconda.org/conda-forge/pyfstat)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3967045.svg)](https://doi.org/10.5281/zenodo.3967045)
[![ASCL](https://img.shields.io/badge/ascl-2102.027-blue.svg?colorB=262255)](https://ascl.net/2102.027)
[![JOSS](https://joss.theoj.org/papers/10.21105/joss.03000/status.svg)](https://doi.org/10.21105/joss.03000)
[![Docker](https://github.com/PyFstat/PyFstat/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/PyFstat/PyFstat/actions/workflows/docker-publish.yml)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/PyFstat/PyFstat/master)
[![Integration Tests](https://github.com/PyFstat/PyFstat/actions/workflows/integration.yml/badge.svg)](https://github.com/PyFstat/PyFstat/actions/workflows/integration.yml)
[![codecov](https://codecov.io/gh/PyFstat/PyFstat/branch/master/graph/badge.svg?token=P0W8MIIUGD)](https://codecov.io/gh/PyFstat/PyFstat)
[![Documentation Status](https://readthedocs.org/projects/pyfstat/badge/?version=latest)](https://pyfstat.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Installation

PyFstat releases can be installed in a variety of ways, including
[`pip install` from PyPI](#pip-install-from-PyPi),
[conda](#conda-installation),
[Docker/Singularity images](#docker-container),
and [from source releases on Zenodo](#install-pyfstat-from-source-zenodo-or-git-clone).
Latest development versions can
[also be installed with pip](#pip-install-from-github)
or [from a local git clone](#install-pyfstat-from-source-zenodo-or-git-clone).

If you don't have a matching `python` installation
(currently `3.9` to `3.12`)
on your system,
then `Docker` or `conda` are the easiest paths.

In either case, be sure to also check out the notes on
[dependencies](#dependencies)
and [citing this work](#citing-this-work).

If you run into problems with ephemerides files, check the wiki page on
[ephemerides installation](https://github.com/PyFstat/PyFstat/wiki/ephemerides-installation).

### pip install from PyPI

PyPI releases are available from https://pypi.org/project/PyFstat/.

A simple
```
pip install pyfstat
```
should give you the latest release version with all dependencies;
recent releases now also include
(or pull in, via dependency)
a sufficient minimal set of ephemerides files.

If you are not installing into a [venv](https://docs.python.org/3/library/venv.html)
or [conda environment](#conda-installation)
(you really should!),
on many systems you may need to use the `--user` flag.

Note that the PyFstat installation will fail at the
LALSuite dependency stage
if your `pip` is too old (e.g. 18.1); to fix this, do
```
pip install --upgrade pip setuptools
```

### conda installation

See [this wiki page](https://github.com/PyFstat/PyFstat/wiki/conda-environments)
for further instructions on installing conda itself,
installing PyFstat into an existing environment,
or for .yml recipes to set up a PyFstat-specific environment
both for normal users and for developers.

If getting PyFstat from conda-forge, it already includes the required ephemerides files.

### Docker container

Ready-to-use PyFstat containers are available at the [Packages](https://github.com/PyFstat/PyFstat/packages)
page. A GitHub account together with a personal access token is required.
[Go to the wiki page](https://github.com/PyFstat/PyFstat/wiki/Containers)
to learn how to pull them from the GitHub registry using `Docker` or `Singularity`.

### pip install from github

Development versions of PyFstat can also be easily installed by
pointing pip directly to this git repository,
which will give you the latest version of the master branch:
```
pip install git+https://github.com/PyFstat/PyFstat
```
or, if you have an ssh key installed in github:
```
pip install git+ssh://git@github.com/PyFstat/PyFstat
```

This should pull in all dependencies in the same way as installing from PyPI.

### install PyFstat from source (Zenodo or git clone)

You can download a source release tarball from [Zenodo](https://doi.org/10.5281/zenodo.3967045)
and extract to an arbitrary temporary directory.
Alternatively, clone this repository:

```
git clone https://github.com/PyFstat/PyFstat.git
```

The module and associated scripts can be installed system wide
(or to the currently active venv),
assuming you are in the (extracted or cloned) source directory, via
```
python setup.py install
```
As a developer, alternatively
```
python setup.py develop
```
or
```
pip install -e /path/to/PyFstat
```
can be useful so you can directly see any changes you make in action.
Alternatively (not recommended!), add the source directory directly to your python path.

To check that the installation
was successful, run
```
python -c 'import pyfstat'
```
if no error message is output, then you have installed `pyfstat`. Note that
the module will be installed to whichever python executable you call it from.

This should pull in all dependencies in the same way as installing from PyPI.

### Dependencies

PyFstat uses the following external python modules,
which should all be pulled in automatically if you use `pip`:

* [corner](https://pypi.python.org/pypi/corner/)
* [dill](https://pypi.python.org/pypi/dill)
* [lalsuite](https://pypi.org/project/lalsuite/)
* [matplotlib](https://matplotlib.org/)
* [numpy](https://www.numpy.org/)
* [pathos](https://pypi.python.org/pypi/pathos)
* [ptemcee](https://github.com/willvousden/ptemcee)
* [scipy](https://www.scipy.org/)
* [solar-system-ephemerides](https://pypi.org/project/solar-system-ephemerides/)
  (from `lalsuite>7.15` on)
* [tqdm](https://pypi.python.org/pypi/tqdm)
* [versioneer](https://pypi.org/project/versioneer/)

For a general introduction to installing modules, see
[here](https://docs.python.org/3/installing/index.html).

NOTE: currently we have pinned to
`numpy<2.0` (same as LALSuite).

NOTE: We require a recent LALSuite (>=7.13) / lalpulsar (>=6.0).
If you need to work with older versions,
the last PyFstat release supporting those was `1.19.1`.

NOTE: Through pip, we install the dependency `lalsuite[lalpulsar]`.
The optional flag `[lalpulsar]` is ignored for `lalsuite <=  7.15`,
as those releases carry their own ephemerides.
For `lalsuite > 7.15`, `[lalpulsar]` installs `solar-system-ephemerides`
and uses this auxiliary package to access the ephemerides.

### Optional dependencies

PyFstat manages optional dependencies through setuptool's `extras_require`.

Available sets of optional dependencies are:

* `chainconsumer` ([Samreay/Chainconsumer](https://github.com/Samreay/ChainConsumer)): Required to run some optional
plotting methods and some of the [example scripts](./examples).
* `dev`: Collects `docs`, `style`, `test` and `wheel`.
* `docs`: Required dependencies to build the documentation.
* `pycuda` ([PyPI](https://pypi.org/project/pycuda/)): Required for the `tCWFstatMapVersion=pycuda`
  option of the `TransientGridSearch` class. (Note: Installing `pycuda` requires a working
  `nvcc` compiler in your path.)
* `style`: Includes the `flake8` linter ([flake8.pycqa](https://flake8.pycqa.org/en/latest)),
  `black` style checker ([black.readthedocs](https://black.readthedocs.io)),
  and `isort` for import ordering ([pycqa.github.io](https://pycqa.github.io/isort/)).
  These checks are required to pass by the online integration pipeline.
* `test`: For running the test suite locally using [pytest](https://docs.pytest.org) and some of its addons
  (`python -m pytest tests/`).
* `wheel`: Includes `wheel` and `check-wheel-contents`.

Installation can be done by adding one or more of the aforementioned tags to the installation command.

For example, installing PyFstat including `chainconsumer`, `pycuda` and `style` dependencies would look like
(mind the lack of whitespaces!)
```
pip install pyfstat[chainconsumer,pycuda,style]
```
This command accepts the "development mode" tag `-e`.

Note that LALSuite is a default requirement, not an optional one,
but its installation from PyPI can be disabled
by setting the `NO_LALSUITE_FROM_PYPI` environment variable,
e.g. for a development install from a local git clone:
```
NO_LALSUITE_FROM_PYPI=1 pip install -e .
```
This can be useful to avoid duplication when in a conda environment
or installing LALSuite from source.

### Using LALSuite built from source

Instructions to use a custom local LALSuite installation can be found in [here on the wiki](https://github.com/PyFstat/PyFstat/wiki/Using-LALSuite-built-from-source).

## Contributing to PyFstat

This project is open to development, please feel free to contact us
for advice or just jump in and submit an
[issue](https://github.com/PyFstat/PyFstat/issues/new/choose) or
[pull request](https://github.com/PyFstat/PyFstat/compare).

Here's what you need to know:
* As a developer, you should install directly from a git clone,
  with either `pip install -e .[dev]` into some environment
  or creating a development-enabled conda environment directly from the
  `pyfstat-dev.yml` file
  as explained on [this wiki page](https://github.com/PyFstat/PyFstat/wiki/conda-environments).
  Please also run, just once after installing:
  ```
  pre-commit install
  ```

  This sets up everything for automated code quality tests (see below)
  to be checked for you at every commit.
* The github automated tests currently run on `python` [3.9,3.10,3.11,3.12]
  and new PRs need to pass all these.
* You can also run the full test suite locally via `pytest tests/`,
  or run individual tests as explained
  [on this page](https://docs.pytest.org/en/6.2.x/usage.html#specifying-tests-selecting-tests).
* The automated test on github also runs
  the [black](https://black.readthedocs.io) style checker,
  the [flake8](https://flake8.pycqa.org/en/latest/) linter,
  and the [isort](https://pycqa.github.io/isort/) import ordering helper.
* If you have installed the dev dependencies correctly via pip or conda,
  and ran `pre-commit install` once,
  then you're ready to let the `pre-commit` tool do all of this automatically for you
  every time you do `git commit`.
  For anything that would fail on the github integration tests,
  it will then either automatically reformat your code to match our style
  or print warnings for things to fix.
  The first time it will take a while for setup,
  later it should be faster.
* If for some reason you can't use `pre-commit`,
  you can still manually run these tools before pushing changes / submitting PRs:
  `isort .` to sort package imports,
  `flake8 --count --statistics .` to find common coding errors and then fix them manually,
  `black --check --diff .` to show the required style changes, or `black .` to automatically apply them.

## Contributors

Maintainers:
* Greg Ashton
* David Keitel

Active contributors:
* Reinhard Prix
* Rodrigo Tenorio

Other contributors:
* Karl Wette
* Sylvia Zhu
* Dan Foreman-Mackey (`pyfstat.gridcorner` is based on DFM's [corner.py](https://github.com/dfm/corner.py))


## Citing this work

If you use `PyFstat` in a publication we would appreciate if you cite both a release DOI for the software itself (see below)
and one or more of the following scientific papers:
* The recent JOSS (Journal of Open Source Software) paper summarising the package:
[Keitel, Tenorio, Ashton & Prix 2021](https://doi.org/10.21105/joss.03000)
([inspire:1842895](https://inspirehep.net/literature/1842895)
/ [ADS:2021arXiv210110915K](https://ui.adsabs.harvard.edu/abs/2021arXiv210110915K/)).
* The original paper introducing the package and the MCMC functionality:
[Ashton&Prix 2018](https://doi.org/10.1103/PhysRevD.97.103020)
([inspire:1655200](https://inspirehep.net/literature/1655200)
/ [ADS:2018PhRvD..97j3020A](https://ui.adsabs.harvard.edu/abs/2018PhRvD..97j3020A/)).
* The methods paper introducing a Bayes factor to evaluate the multi-stage follow-up:
[Tenorio, Keitel, Sintes 2021](https://doi.org/10.1103/PhysRevD.104.084012)
([inspire:1865975](https://inspirehep.net/literature/1865975)
/ [ADS:2021PhRvD.104h4012T](https://ui.adsabs.harvard.edu/abs/2021PhRvD.104h4012T/))
* For transient searches:
[Keitel&Ashton 2018](https://doi.org/10.1088/1361-6382/aade34)
([inspire:1673205](https://inspirehep.net/literature/1673205)
/ [ADS:2018CQGra..35t5003K](https://ui.adsabs.harvard.edu/abs/2018CQGra..35t5003K/)).
* For glitch-robust searches:
[Ashton, Prix & Jones 2018](https://doi.org/10.1103/PhysRevD.98.063011)
([inspire:1672396](https://inspirehep.net/literature/1672396)
/ [ADS:2018PhRvD..98f3011A](https://ui.adsabs.harvard.edu/abs/2018PhRvD..98f3011A/)

If you'd additionally like to cite the `PyFstat` package in general,
please refer to the [version-independent Zenodo listing](https://doi.org/10.5281/zenodo.3967045)
or use directly the following BibTeX entry:
```
@misc{pyfstat,
  author       = {Ashton, Gregory and
                  Keitel, David and
                  Prix, Reinhard
                  and Tenorio, Rodrigo},
  title        = {{PyFstat}},
  month        = jul,
  year         = 2020,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.3967045},
  url          = {https://doi.org/10.5281/zenodo.3967045},
  note         = {\url{https://doi.org/10.5281/zenodo.3967045}}
}
```
You can also obtain DOIs for individual versioned releases (from 1.5.x upward)
from the right sidebar at [Zenodo](https://doi.org/10.5281/zenodo.3967045).

Alternatively, if you've used PyFstat up to version 1.4.x in your works,
the DOIs for those versions can be found from the sidebar at
[this older Zenodo record](https://doi.org/10.5281/zenodo.1243930)
and please amend the BibTeX entry accordingly.


PyFstat uses the [`ptemcee` sampler](https://github.com/willvousden/ptemcee), which can be
cited as
[Vousden, Far & Mandel 2015](https://doi.org/10.1093/mnras/stv2422)
([ADS:2016MNRAS.455.1919V](https://ui.adsabs.harvard.edu/abs/2016MNRAS.455.1919V/abstract))
and [Foreman-Mackey, Hogg, Lang, and Goodman 2012](https://doi.org/10.1086/670067)
([2013PASP..125..306F](https://ui.adsabs.harvard.edu/abs/2013PASP..125..306F/abstract)).

PyFstat also makes generous use of functionality from the LALSuite library
and it will usually be appropriate to also cite that project
(see [this recommended bibtex entry](https://git.ligo.org/lscsoft/lalsuite/#acknowledgment))
and also [Wette 2020](https://doi.org/10.1016/j.softx.2020.100634)
([inspire:1837108](https://inspirehep.net/literature/1837108)
/ [ADS:2020SoftX..1200634W](https://ui.adsabs.harvard.edu/abs/2020SoftX..1200634W/))
for the C-to-python [SWIG](http://www.swig.org) bindings.
