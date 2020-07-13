# PyFstat

This is a python package providing an interface to perform F-statistic based
continuous gravitational wave (CW) searches.

Getting started:
* This README provides information on
installation,
[contributing](#contributors) to 
and [citing](#citing-this-work) PyFstat.
* Additional usage documentation will be added to the
[project wiki](https://github.com/PyFstat/PyFstat/wiki) (work in progress).
* We also have a number of
[examples](https://github.com/PyFstat/PyFstat/tree/master/examples),
demonstrating different use cases.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1243930.svg)](https://doi.org/10.5281/zenodo.1243930)
[![PyPI version](https://badge.fury.io/py/PyFstat.svg)](https://badge.fury.io/py/PyFstat)
![Integration Tests](https://github.com/PyFstat/PyFstat/workflows/Integration%20Tests/badge.svg)
![Docker](https://github.com/PyFstat/PyFstat/workflows/Docker/badge.svg)

A [changelog](https://github.com/PyFstat/PyFstat/blob/master/CHANGELOG.md)
is also available (only maintained from v1.2 onwards).

## Installation

PyFstat releases can be installed in a variety of ways, including
[Docker/Singularity images](#docker-container),
[`pip install` from PyPi](#pip-install-from-PyPi),
and [from source releases on Zenodo](#install-pyfstat-from-source-zenodo-or-git-clone).
Latest development versions can
[also be installed with pip](#pip-install-from-github)
or [from a local git clone](#install-pyfstat-from-source-zenodo-or-git-clone).

If you don't have a recent `python` installation (`3.6+`) on your system or
prefer `conda` environments over system-wide installation / venvs, please
start with the [conda installation](#conda-installation) section.

In either case, be sure to also check out the notes on
[dependencies](#dependencies),
[ephemerides files](#ephemerides-installation)
and [citing this work](#citing-this-work).

### Docker container
Ready-to-use PyFstat containers are available at the [Packages](https://github.com/PyFstat/PyFstat/packages)
page. A git-hub account together with a personal access token is required. [Go to the wiki page](https://github.com/PyFstat/PyFstat/wiki/Containers) to learn how to pull them from the git-hub
registry using Docker or Singularity.

### pip install from PyPi

PyPi releases are available from https://pypi.org/project/PyFstat/.

Note that the PyFstat installation will fail at the
[LALSuite](https://pypi.org/project/lalsuite/) dependency stage
if your `pip` is too old (e.g. 18.1); to be on the safe side, before starting do
```
pip install --upgrade pip
```

Then, a simple
```
pip install pyfstat
```
should give you the latest release version with all dependencies.

If you are not installing into a [venv](https://docs.python.org/3/library/venv.html)
or [conda environment](#conda-installation),
on many systems you may need to use the `--user` flag.

### conda installation
PyFstat requires `python3.6+`.
While many systems come with a system-wide python installation,
it may not be sufficiently recent for this package;
anyway, it can often be easier to manage a user-specific python installation
(this way one does not require root access to install or remove modules).
One method to do this is to use the `conda` system, either through
the stripped down [miniconda](https://conda.pydata.org/miniconda.html)
installation, or the full-featured
[anaconda](https://www.anaconda.com/products/individual#Downloads)
(these are essentially the
same, but the `anaconda` version installs a variety of useful packages such as
`numpy` and `scipy` by default).
The fastest/easiest method is to follow your OS instructions
[here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
which will install Miniconda.

After you have installed a version of conda and
[set up an environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html), 
`pip` can also be used to install modules
(not all packages can be directly installed with `conda` itself,
including PyFstat at the moment, and for those using alternatives
to `conda`, `pip` is more universal).
This can be installed with
```
conda install pip
```
and then you can continue with the `pip` instructions,
either [a release version from PyPi](#pip-install-from-PyPi)
as described above
or [the latest development version directly from github](#pip-install-from-github)
as described below.

You may also consider starting to build your own `conda` environment from an
[igwn environment](https://computing.docs.ligo.org/conda/),
which contains most packages relevant to gravitational waves,
instead of doing it from scratch.

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


### Dependencies

PyFstat uses the following external python modules,
which should all be pulled in automatically if you use `pip`:

* [numpy](http://www.numpy.org/)
* [matplotlib](http://matplotlib.org/) >= 2.1
* [scipy](https://www.scipy.org/)
* [ptemcee](https://github.com/willvousden/ptemcee)
* [corner](https://pypi.python.org/pypi/corner/)
* [dill](https://pypi.python.org/pypi/dill)
* [tqdm](https://pypi.python.org/pypi/tqdm)
* [bashplotlib](https://github.com/glamp/bashplotlib)
* [peakutils](https://pypi.python.org/pypi/PeakUtils)
* [pathos](https://pypi.python.org/pypi/pathos)
* [lalsuite](https://pypi.org/project/lalsuite/) >= 6.72
* [versioneer]

In case the automatic install doesn't properly pull in all dependencies,
to install all of these modules manually, you can also run
```
pip install -r /PATH/TO/THIS/DIRECTORY/requirements.txt
```
For a general introduction to installing modules, see
[here](https://docs.python.org/3/installing/index.html).

*Optional dependencies*:
* [pycuda](https://pypi.org/project/pycuda/), required for the `tCWFstatMapVersion=pycuda`
  option of the `TransientGridSearch` class.
  (Note: `pip install pycuda` requires a working `nvcc` compiler in your path.)
* [pytest](https://docs.pytest.org) for running the test suite locally (`python -m pytest tests.py`)
* Developers are also highly encouraged to use the [black](https://black.readthedocs.io) style checker locally
(`black --check --diff .`),
as it is required to pass by the online integration pipeline.
* Some of the [examples](./examples) require [gridcorner](https://gitlab.aei.uni-hannover.de/GregAshton/gridcorner);
for `pip` users this is most conveniently installed by
```
pip install git+https://gitlab.aei.uni-hannover.de/GregAshton/gridcorner
```
* If you prefer to make your own LALSuite installation
[from source](https://git.ligo.org/lscsoft/lalsuite/),
make sure it is **swig-enabled** and contains at least the `lalpulsar` and `lalapps` packages.
A minimal configuration line to use would be e.g.:
```
./configure --prefix=${HOME}/lalsuite-install --disable-all-lal --enable-lalpulsar --enable-lalapps --enable-swig
```


### install PyFstat from source (Zenodo or git clone)

You can download a source release tarball from [Zenodo](https://doi.org/10.5281/zenodo.1243930)
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

### Ephemerides installation

PyFstat requires paths to earth and sun ephemerides files
in order to use the `lalpulsar.ComputeFstat` module and various `lalapps` tools.

If you have done `pip install lalsuite`
(or it got pulled in automatically as a dependency),
you need to manually download at least these two files:
*  [earth00-40-DE405.dat.gz](https://git.ligo.org/lscsoft/lalsuite/raw/master/lalpulsar/lib/earth00-40-DE405.dat.gz)
*  [sun00-40-DE405.dat.gz](https://git.ligo.org/lscsoft/lalsuite/raw/master/lalpulsar/lib/sun00-40-DE405.dat.gz)

(Other ephemerides versions exist, but these two files should be sufficient for most applications.)
You then need to tell PyFstat where to find these files,
by either setting an environment variable `$LALPULSAR_DATADIR`
or by creating a `~/.pyfstat.conf` file as described further below.
If you are working with a virtual environment,
you should be able to get a full working ephemerides installation with these commands:
```
mkdir $VIRTUAL_ENV/share/lalpulsar
wget https://git.ligo.org/lscsoft/lalsuite/raw/master/lalpulsar/lib/earth00-40-DE405.dat.gz -P $VIRTUAL_ENV/share/lalpulsar
wget https://git.ligo.org/lscsoft/lalsuite/raw/master/lalpulsar/lib/sun00-40-DE405.dat.gz -P $VIRTUAL_ENV/share/lalpulsar
echo 'export LALPULSAR_DATADIR=$VIRTUAL_ENV/share/lalpulsar' >> ${VIRTUAL_ENV}/bin/activate
deactivate
source path/to/venv/bin/activate
```

If instead you have built and installed lalsuite from source,
and set your path up properly through something like
`source $MYLALPATH/etc/lalsuite-user-env.sh`,
then the ephemerides path should be automatically picked up from
the `$LALPULSAR_DATADIR` environment variable.

Alternatively, you can place a file
`~/.pyfstat.conf` into your home directory which looks like

```
earth_ephem = '/home/<USER>/lalsuite-install/share/lalpulsar/earth00-19-DE405.dat.gz'
sun_ephem = '/home/<USER>/lalsuite-install/share/lalpulsar/sun00-19-DE405.dat.gz'
```
Paths set in this way will take precedence over the environment variable.

Finally, you can manually specify ephemerides files when initialising
each PyFstat search (as one of the arguments).

## Contributors

Maintainers:
* Greg Ashton
* David Keitel

Other contributors:
* Reinhard Prix
* Rodrigo Tenorio
* Karl Wette
* Sylvia Zhu

This project is open to development, please feel free to contact us
for advice or just jump in and submit an
[issue](https://github.com/PyFstat/PyFstat/issues/new/choose) or
[pull request](https://github.com/PyFstat/PyFstat/compare).

Here's what you need to know:
* The github automated tests currently run on `python` [3.6,3.7,3.8] and new PRs need to pass all these.
* The automated test also runs the [black](https://black.readthedocs.io) style checker. If possible, please run this locally before pushing changes / submitting PRs: `black --check --diff .` to show the required changes, or `black .` to automatically apply them.

## Citing this work

If you use `PyFstat` in a publication we would appreciate if you cite both a DOI for the software itself (see below)
and the original paper introducing the code: Ashton&Prix 2018 [[inspire](https://inspirehep.net/literature/1655200)] [[ADS](https://ui.adsabs.harvard.edu/abs/2018PhRvD..97j3020A/)].
If you use the transient module, please also cite: Keitel&Ashton 2018 [[inspire](https://inspirehep.net/literature/1673205)] [[ADS](https://ui.adsabs.harvard.edu/abs/2018CQGra..35t5003K/)].

If you'd like to cite the `PyFstat` package in general,
please refer to the [version-independent Zenodo listing](https://doi.org/10.5281/zenodo.1243930)
or use directly the following BibTeX entry:
```
@misc{pyfstat,
  author       = {Ashton, Gregory and
                  Keitel, David and
                  Prix, Reinhard,
                  and Tenorio, Rodrigo},
  title        = {PyFstat},
  month        = jan,
  year         = 2020,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.1243930},
  url          = {https://doi.org/10.5281/zenodo.1243930}
  note         = {\url{https://doi.org/10.5281/zenodo.1243930}}
}
```
You can also obtain DOIs for individual versioned releases
from the right sidebar at [Zenodo](https://doi.org/10.5281/zenodo.1243930).
