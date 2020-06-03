# PyFstat

This is a python package providing an interface to perform F-statistic based
continuous gravitational wave (CW) searches.

Getting started:
* This README provides information on installation, contributing and citing.
* Additional usage documentation will be added to the
[project wiki](https://github.com/PyFstat/PyFstat/wiki) (work in progress).
* We also have a number of
[examples](./examples),
demonstrating different use cases.

![Integration Tests](https://github.com/PyFstat/PyFstat/workflows/Integration%20Tests/badge.svg)

## Installation

### python installation
This package requires `python3.6+`.
While many systems come with a system wide python
installation, it can often be easier to manage a user-specific python
installation. This way one does not require root access to install or remove
modules. One method to do this, is to use the `conda` system, either through
the stripped down [miniconda](https://conda.pydata.org/miniconda.html)
installation, or the full-featured
[anaconda](https://www.continuum.io/downloads) (these are essentially the
same, but the `anaconda` version installs a variety of useful packages such as
`numpy` and `scipy` by default).

The fastest/easiest method is to follow your OS instructions
[here](https://conda.io/docs/install/quick.html) which will install Miniconda.

For the rest of this tutorial, we will make use of `pip` to install modules (
not all packages can be installed with `conda` and for those using alternatives
to `conda`, `pip` is more universal).

This can be installed with
```
conda install pip
```

### install PyFstat the easy way

Currently, the easiest way to install PyFstat is to point pip to this git repository,
which will give you the latest master version:
```
pip install git+https://github.com/PyFstat/PyFstat
```
or, if you have an ssh key installed in github:
```
pip install git+ssh://git@github.com/PyFstat/PyFstat
```

See further down for installing manually from a
[Zenodo source release](https://doi.org/10.5281/zenodo.1243930)
or from a local git clone.


### Dependencies

PyFstat uses the following external python modules,
which should all be pulled in automatically if you use pip:

* [numpy](http://www.numpy.org/)
* [matplotlib](http://matplotlib.org/) >= 2.1
* [scipy](https://www.scipy.org/)
* [ptemcee](https://github.com/willvousden/ptemcee)
* [corner](https://pypi.python.org/pypi/corner/)
* [dill](https://pypi.python.org/pypi/dill)
* [peakutils](https://pypi.python.org/pypi/PeakUtils)
* [pathos](https://pypi.python.org/pypi/pathos)
* [tqdm](https://pypi.python.org/pypi/tqdm)
* [bashplotlib](https://github.com/glamp/bashplotlib)
* [lalsuite](https://pypi.org/project/lalsuite/)

*Optional*
* [pycuda](https://pypi.org/project/pycuda/), required for the tCWFstatMapVersion=pycuda
  option of the TransientGridSearch class.
  (Note: 'pip install pycuda' requires a working nvcc compiler in your path.)
* [pytest](https://docs.pytest.org) for running the test suite locally (`python -m pytest tests.py`)
* developers are also highly encouraged to use the [black](https://black.readthedocs.io) style checker locally
(`black --check --diff .`),
as it is required to pass by the online integration pipeline
* some of the [examples](./examples) require [gridcorner](https://gitlab.aei.uni-hannover.de/GregAshton/gridcorner);
for pip users this is most conveniently installed by
```
pip install git+https://gitlab.aei.uni-hannover.de/GregAshton/gridcorner
```

In case the automatic install doesn't properly pull in all dependencies,
to install all of these modules manually, you can also run
```
pip install -r /PATH/TO/THIS/DIRECTORY/requirements.txt
```
For a general introduction to installing modules, see
[here](https://docs.python.org/3.6/installing/index.html).

If you prefer to make your own LALSuite installation
[from source](https://git.ligo.org/lscsoft/lalsuite/),
make sure it is **swig-enabled** and contains at least the `lalpulsar` and `lalapps` packages.
A minimal confuration line to use would be e.g.:

```
./configure --prefix=${HOME}/lalsuite-install --disable-all-lal --enable-lalpulsar --enable-lalapps --enable-swig
```


### PyFstat installation from source

In a terminal, clone this repository:

```
git clone https://github.com/PyFstat/PyFstat.git
```

The module and associated scripts can be installed system wide
(or to the currently active venv),
assuming you are in the source directory, via
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
Alternatively, add the source directory directly to your python path.

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

If you have done `pip install lalsuite`,
you need to manually download at least these two files:
*  [earth00-40-DE405.dat.gz](https://git.ligo.org/lscsoft/lalsuite/raw/master/lalpulsar/lib/earth00-40-DE405.dat.gz)
*  [sun00-40-DE405.dat.gz](https://git.ligo.org/lscsoft/lalsuite/raw/master/lalpulsar/lib/sun00-40-DE405.dat.gz)

(Other ephemerides versions exist, but these should be sufficient for most applications.)
You then need to tell PyFstat where to find these files,
by either setting an environment variable $LALPULSAR_DATADIR
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
the $LALPULSAR_DATADIR environment variable.
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

Other past contributors:
* Reinhard Prix
* Rodrigo Tenorio
* Karl Wette
* Sylvia Zhu

This project is open to development, please feel free to contact us
for advice or just jump in and submit an issue or pull request.

Here's what you need to know:
* The github automated tests currently run on python [3.6,3.7,3.8] and new PRs need to pass all these.
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
                  Prix, Reinhard},
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
