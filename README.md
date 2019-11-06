# PyFstat

This is a python package providing an interface to perform F-statistic based
continuous gravitational wave (CW) searches.

For documentation, please use the [wiki](https://gitlab.aei.uni-hannover.de/GregAshton/PyFstat/wikis/home).

In the
[examples](https://gitlab.aei.uni-hannover.de/GregAshton/PyFstat/tree/master/examples),
we have a number of scripts demonstrating different use cases.


## Installation

### `python` installation
The scripts are written in `python 2.7+` and therefore require a working
`python` installation. While many systems come with a system wide python
installation, it can often be easier to manage a user-specific python
installation. This way one does not require root access to install or remove
modules. One method to do this, is to use the `conda` system, either through
the stripped down [miniconda](http://conda.pydata.org/miniconda.html)
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
$ conda install pip
```

### Clone the repository

In a terminal, clone the directory:

```
$ git clone https://gitlab.aei.uni-hannover.de/GregAshton/PyFstat.git
```

### Dependencies

`pyfstat` uses the following external python modules:

* [numpy](http://www.numpy.org/)
* [matplotlib](http://matplotlib.org/) >= 1.4
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

For an introduction to installing modules see
[here](https://docs.python.org/3.5/installing/index.html). If you are using
`pip`, to install all of these modules, run
```
$ pip install -r /PATH/TO/THIS/DIRECTORY/requirements.txt
```

If you prefer to make your own LALSuite installation
[https://git.ligo.org/lscsoft/lalsuite/](from source),
make sure it is **swig-enabled** and contains at least the `lalpulsar` package.
A minimal confuration line to use would be e.g.:

```
$ ./configure --prefix=${HOME}/lalsuite-install --disable-all-lal --enable-lalpulsar --enable-lalapps --enable-swig
```


### `pyfstat` installation

The module and associated scripts can be installed system wide (or to the currently active venv),
assuming you are in the source directory, via
```
$ python setup.py install
```
or simply add this directory to your python path. To check that the installation
was successful, run
```
$ python -c 'import pyfstat'
```
if no error message is output, then you have installed `pyfstat`. Note that
the module will be installed to whichever python executable you call it from.

### Ephemeris installation

The scripts require paths to earth and sun ephemeris files in order to use the
`lalpulsar.ComputeFstat` module. This should be automatically picked up from
the $LALPULSAR_DATADIR environment variable, defaulting to the
00-40-DE421 ephemerides or 00-19-DE421 as a backup.
Alternatively, these can either be manually specified when initialising
each search (as one of the arguments), or simply by placing a file
`~/.pyfstat.conf` into your home directory which looks like

```
earth_ephem = '/home/<USER>/lalsuite-install/share/lalpulsar/earth00-19-DE421.dat.gz'
sun_ephem = '/home/<USER>/lalsuite-install/share/lalpulsar/sun00-19-DE421.dat.gz'
```
Paths set in this way will take precedence over the environment variable.

### Contributors

* Greg Ashton
* David Keitel
* Reinhard Prix
* Karl Wette
* Sylvia Zhu

This project is open to development, please feel free to contact us
for advice or just jump in and submit a pull request.

## Citing this work

If you use `PyFstat` in a publication we would appreciate if you cite the
original paper introducing the code, the [ADS page can be found
here](http://adsabs.harvard.edu/abs/2018arXiv180205450A) and the version
release:

```
@misc{pyfstat,
  author       = {{Ashton}, G. and {Keitel}, D.},
  title        = {{PyFstat-v1.2}},
  month        = may,
  year         = 2018,
  doi          = {10.5281/zenodo.1243931},
  url          = {https://doi.org/10.5281/zenodo.1243931},
  note= {\url{https://doi.org/10.5281/zenodo.1243931}}
}
```


