# PyFstat

This is a python package providing an interface to perform F-statistic based
continuous gravitational wave (CW) searches.

For documentation, please use the [wiki](https://gitlab.aei.uni-hannover.de/GregAshton/PyFstat/wikis/home).

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

In a terminal, to clone the directory:

```
$ git clone git@gitlab.aei.uni-hannover.de:GregAshton/PyFstat.git
```

### Dependencies

`pyfstat` makes uses the following external python modules:

* [numpy](http://www.numpy.org/)
* [matplotlib](http://matplotlib.org/) >= 1.4
* [scipy](https://www.scipy.org/)
* [emcee](http://dan.iel.fm/emcee/current/)
* [corner](https://pypi.python.org/pypi/corner/)
* [dill](https://pypi.python.org/pypi/dill)
* [peakutils](https://pypi.python.org/pypi/PeakUtils)

*Optional*
* [tqdm](https://pypi.python.org/pypi/tqdm)(optional), if installed, this
  provides a useful progress bar and estimate of the remaining run-time.
* [bashplotlib](https://github.com/glamp/bashplotlib), if installed, presents
  a histogram of the loaded SFT data

For an introduction to installing modules see
[here](https://docs.python.org/3.5/installing/index.html). If you are using
`pip`, to install all of these modules, run
```
$ pip install -r /PATH/TO/THIS/DIRECTORY/requirements.txt
```

In addition to these modules, you also need a working **swig-enabled**
[`lalapps`](http://software.ligo.org/docs/lalsuite/lalsuite/) with
  at least `lalpulsar`. A minimal confuration line to use when installing
`lalapps` is

```
$ ./configure --prefix=${HOME}/lalsuite-install --disable-all-lal --enable-lalpulsar --enable-lalapps --enable-swig
```


### `pyfstat` installation

The script can be installed system wide, assuming you are in the source directory, via
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

The scripts require a path to ephemeris files in order to use the
`lalpulsar.ComputeFstat` module. This can either be specified when initialising
each search (as one of the arguments), or simply by placing a file
`~/.pyfstat.conf` into your home directory which looks like

```
earth_ephem = '/home/<USER>/lalsuite-install/share/lalpulsar/earth00-19-DE421.dat.gz'
sun_ephem = '/home/<USER>/lalsuite-install/share/lalpulsar/sun00-19-DE421.dat.gz'
```
here, we use the default ephemeris files provided with `lalsuite`.


