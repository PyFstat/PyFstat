# PyFstat

This is a python package providing an interface to perform F-statistic based
continuous gravitational wave (CW) searches. At its core, this is a simple
wrapper of selected tools in
['lalpulsar'](http://software.ligo.org/docs/lalsuite/lalpulsar/). The general
idea is to allow easy scripting of new search pipelines, we focus
primarily on interfacing the CW routines with
[emcee](http://dan.iel.fm/emcee/current/) a python MCMC sampler.


## Examples

We include a variety of example search scripts [here](examples), for each
example there is also a more descriptive write-up containing examples of the
output which we list below. Before running any of the search examples, be sure
to have run the [script to generate fake data](examples/make_fake_data.py).

* [Making fake data with and without glitches](docs/make_fake_data.md)
* [Fully-coherent MCMC search](docs/fully_coherent_search_using_MCMC.md)
* [Fully-coherent MCMC search on data containing a single glitch](docs/fully_coherent_search_using_MCMC_on_glitching_data.md)
* [Semi-coherent MCMC glitch-search on data containing a single glitch](docs/semi_coherent_glitch_search_using_MCMC_on_glitching_data.md)
* [Semi-coherent MCMC glitch-search on data containing two glitches](docs/semi_coherent_glitch_search_with_two_glitches_using_MCMC_on_glitching_data.md)
* [Semi-coherent Follow-Up MCMC search (dynamically changing the coherence time)](docs/follow_up.md)

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
`numpy` and `scipy` by default). Instructions to install miniconda/anaconda
are provided in the links.

### Dependencies

`pyfstat` makes uses the following external python modules:

* [numpy](http://www.numpy.org/)
* [matplotlib](http://matplotlib.org/)
* [scipy](https://www.scipy.org/)
* [emcee](http://dan.iel.fm/emcee/current/)
* [corner](https://pypi.python.org/pypi/corner/)
* [dill](https://pypi.python.org/pypi/dill)
* [tqdm](https://pypi.python.org/pypi/tqdm)(optional), if installed, this
  provides a useful progress bar and estimate of the remaining run-time.

For an introduction to installing modules see
[here](https://docs.python.org/3.5/installing/index.html). If you are using
`pip`, to install all of these modules, run
```
$ pip install -r /PATH/TO/THIS/DIRECTORY/requirements.txt
```
If you have installed python from conda then `pip` itself can be installed via
`conda install pip`.

In addition to these modules, you also need a working **swig-enabled**
[`lalapps`](http://software.ligo.org/docs/lalsuite/lalsuite/) with
  at least `lalpulsar`. A minimal confuration line to use when installing
`lalapps` is

```
$ ./configure --prefix=${HOME}/lalsuite-install --disable-all-lal --enable-lalpulsar --enable-lalapps --enable-swig
```


### `pyfstat` installation

The script can be installed system wide, assuming you are in this directory, via
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


