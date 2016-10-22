# PyFstat

This is a python package containing basic wrappers of the `lalpulsar` module
with capabilities to perform a variety of searches, primarily focusing on
semi-coherent glitch searches.

## Examples

We include a variety of example search scripts [here](examples), for each
example there is also a more descriptive write-up containing examples of the
output which we list below. Before running any of the search examples, be sure
to have run the [script to generate fake data](examples/make_fake_data.py).

* [Making fake data with and without glitches](docs/make_fake_data.md)
* [Fully coherent MCMC search](docs/fully_coherent_search_using_MCMC.md)
* [Fully coherent MCMC search on data containing glitching signals](docs/fully_coherent_search_using_MCMC_on_glitching_data.md)

## Installation

The script can be installed system wide via
```
$ python setup.py install
```
or simply add this directory to your python path. To check that the installation
was successful, run
```
$ python -c 'import pyfstat'
```
if no error message is output, then you have installed `pyfstat`.

### Ephemeris installation

The scripts require a path to ephemeris files in order to use the
`lalpulsar.ComputeFstat` module. This can either be specified when initialising
each search, or more simply by placing a file `~/pyfstat.conf` into your home
directory which looks like

```
earth_ephem = '/home/<USER>/lalsuite-install/share/lalpulsar/earth00-19-DE421.dat.gz'
sun_ephem = '/home/<USER>/lalsuite-install/share/lalpulsar/sun00-19-DE421.dat.gz'
```
here, we use the default ephemeris files provided with `lalsuite`.

### Dependencies

* swig-enabled lalpulsar, a minimal configuration is given by

```
$ ./configure --prefix=${HOME}/lalsuite-install --disable-all-lal --enable-lalpulsar --enable-lalapps --enable-swig
```

* [emcee](http://dan.iel.fm/emcee/current/)[^1]
* [corner](https://pypi.python.org/pypi/corner/)[^1]
* [dill](https://pypi.python.org/pypi/dill)[^1]
* [tqdm](https://pypi.python.org/pypi/tqdm)[^1] (optional), if installed, this
  provides a useful progress bar and estimate of the remaining run-time.

[^1]: Most easily installed using either `conda` or `pip`.

