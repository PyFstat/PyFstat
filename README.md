# PyFstat

This is a python package containing basic wrappers of the `lalpulsar` module
with capabilities to perform a variety of searches, primarily focussing on
semi-coherent glitch searches.

## Examples

All examples can be run from their source scripts in [examples](examples), or
for each example there is descriptive documentation:

* [Making fake data with and without glitches](docs/make_fake_data.md)
* [Fully coherent searches MCMC](docs/fully_coherent_search.md)

## Installation

The script can be installed system wide via
```
python setup.py install
```
or simply add this directroy to your python path

### Ephemeris installation

The scripts require a path to ephemeris files in order to use the
`lalpulsar.ComputeFstat` module. This can either be specified when initialsing
each search, or more simply by playing a file `~/pyfstat.conf` in your home
directory which looks like

```
earth_ephem = '/home/<USER>/lalsuite-install/share/lalpulsar/earth00-19-DE421.dat.gz'
sun_ephem = '/home/<USER>/lalsuite-install/share/lalpulsar/sun00-19-DE421.dat.gz'
```

where this uses the default ephemeris files provided with `lalsuite`.

### Dependencies

* swig-enabled lalpulsar, a minimal configuration is given by

```
./configure --prefix=${HOME}/lalsuite-install --disable-all-lal --enable-lalpulsar --enable-lalapps --enable-swig
```

* [emcee](http://dan.iel.fm/emcee/current/)[^1]
* [corner](https://pypi.python.org/pypi/corner/)[^1]
* [dill](https://pypi.python.org/pypi/dill)[^1]

[^1]: Most easily installed using either `conda` or `pip`

