## 1.4.1 [13/07/2020]

 - Writer: fix SFT counting for non-contiguous or overlapping SFTs

## 1.4.0 [13/07/2020]

- now fully python3.8 compatible
- now using versioneer for versioning
- require lalsuite>=6.72
- added docker images, provided through github packages
- Writer class:
  - new options randSeed, noiseSFTs and windowing
  - change default sqrtSX from 1 to 0
  - improved support for sources in binaries
- search classes:
  - improved support for sources in binaries
  - additional file outputs for MCMC
    (including full posterior samples)
  - improved output file headers with version and options strings
  - improved MCMC walker plotting
  - implemented CFS feature injectSqrtSX
  - manual ephemerides option
  - new options estimate_covering_band and RngMedWindow
  - extended get_covering_band() to deal with fkdot bands
  - improved GridSearch logic to reuse (or not) old results
- removed injection_helper_functions module
- fixed placement of temporary output files
- improved error handling for lal programs
- added 1to1 test against CFSv2
- reorganised and improved examples
- various minor bug fixes and code cleanup

## 1.3 [21/01/2020]

- python3 migration
- enforce black style checker
- smarter ephemerides finding
- pycuda as optional dependency
- improved test suite
- improved logic in Writer class
- fixes to ComputeFstat flags handling
- various minor bug fixes

## 1.2 [08/05/2018]

- reorganised examples
- timing of transient F-stat map function

## 1.1.2 [23/04/2018] and before

- see git commit history
