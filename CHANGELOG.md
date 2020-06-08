## target version: 1.4

- additional file outputs for MCMC search classes
  (including full posterior samples)
- improved binary support in Writer and search classes
- improved output file headers with version and options strings
- improved MCMC walker plotting
- implement CFS feature injectSqrtSX
- manual ephemerides options for search classes
- reorganised and improved examples
- fixed placement of temporary output files
- use versioneer for versioning
- various minor bug fixes and code cleanup
- now fully python3.8 compatible
- new options estimate_covering_band and RngMedWindow for search classes
- new option randSeed for Writer class
- extended get_covering_band() to deal with fkdot bands
- added 1to1 test against CFSv2
- improved error handling for lal programs

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
