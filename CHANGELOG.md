## 1.7.0 [31/08/2020]

 - Writer: fix binary tp handling and clean up injection parameter parsing
 - MCMCSearch plotting improvements:
   - add injection parameters through "truths" kwarg
   - F->2F fixes
   - fix failures for single-parameter posteriors
 - removed unused Bunch class
 - refactored some core functions into BaseSearchClass methods
   or moved to helper_functions
 - removed deprecated options/functions:
   - ComputeFstat option estimate_covering_band
   - Writer options minStartTime, maxStartTime
   - MCMCSearch.get_median_stds()
 - new internal class SearchForSignalWithJump
   which SemiCoherentGlitchSearch and GlitchWriter inherit from

## 1.6.0 [19/08/2020]

 - Writer classes, including several backwards-incompatbile changes:
   - removed lots of default values where there isn't really
     a logical "default"; e.g. detectors and signal parameters
     now always need to be explicitly set.
   - more flexible setup getting info from noiseSFTs
     (tstart, duration now optional and used as constraints)
   - can now auto-estimate SFT frequency band if not set by user
     (assuming a single-template search)
   - added manual ephemerides options
   - removed add_noise option, same behaviour can still
     be controlled through options sqrtSX or noiseSFTs
   - no .cff file generated any more if h0==0 (no signal)
   - verbose option for make_cff() and make_data() methods
   - clearer error messages in many cases
 - ComputeFstat, SemiCoherentSearch and derived classes:
   - fixed internal maxStartTime default if not set by user
   - by that, fixed the SemiCoherentSearch segments auto-setup
   - added Tsft as user option
   - clearer error and logging messages
 - improved helper_functions.get_sft_array()
 - extended, cleaned up and further modularised test suite
 - updated examples to changes in Writer and other classes
 
## 1.5.2 [06/08/2020]

 - fixed semi-coherent search bug introduced in 1.5.0:
    - last segment was always skipped due to off-by-one error
      in segment boundary calculation.
 - MCMC searches:
    - extended print_summary()
      with new get_summary_stats() helper
      and deprecated get_median_stds()
    - fixes to some of the more exotic prior types.
 - Extended MCMC test coverage.
   
## 1.5.1 [30/07/2020]

 - The only change in this release is an updated README
   to point to the new 1.5+ Zenodo record.

## 1.5.0 [30/07/2020]

 - new default coverage band behaviour for all search classes:
   - estimate from search ranges (GridSearch) or prior (MCMCSearch)
     unless minCoverFreq, maxCoverFreq set
   - negative values can be used to reproduce old default
     of setting from SFT width
   - explicit option estimate_covering_band deprecated
 - semicoherent searches:
   - sped up by only calling ComputeTransientFstatMap once per point
   - BSGL now computed from summed F-stats, not for each segment
   - per-segment results now stored in attribute twoF_per_segment
     instead of det_stat_per_segment
 - MCMC searches: save twoF for each sample to .dat file
 - Writer:
   - options minStartTime, maxStartTime deprecated
   - always use tstart, duration for actual data range
   - and use transientStartTime, transientTau for transients
 - transient-on-GPU output file writing fix
 - examples:
   - all output now goes to a directory "PyFstat_example_data"
   - added mcmc_vs_grid_simple_example

## 1.4.2 [14/07/2020]

 - small fixes to search classes:
   - get_max_twoF() fixed for TransientGridSearch
   - fix column header format for per-Doppler-point transient Fmn output files
   - fixed regexp deprecation warning
   - throw warning if using MCMCSearch.generate_loudest() called for transients

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
