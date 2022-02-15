## 1.13.1 [15/02/2022]

 - add new helper function `get_sft_as_arrays()`
 - deprecate `get_sft_array()`
 - the new one supports multiple IFOs
   and returns full complex amplitude info
 - note that the order of returned values is different

## 1.13.0 [01/02/2022]

 - now tested for python 3.10
 - simpler handling of ephemerides files:
   - lalsuite >= 7.2.0 (lalpulsar >= 3.1.1)
     now includes a sufficient minimal set by default
     and has gotten much better at resolving default paths
   - no manual setup should be required any more for `pip` installs
   - remove download script
   - deprecate using `$LALPULSAR_DATADIR`
 - bump LALSuite version requirements correspondingly
 - `Writer` classes: fix bug with spurious expected filename mismatch
   if given `noiseSFTs` including multiple frequency-segmented SFTs
 - `GridSearch` now has `generate_loudest()` method like MCMC classes
 - fix some deprecation warnings

## 1.12.1 [04/12/2021]

 - add backwards compatibility workaround
   for lalapps 7.3.0 in `FrequencyModulatedArtifactWriter`
   to fix running with conda dependencies

## 1.12.0 [03/12/2021]

 - drop python 3.6 support
 - require LALSuite >= 7.1
 - no longer pinning numpy
 - resolved various deprecation warnings
 - follow updated "narrowband" SFT file name convention
   from lalapps_splitSFTs
 - new `timestampsFiles` option for `Writer` classes
 - new `allowedMismatchFromSFTLength` option
   for core and MCMC classes
 - new `singleFstats` option for `ComputeFstat` and derived classes
 - new `randSeed` option for `ComputeFstat` and derived classes
   for reproducible on-the-fly Gaussian noise generation
 - fix data-from-disk reuse in `LineWriter`
   and some internal cleanup to `Writer` and derived classes in general

## 1.11.6 [14/04/2021]

 - new reference paper for PyFstat: https://doi.org/10.21105/joss.03000
 - all requirements are now handled through setup.py,
   e.g. instead of `pip install -r requirements.txt`
   and manually installing optional dependencies,
   just use `pip install pyfstat[optionalpackage]`
 - improved formatting of MCMC corner plots
 - extended cumulative 2F plots for the transient case
 - ComputeFstat.get_semicoherent_twoF() now returns its value
 - minor internal cleanups
 - test coverage improvements

## 1.11.5 [02/04/2021]

 - PyPI source tarball for v1.11.4 didn't include files needed to build conda package
 - improved codecov setup
 - no actual changes to package

## 1.11.4 [31/03/2021]

 - python 3.9 now supported
 - improvements to documentation and examples
 - for developers: flake8-docstrings and flake8-executable rules now enforced
 - added a codemeta.json file
 - started tracking test coverage with codecov

## 1.11.3 [16/02/2021]

 - added LineWriter class for simulating unmodulated noise artifacts

## 1.11.2 [12/02/2021]

 - pinned numpy dependency to <1.20 to fix incompatibility with lalsuite 6.81
 - updated ephemerides instructions and citation requests in README
 - fixed FrequencyModulatedArtifactWriter class

## 1.11.1 [26/01/2021]

 - matplotlib will no longer be automatically forced to `agg`,
   but only if `env["DISPLAY"]` is not set
 - improved default scaling of MCMC corner plots
 - fixed missing unit for `tp` binary parameter in MCMC default plot labels

## 1.11.0 [20/01/2021]

 - LALSuite >= 6.80 now required
 - simplified calls to F-stat prediction
 - Writer:
   - internal cleanup of sftfilenames vs sftfilepath
   - SFTs are validated at end of `run_makefakedata()`
 - core and search classes: improved and fixed non-standard detection statistics
   - fixed basic twoFX and BSGL computation logic for transient and semicoherent cases, generalized to Ndet>2
   - reorganised methods for these computations
   - additional results are stored as class attributes
   - for transients, the detection statistic is now called explicitly `maxTwoF`
 - grid classes now initiate their `.search` object at instantiation instead of at first `run()` call

## 1.10.1 [20/01/2021]

 - bugfix for `MCMCSearch.plot_prior_posterior()`

## 1.10.0 [08/01/2021]

 - documentation now available from https://pyfstat.readthedocs.io
   - and examples can be run on https://mybinder.org/v2/gh/PyFstat/PyFstat/master
   - still being improved
 - added `gridcorner` module for plotting GridSearch results
 - removed the specialist grid-based classes which were deprecated in 1.9.0
 - internal changes to search classes
   - should ideally not change anything for standard CW use cases
   - but fix some corner cases and make for more robustness
   - made tstart,tend optional in `get_fullycoherent_twoF()`
   - adapted most high-level search classes to this change
   - grid searches now internally use named-column ndarrays
   - some cleanup to keys storage in MCMC classes
   - changed `helper_functions.get_doppler_params_output_format()` to return dict
   - fixed sorting of output fmt specifiers for both MCMC and grid classes
   - changed `GridSearch.inititate_search_object()` to internal method `_initiate_search_object()`
 - line-robust statistics are now always stored and returned as log10BSGL
   (for consistency with LALSuite)
 - made various class methods private that had no obvious end-user use case
 - `MCMCSearch.run()`: initiate search object before checking for old data
 - `MCMCGlitchSearch.plot_cumulative_max()`: add savefig option (defaults to false)
 - `GridSearch.plot_2D()`: renamend `save` option to `savefig`
   for consistency with other plotting functions
 - removed unused `ComputeFstat.get_full_CFSv2_output()`
 - removed unused helper functions:
   - `compute_P_twoFstarcheck()`
   - `compute_pstar()`
   - `twoFDMoffThreshold()`
 - new example `other_examples/PyFstat_example_spectrogram.py`
 - some improvements to tests and examples
 - KNOWN ISSUES: implementation of line-robust statistic BSGL will need to be overhauled

## 1.9.0 [30/10/2020]

 - new class `InjectionParametersGenerator`
   - draws dicts of parameters from arbitrary priors
   - can then be directly passed e.g. to Writer as `**params`
   - and derived `AllSkyInjectionParametersGenerator`
 - deprecate various grid-based specialist classes:
   - these were all more or less unmaintained since a long time
     - SliceGridSearch
     - GridUniformPriorSearch
     - SlidingWindow
     - FrequencySlidingWindow
     - EarthTest
     - DMoff_NO_SPIN
   - will be removed in next version unless users speak up

## 1.8.0 [13/10/2020]

 - big overhaul of cumulative twoF calculations and plotting
 - Writer: fix expected sftfilepattern in multi-IFO and custom-TSFT cases
 - fixed internal consistency of parameter names:
   - Freq->F0 and phi0->phi for all PyFstat classes/functions
   - consistent conversion to lalappas arguments
 - fixed parsing of injectSqrtSX and assumeSqrtSX arguments
 - helper_functions.predict_fstat() no longer takes `**kwargs`
 - new instructions and helpful scripts for setting up developer environments (venv or conda)
 - various improvements to test suite

## 1.7.3 [21/09/2020]

 - GridSearch: will now always include the end point in each 1D parameter points array
 - restored MCMCSearch.plot_chainconsumer() to a workable state
   - optional dependency checked and documented more cleanly
   - the injections parameter option for this is actually called "truth" not "truths
 - fixed checks of injection parameter keys for other MCMC plotting functions

## 1.7.2 [18/09/2020]

 - improved GridSearch.check_old_data_is_okay_to_use()
 - minor logging improvements for Writer, GridSearch, TransientGridSearch
 - new binary_mcmc_vs_grid example
 - minor improvements to tests and other examples

## 1.7.1 [16/09/2020]

 - LALSuite 6.76 now required
 - predict_fstat() helper function made more flexible
 - Writer: fixed check_cached_data_okay_to_use() for multiple IFOs
 - MCMCSearch and derived classes now store sampler as an attribute
 - fixes to MCMC plotting
   - proper scaling of injection parameters in walker plots
   - proper display of prior (not logprior) in prior_posterior comparisons
 - consistently close figure objects after saving
 - all classes now announce their creation to the logger
 - flake8 compliance
 - code style now complying to stricter black 20.8b standards
 - improvements to examples

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
