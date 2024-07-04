## 2.1.0 [04/07/2024]

 - This is a maintenance release for updating python and numpy compatibility and fixing minor bugs.
 - Dropped python 3.8 support.
 - Enabled python 3.12 support.
 - Enabled support for numpy >=1.24.0 as long as it's still <2.0.
 - Fixed binder notebooks.
 - Fixes to some MCMC plotting methods in case of returning fig and axes (rather than saving to files).
 - Fix for transient parameters in get_predict_fstat_parameters_from_dict().
 - for developers: upgraded to black 24 style,
   refactored and updated the github actions

## 2.0.2 [12/10/2023]

 - Pinned to python<3.12 until more dependencies are updated
   and a few things fixed on our end.

## 2.0.1 [11/10/2023]

 - Fixed compatibility with matplotlib=3.8 and bumped minimum requirement to >=3.3.
 - Fixed pulling in ephemerides via lalpulsar optional dependency.
 - Fixed line simulation in "generating noise" tutorial.
 - Fixed some type checks to conform to flake8>=1.7 standards.
 - PyPI uploads now done using "trusted publishing" (OIDC).

## 2.0.0 [18/04/2023]

 - This is a major release of PyFstat in the sense that
   users will likely have to make some changes to the way they call it,
   but mostly just regarding class instance labels / file naming.
 - Mainly, we follow the recent `lalpulsar` upgrade to v3 of the SFT file format specification
   (see https://dcc.ligo.org/T040164-v2/public
    and note the v2 in the URL is not a typo,
    this file version describes both the v2 and v3 specifications).
   - We now require `lalsuite>=7.13`.
   - The file format update itself is fully backwards compatible:
     the only change is including window information in the header,
     which however reuses existing padding bytes, and hence does not affect compatibility.
   - The file naming convention however has become more restrictive:
     SFT files, and hence `label` arguments for `Writer` and derived classes,
     now may only contain ASCII alphanumerical characters,
     which specifically makes the old PyFstat habit of using underscores illegal.
     We suggest CamelCase instead.
   - Instead of `SFTWindowBeta`, one must now use `SFTWindowParam`.
   - `noiseSFTs` no longer requires `SFTWindow[Beta/Param]`,
     only if a window specification cannot be read from the headers of input SFTs.
 - Fixed an error that prevented one parameter to be printed in search output file header comments.
 - Fixed compatibility with `corner=2.2.2`.
 - Removed `utils.get_lal_exec()`, now always expect executables to be named `lalpulsar_`.
   (Old LALSuite versions with `lalapps_` CW executables are quite outdated by now.)
 - Removed deprecated prior formats in `InjectionParametersGenerator` class.
 - Transient F-stat GPU implementation:
   - Introduced F-stat condition number check,
     with threshold equivalent to defaults of
     `XLALComputeAntennaPatternSqrtDeterminant()`
     and `estimateAntennaPatternConditionNumber()`.
   - Now falls back to F=2 (2F=4) if Ddinv=0,
     also equivalent to `lalpulsar`.
   - Added unit tests.
   - Improved CUDA device info logging.
 - For developers: updated coding style to `black` 23.1.0 rules
   (mostly newlines policy).

## 1.19.1 [19/12/2022]

 - Pinned to `numpy<1.24.0` to avoid incompatibility with `ptemcee`.

## 1.19.0 [01/12/2022]

 - This is the first PyFstat release to officially support python 3.11.
 - LALSuite is introducing an SFT specification and filename update
   - see https://dcc.ligo.org/T040164-v2/public)
   - This version of PyFstat is pinned to `lalsuite<=7.11` (or `lalpulsar<6.0`)
     so that it is ensured to keep working with the old convention.
   - Next PyFstat release will adapt to the LALSuite changes.
   - Added `get_official_sft_filename()` utility function to ease migration.
 - Improvements to `injection_parameters` module with new priors logic
   (see documentation);
   old input style still supported for now but deprecated.
 - `Writer`: improved timestamps handling:
   support 1-column format (without nanoseconds),
   which has been the LALPulsar default for a while.
 - Changed `ComputeFstat.write_atoms_to_file()` method
   to use consistent `%%` comment markers.
 - Fixed segfault in `get_semicoherent_single_IFO_twoFs()` with recent LALSuite
   (need to properly copy FstatAtomVector struct).
 - Fixed `TransientGridSearch` when not setting any actual transient parameters.
 - Added (h0,cosi)<->(aplus,across) functions to `utils.converting`.
 - Added `utils.atom` submodule with tools related to F-stat atoms.
 - `utils.get_sft_as_arrays()` now user-accessible
 - `Writer`: warning about inferring parameters (in `noiseSFTs is not None` case)
   downgraded to info message
 - improvements to documentation and test suite

## 1.18.1 [03/10/2022]

 - fixed finalizer setup for calling `ComputeFstat` and its children
   in context-manager mode
 - reverted "walrus operators" (`:=`) and use of `Literal` typing checks
   to facilitate `python 3.7` backwards compatibility
   (not officially supported from release, but can be easily restored on branches / local clones)
 - lalsuite-from-source install instructions no longer included in README, now live on wiki
 - fixed logging use in examples
 - improved testing of tutorial notebooks

## 1.18.0 [06/09/2022]

 - refurbished logging system:
   - on `import.pyfstat`, stdout logging at INFO level is activated
     unless there are already handlers attached to the root logger
   - recommended to further call `pyfstat.set_up_logger` and define an output log file,
     as demonstrated in our examples
   - see https://pyfstat.readthedocs.io/en/v1.18.0/pyfstat.html#module-pyfstat.logging for details
   - removed the last global argparse options `--quite` and `--verbose`,
     along with the remainders of the `helper_functions.set_up_command_line_arguments()` function.
     Please use `pyfstat.set_up_logger` instead to determine verbosity level.
   - tests now by default print all >=WARNING messages even for passing cases
     and full >=INFO for failing cases
   - improved handling of LALSuite executables through `run_commandline`
     - better stderr/stdout capture
     - return is now either a `subprocess.CompletedProcess` object
       or `None`
     - default is now `return_output=False`
     - `log_level` argument removed
       (please use `pyfstat.set_up_logger` instead)
   - capture and handling of output from SWIG-wrapped LALSuite functions
     likely to be further improved in future versions
 - refactored `helper_functions` module into `utils` subpackage
   with multiple source files;
   - user can access all functions directly as `utils.some_function`
     without worrying about the level one further down
   - moved `matplotlib` setup into new `utils.safe_X_less_plt()`
   - removed deprecated/unused helper functions
        - `get_peak_values`
        - `get_comb_values`
        - `get_sft_array`
 - can install with `NO_LALSUITE_FROM_PYPI` environment variable,
   e.g. to avoid duplication of dependencies from conda and pip
   (now used this way in recommended `pyfstat-dev.yml`)
 - removed `peakutils` dependency
 - optional `[dev]` set of dependencies now also includes `docs` dependencies
 - added `sphinx_autodoc_typehints` to  `docs` dependencies
   and updated pinned versions of other sphinx packages
 - `DetectorStates`: fixed passing plain lists as values of a `timestamps` dict

## 1.17.0 [26/08/2022]

 - dropped python 3.7 support
 - dropped dependency on `lalapps` and now requiring `lalpulsar>=5.0.0` instead,
   where LALSuite executables now live
   (corresponding to `lalsuite>=7.7`)
   - there is a `get_lal_exec()` helper function to still allow running
     on old installations where the executables live in `lalapps`
 - removed most of the old package-level hardcoded `argparse` options,
   this should reduce conflicts when users import it in their own caller scripts:
   - `--clean` to be given as an argument to supporting classes
   - `-N` to be given as class argument `num_threads`
     to the one class `FrequencyModulatedArtifactWriter` that supported it
   - `--setup-only` and `--no-template-counting` to be given as class arguments
     to the (now deprecated) `MCMCFollowUpSearch`
   - `--no-interactive` was not supported anywhere
 - LaTeX no longer enabled for plotting by default
 - removed `bashplotlib` dependency and SFT timestamps ASCII art
 - removed fallbacks in case `tqdm` is not available,
   which is a dependency anyway
 - `snr` class: fixed behaviour when instantiating with a dictionary of timestamps
 - removed old backwards compatibility code from `FrequencyModulatedArtifactWriter`
 - developers: updated pre-commit hooks

## 1.16.0 [25/07/2022]

 - include local versions of autocorr functions from `ptemcee`
   to restore compatibility with `numpy>=1.23.0`
 - `Writer`: remove deprecated `timestampsFiles` option,
   please use the more general `timestamps` instead
 - added `BtSG` option to transient searches
   (transient Bayes factor from PGM2011)
 - `pyTransientFstatMap` is now available via `import pyfstat`
 - some refactoring of internal detection statistics functions
 - use of `lalapps_tconvert` replaced by new `gps_to_datestr_utc()` helper function
   (which uses `XLALGPSToUTC` and `datetime`)
 - fix CUDA context detaching at garbage collection time for
   `ComputeFstat` and `TransientGridSearch` classes
   - using `weakref`
   - should not require any caller code changes in standard use cases
   - but if user wants to initiate more than one such object from one session/script,
     these should be used in context manager style (`with ComputeFstat as` etc)
 - remove redundant complex entries from output of `tcw_fstat_map_funcs.reshape_FstatAtomsVector()`
 - streamline installation instructions for developers
   and remove single-use helper bash scripts

## 1.15.0 [27/06/2022]

 - pin `numpy<1.23.0` to work around `ptemcee` incompatibility
 - `pyfstat.__version__` now reported with the leading `v` stripped out
   (e.g. just `1.15.0`)
 - added new method `compute_h0_from_snr2()` to `SignalToNoiseRatio` class
 - `init_transient_fstat_map_features()` now is stricter about feature name strings
 - added tutorial notebooks to binder
 - improved installation instructions

## 1.14.1 [12/05/2022]

 - fixed `phi0` argument name to `phi`
   in `SignalToNoiseRatio.compute_snr2()` function
   to match conventions elsewhere
 - fixed syntax error in `Writer`
   when both timestamps and detector names are given
 - added new set of intro notebooks
   in `examples/tutorials`

## 1.14.0 [31/03/2022]

 - new dependencies: `attr` (core), `flaky` (for tests only)
 - removed deprecated option to use `$LALPULSAR_DATADIR` for ephemerides
 - added `SignalToNoiseRatio` class (equivalent to `lalapps_PredictFstat`)
   and `DetectorStates` helper class in new `snr` module
 - ` InjectionParametersGenerator` and its children moved
   to separate module `injection_parameters`
 - `Writer` (and its children) now accepts `timestamps`
   as a list, dict over detectors,
   or comma-separated string of files per detector
   (`timestampsFiles` input is now deprecated)
 - some simplifications to `Writer` internal methods
 - for developers:
   - test suite now split up by module,
     full suite can now be run with `pytest tests/`
   - flaky MCMC tests will be rerun 3 times if needed
   - now enforcing `isort` import ordering style
     and some other simple pre-commit-hook rules

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
