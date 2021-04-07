---
title: 'PyFstat: a Python package for continuous gravitational-wave data analysis'
tags:
  - Python
  - gravitational waves
  - continuous waves
  - pulsars
  - data analysis
authors:
  - name: David Keitel
    orcid: 0000-0002-2824-626X
    affiliation: "1"
  - name: Rodrigo Tenorio
    orcid: 0000-0002-3582-2587
    affiliation: "1"
  - name: Gregory Ashton
    orcid: 0000-0001-7288-2231
    affiliation: "2"
  - name: Reinhard Prix
    orcid: 0000-0002-3789-6424
    affiliation: "3, 4"
affiliations:
  - name: Departament de Física, Institut d'Aplicacions Computacionals i de Codi Comunitari (IAC3), Universitat de les Illes Balears, and Institut d'Estudis Espacials de Catalunya (IEEC), Crta. Valldemossa km 7.5, E-07122 Palma, Spain
    index: 1
  - name: OzGrav, School of Physics & Astronomy, Monash University, Clayton 3800, Victoria, Australia
    index: 2
  - name: Max-Planck-Institut für Gravitationsphysik (Albert-Einstein-Institut), D-30167 Hannover, Germany
    index: 3
  - name: Leibniz Universität Hannover, D-30167 Hannover, Germany
    index: 4
date: 06 April 2021
bibliography: paper.bib
---

# Summary

Gravitational waves in the sensitivity band of ground-based detectors
can be emitted by a number of astrophysical sources,
including not only binary coalescences, but also individual spinning neutron stars.
The most promising signals from such sources,
although as of 2020 not yet detected,
are the long-lasting, quasi-monochromatic 'Continuous Waves' (CWs).
Many search methods have been developed and applied on
LIGO [@TheLIGOScientific:2014jea]
and Virgo [@TheVirgo:2014hva] data.
See @Prix:2009oha, @Riles:2017evm, and @Sieniawska:2019hmd for reviews of the field.

The `PyFstat` package provides tools
to perform a range of CW data analysis tasks.
It revolves around the $\mathcal{F}$-statistic,
first introduced by @Jaranowski:1998qm:
a matched-filter detection statistic for CW signals
described by a set of frequency evolution parameters
and maximized over amplitude parameters.
This has been one of the standard methods for LIGO-Virgo CW searches for two decades.
`PyFstat` is built on top of established routines
in `LALSuite` [@lalsuite]
but through its more modern `Python` interface
it enables a flexible approach to designing new search strategies.

Classes for various search strategies and target signals
are contained in three main submodules:

- `core`: The basic wrappers to `LALSuite`'s $\mathcal{F}$-statistic algorithm.
End-users should rarely need to access these directly.
- `grid_based_searches`: Classes to search over regular parameter-space grids.
- `mcmc_based_searches`: Classes to cover promising parameter-space regions
through stochastic template placement with the Markov Chain Monte Carlo (MCMC) sampler `ptemcee` [@Vousden:2015pte].

Besides standard CWs from isolated neutron stars, `PyFstat` can also be used
to search for CWs from sources in binary systems (including the additional orbital parameters),
for CWs with a discontinuity at a pulsar glitch,
and for CW-like long-duration transient signals, e.g., from _after_ a pulsar glitch.
Specialized versions of both grid-based and MCMC-based search classes
are provided for these scenarios.
Both fully-coherent and semi-coherent searches
(where the data is split into several segments for efficiency)
are covered,
and an extension to the $\mathcal{F}$-statistic
that is more robust against single-detector noise artifacts
[@Keitel:2013wga]
is also supported.
While `PyFstat`'s grid-based searches do not compete with the sophisticated
grid setups and semi-coherent algorithms implemented in various `LALSuite` programs,
its main scientific use cases so far are for the MCMC exploration
of interesting parameter-space regions
and for the long-duration transient case.

`PyFstat` was first introduced in @Ashton:2018ure, which remains the main reference
for the MCMC-based analysis implemented in the package.
The extension to transient signals, which uses `PyCUDA` [@Kloeckner:2012pyc] for speedup,
is discussed in detail in @Keitel:2018pxz,
and the glitch-robust search approaches in @Ashton:2018qth.

Additional helper classes, utility functions, and internals are included for
handling the common Short Fourier Transform (SFT) data format for LIGO data,
simulating artificial data with noise and signals in them,
and plotting results and diagnostics.
Most of the underlying `LALSuite` functionality is accessed through SWIG wrappings [@Wette:2020air]
though for some parts, such as the SFT handling,
we still (as of the writing of this paper) call stand-alone `lalapps` executables.
Completing the backend migration to pure SWIG usage is planned for the future.

The source of `PyFstat` is hosted on [GitHub](https://github.com/PyFstat/PyFstat/).
The repository also contains an automated test suite
and a set of introductory example scripts.
Issues with the software can be submitted through GitHub
and pull requests are always welcome.
`PyFstat` can be installed through pip, conda or docker containers.
Documentation in html and pdf formats is available from [readthedocs.org](https://readthedocs.org/projects/pyfstat/)
and installation instructions can be found there
or in the [README](https://github.com/PyFstat/PyFstat/blob/master/README.md) file.
PyFstat is also listed in the Astrophysics Source Code Library as [ascl:2102.027](https://ascl.net/2102.027).


# Statement of need

The sensitivity of searches for CWs and long-duration transient GWs
is generally limited by computational resources,
as the required number of matched-filter templates increases steeply
for long observation times and wide parameter spaces.
The C-based `LALSuite` library [@lalsuite] contains many sophisticated search methods
with a long development history and high level of optimization,
but is not very accessible for researchers new to the field or for students;
nor is it convenient for rapid development and integration with modern technologies
like GPUs or machine learning.
Hence, `PyFstat` serves a dual function of
(i) making `LALSuite` CW functionality more easily accessible through a `Python` interface,
thus facilitating the new user experience and,
for developers, the exploratory implementation of novel methods;
and (ii) providing a set of production-ready search classes for use cases not yet covered by `LALSuite` itself,
most notably for MCMC-based followup of promising candidates from wide-parameter-space searches.

So far, `PyFstat` has been used for

- the original proposal of MCMC followup for CW candidates [@Ashton:2018ure];
- developing glitch-robust CW search methods [@Ashton:2018qth];
- speeding up long-transient searches with GPUs [@Keitel:2018pxz];
- followup of candidates from all-sky searches for CWs from sources in binary systems,
see @Covas:2020nwy and @Abbott:2020mev;
- studying the impact of neutron star proper motions on CW searches [@Covas:2020hcy].

# Acknowledgements

We acknowledge contributions to the package from Karl Wette, Sylvia Zhu and Dan Foreman-Mackey;
as well as helpful suggestions by John T. Whelan, Luca Rei,
and the LIGO-Virgo-KAGRA Continuous Wave working group.
D.K. and R.T. are supported by European Union FEDER funds;
the Spanish Ministerio de Ciencia, Innovación y Universidades and Agencia Estatal de Investigación grants
PID2019-106416GB-I00/AEI/10.13039/501100011033,
RED2018-102661-T,
RED2018-102573-E,
FPA2017-90687-REDC,
FPU 18/00694,
and BEAGAL 18/00148 (cofinanced by the Universitat de les Illes Balears);
the Comunitat Autonoma de les Illes Balears
through the Direcció General de Política Universitaria i Recerca with funds from the Tourist Stay Tax Law ITS 2017-006 (PRD2018/24)
and the Conselleria de Fons Europeus, Universitat i Cultura;
the Generalitat Valenciana (PROMETEO/2019/071);
and
EU COST Actions CA18108, CA17137, CA16214, and CA16104.
This paper has been assigned document number LIGO-P2100008.

# References
