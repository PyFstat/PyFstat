---
title: 'PyFstat: a python package for gravitational wave analysis with the F-statistic'
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
date: 08 January 2021
bibliography: paper.bib
---

# Summary

Gravitational waves in the sensitivity band of ground-based detectors
can be emitted by a number of astrophysical sources,
not only from binary coalescences, but also by individual spinning neutron stars.
The most promising signals from such sources,
although as of 2020 not yet detected,
are so-called 'Continuous Waves' (CWs):
long-lasting, quasi-monochromatic gravitational waves.
Many search methods have been developed and applied on
LIGO [@TheLIGOScientific:2014jea]
and Virgo [@TheVirgo:2014hva] data,
most of them based on variants of matched filtering.
See @Prix:2009oha, @Riles:2017evm, and @Sieniawska:2019hmd for reviews of the field.

The *PyFstat* package provides an interface,
built on top of the LIGO Scientific Collaboration's LALSuite library [@lalsuite],
to perform $\mathcal{F}$-statistic based CW data analysis.
The $\mathcal{F}$-statistic, first introduced by @Jaranowski:1998qm,
is a matched-filter detection statistic for CW signals
described by a set of frequency evolution parameters
(for an isolated neutron star:
its frequency, inherent spin-down, and sky location)
and maximized over its amplitude parameters.
It has been one of the standard methods for LIGO-Virgo CW searches for two decades.

*PyFstat* provides classes for various search strategies and target signals,
contained in three main submodules:
- `core` : `ComputeFstat` is the basic wrapper to LALSuite's $\mathcal{F}$-statistic algorithm,
though, like other base classes in this submodule, it should be rarely accessed by end-users.
- `grid_based_searches` : Simple search classes based on regular grids over the parameter space.
- `mcmc_based_searches` : Classes to cover small parameter space regions around
search targets or promising signal candidates from wider searches ('followup' use case)
with stochastic template placement through the ptemcee sampler.

Besides standard CW signals from isolated neutron stars, *PyFstat* can also be used
for CW signals from sources in binary systems (including the additional orbital parameters),
for CWs with a discontinuity at a pulsar glitch,
and for a class of CW-like long-duration transient signals expected e.g. from _after_ a pulsar glitch.
Specialized versions of both the grid-based and MCMC-based search classes
are provided for several of these scenarios.


Both fully-coherent and semi-coherent searches
(where the data is split into several segments for efficiency)
are covered,
and in addition to the $\mathcal{F}$-statistic,
an additional detection statistic that is more robust against single-detector noise artifacts
@Keitel:2013wga
is also supported.
However, *PyFstat* does not compete with the sophisticated
grid setups and semi-coherent algorithms implemented in various LALSuite programs.
As discussed below, the main scientific use cases for *PyFstat* at the time of publication
are for the MCMC exploration of small parameter-space regions
and for the long-duration transient case

Additional helper classes, utility functions and internals are included to
- handle the Short Fourier Transform (SFT) data format popularly used for CW searches in LIGO data;
- simulate artificial data with noise and signals in them;
- plotting.

*PyFstat* was first described in @Ashton:2018ure which remains the main reference
for the MCMC-based analysis implemented in the package.
The extension to transient signals, which uses pyCUDA for speedup,
is discussed in detail in @Keitel:2018pxz .
Most of the underlying LALSuite functionality is accessed through SWIG wrappings [@Wette:2020air]
though some parts, such as the SFT handling,
is as of the writing of this paper still called through stand-alone `lalapps` executables,
though a backend migration to pure SWIG usage is planned for the future.

The source of *PyFstat* is hosted on [GitHub](https://github.com/PyFstat/PyFstat/).
This repository also contains an automated test suite
and a set of introductory example scripts.
Issues with the software can be submitted through GitHub
and pull requests are always welcome.
Documentation in html and pdf formats is available from https://readthedocs.org/projects/pyfstat/
and installation instructions can be obtained both from there
and from the project's [README](https://github.com/PyFstat/PyFstat/blob/master/README.md) file.


# Statement of need

The sensitivity of searches for CWs and long-duration transient GWs
is generally limited by computational resources,
as the required number of matched-filter templates increases steeply
for long observation times and wide parameter spaces.

The C-based LALSuite library [@lalsuite] contains many sophisticated search methods
with a long development history and high level of optimization,
but is not very accessible for researchers new to the field or for students,
nor very convenient for rapid development and integration with modern technologies
like GPUs or machine learning.

*PyFstat* serves a dual function of
- making LALSuite CW functionality more easily accessible through a Python interface,
thus facilitating the new user experience and,
for developers, the exploratory implementation of novel methods;
- providing a set of production-ready search classes for use cases not yet covered by LALSuite itself,
most notably for MCMC-based candidate followup.

So far, *PyFstat* has been used
- for the original proposal of using MCMC for CW candidate followup [@Ashton:2018ure];
- for developing glitch-robust CW search methods [@Ashton:2018qth];
- for speeding up long-transient searches with GPUs [@Keitel:2018pxz];
- for the followup from all-sky searches for CWs from sources in binary systems,
see @Covas:2020nwy and @Abbott:2020mev.

# Acknowledgements

We acknowledge contributions to the package from Karl Wette, Sylvia Zhu and Dan Foreman-Mackey.
and helpful suggestions by John T. Whelan and Luca Rei
and the LIGO-Virgo-KAGRA Continuous Wave working group.
D.~K. and R.~T. are supported by European Union FEDER funds, the Ministry of Science, 
Innovation and Universities and the Spanish Agencia Estatal de Investigación grants
PID2019-106416GB-I00/AEI/10.13039/501100011033,
FPA2016-76821-P,
RED2018-102661-T,
RED2018-102573-E,
FPA2017-90687-REDC,
Comunitat Autonoma de les Illes Balears through the Direcció General de Política Universitaria i Recerca with funds from the Tourist Stay Tax Law ITS 2017-006 (PRD2018/24),
Generalitat Valenciana (PROMETEO/2019/071),
EU COST Actions CA18108, CA17137, CA16214, and CA16104,
and the Spanish Ministerio de Ciencia, Innovación y Universidades
(R.~T.: ref.~FPU 18/00694;
D.~K.: ref.~BEAGAL 18/00148, cofinanced by the Universitat de les Illes Balears
).

# References
