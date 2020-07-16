import numpy as np
import pyfstat
from PyFstat_example_make_data_for_search_on_1_glitch import (
    tstart,
    duration,
    tref,
    F0,
    F1,
    F2,
    Alpha,
    Delta,
    outdir,
)
import os

outdir = os.path.join("PyFstat_example_data", "PyFstat_example_glitch_robust_search")
label = "standard_directed_MCMC_search_on_1_glitch"

Nstar = 10000
F0_width = np.sqrt(Nstar) * np.sqrt(12) / (np.pi * duration)
F1_width = np.sqrt(Nstar) * np.sqrt(180) / (np.pi * duration ** 2)

theta_prior = {
    "F0": {"type": "unif", "lower": F0 - F0_width / 2.0, "upper": F0 + F0_width / 2.0},
    "F1": {"type": "unif", "lower": F1 - F1_width / 2.0, "upper": F1 + F1_width / 2.0},
    "F2": F2,
    "Alpha": Alpha,
    "Delta": Delta,
}

ntemps = 2
log10beta_min = -0.5
nwalkers = 100
nsteps = [500, 2000]

mcmc = pyfstat.MCMCSearch(
    label=label,
    outdir=outdir,
    sftfilepattern=os.path.join(outdir, "*1_glitch*sft"),
    theta_prior=theta_prior,
    tref=tref,
    minStartTime=tstart,
    maxStartTime=tstart + duration,
    nsteps=nsteps,
    nwalkers=nwalkers,
    ntemps=ntemps,
    log10beta_min=log10beta_min,
)

mcmc.transform_dictionary["F0"] = dict(subtractor=F0, symbol="$f-f^\mathrm{s}$")
mcmc.transform_dictionary["F1"] = dict(
    subtractor=F1, symbol="$\dot{f}-\dot{f}^\mathrm{s}$"
)

mcmc.run()
mcmc.plot_corner()
mcmc.print_summary()
