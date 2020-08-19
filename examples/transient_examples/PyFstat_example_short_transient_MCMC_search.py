#!/usr/bin/env python

import pyfstat
import os
import numpy as np

outdir = os.path.join("PyFstat_example_data", "PyFstat_example_short_transient_search")
if not os.path.isdir(outdir) or not np.any(
    [f.endswith(".sft") for f in os.listdir(outdir)]
):
    raise RuntimeError(
        "Please first run PyFstat_example_make_data_for_short_transient_search.py !"
    )

F0 = 30.0
F1 = -1e-10
F2 = 0
Alpha = 0.5
Delta = 1

minStartTime = 1000000000
maxStartTime = minStartTime + 2 * 86400
Tspan = maxStartTime - minStartTime
tref = minStartTime

Tsft = 1800

DeltaF0 = 1e-2
DeltaF1 = 1e-9

theta_prior = {
    "F0": {"type": "unif", "lower": F0 - DeltaF0 / 2.0, "upper": F0 + DeltaF0 / 2.0},
    "F1": {"type": "unif", "lower": F1 - DeltaF1 / 2.0, "upper": F1 + DeltaF1 / 2.0},
    "F2": F2,
    "Alpha": Alpha,
    "Delta": Delta,
    "transient_tstart": {
        "type": "unif",
        "lower": minStartTime,
        "upper": maxStartTime - 2 * Tsft,
    },
    "transient_duration": {
        "type": "unif",
        "lower": 2 * Tsft,
        "upper": Tspan - 2 * Tsft,
    },
}

ntemps = 2
log10beta_min = -1
nwalkers = 100
nsteps = [100, 100]

mcmc = pyfstat.MCMCTransientSearch(
    label="transient_search",
    outdir=outdir,
    sftfilepattern=os.path.join(outdir, "*simulated_transient_signal*sft"),
    theta_prior=theta_prior,
    tref=tref,
    minStartTime=minStartTime,
    maxStartTime=maxStartTime,
    nsteps=nsteps,
    nwalkers=nwalkers,
    ntemps=ntemps,
    log10beta_min=log10beta_min,
    transientWindowType="rect",
    # minCoverFreq=-0.04,
    # maxCoverFreq=-0.04,
)
mcmc.run()
mcmc.plot_corner(add_prior=True)
mcmc.plot_prior_posterior()
mcmc.print_summary()
