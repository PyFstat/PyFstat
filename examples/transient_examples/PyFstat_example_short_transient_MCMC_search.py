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

tstart = 1000000000
duration = 2 * 86400
Tsft = 1800

inj = {
    "tref": tstart,
    "F0": 30.0,
    "F1": -1e-10,
    "F2": 0,
    "Alpha": 0.5,
    "Delta": 1,
    "transient_tstart": tstart + 0.25 * duration,
    "transient_duration": 0.5 * duration,
}

DeltaF0 = 1e-2
DeltaF1 = 1e-9

theta_prior = {
    "F0": {
        "type": "unif",
        "lower": inj["F0"] - DeltaF0 / 2.0,
        "upper": inj["F0"] + DeltaF0 / 2.0,
    },
    "F1": {
        "type": "unif",
        "lower": inj["F1"] - DeltaF1 / 2.0,
        "upper": inj["F1"] + DeltaF1 / 2.0,
    },
    "F2": inj["F2"],
    "Alpha": inj["Alpha"],
    "Delta": inj["Delta"],
    "transient_tstart": {
        "type": "unif",
        "lower": tstart,
        "upper": tstart + duration - 2 * Tsft,
    },
    "transient_duration": {
        "type": "unif",
        "lower": 2 * Tsft,
        "upper": duration - 2 * Tsft,
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
    tref=inj["tref"],
    nsteps=nsteps,
    nwalkers=nwalkers,
    ntemps=ntemps,
    log10beta_min=log10beta_min,
    transientWindowType="rect",
)
mcmc.run(walker_plot_args={"plot_det_stat": True, "injection_parameters": inj})
mcmc.print_summary()
mcmc.plot_corner(add_prior=True, truths=inj)
mcmc.plot_prior_posterior(injection_parameters=inj)
