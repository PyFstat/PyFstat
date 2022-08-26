"""
MCMC search: Semicoherent F-statistic with initialisation
=========================================================

Directed MCMC search for an isolated CW signal using the
fully-coherent F-statistic. Prior to the burn-in stage, walkers
are initialized with a certain scattering factor.
"""
import os

import numpy as np

import pyfstat

label = "PyFstat_example_MCMC_search_using_initialisation"
outdir = os.path.join("PyFstat_example_data", label)
logger = pyfstat.set_up_logger(label=label, outdir=outdir)

# Properties of the GW data
data_parameters = {
    "sqrtSX": 1e-23,
    "tstart": 1000000000,
    "duration": 100 * 86400,
    "detectors": "H1",
}
tend = data_parameters["tstart"] + data_parameters["duration"]
mid_time = 0.5 * (data_parameters["tstart"] + tend)

# Properties of the signal
depth = 10
signal_parameters = {
    "F0": 30.0,
    "F1": -1e-10,
    "F2": 0,
    "Alpha": np.radians(83.6292),
    "Delta": np.radians(22.0144),
    "tref": mid_time,
    "h0": data_parameters["sqrtSX"] / depth,
    "cosi": 1.0,
}

data = pyfstat.Writer(
    label=label, outdir=outdir, **data_parameters, **signal_parameters
)
data.make_data()

# The predicted twoF, given by lalapps_predictFstat can be accessed by
twoF = data.predict_fstat()
logger.info("Predicted twoF value: {}\n".format(twoF))

DeltaF0 = 1e-7
DeltaF1 = 1e-13
VF0 = (np.pi * data_parameters["duration"] * DeltaF0) ** 2 / 3.0
VF1 = (np.pi * data_parameters["duration"] ** 2 * DeltaF1) ** 2 * 4 / 45.0
logger.info("\nV={:1.2e}, VF0={:1.2e}, VF1={:1.2e}\n".format(VF0 * VF1, VF0, VF1))

theta_prior = {
    "F0": {
        "type": "unif",
        "lower": signal_parameters["F0"] - DeltaF0 / 2.0,
        "upper": signal_parameters["F0"] + DeltaF0 / 2.0,
    },
    "F1": {
        "type": "unif",
        "lower": signal_parameters["F1"] - DeltaF1 / 2.0,
        "upper": signal_parameters["F1"] + DeltaF1 / 2.0,
    },
}
for key in "F2", "Alpha", "Delta":
    theta_prior[key] = signal_parameters[key]

ntemps = 1
log10beta_min = -1
nwalkers = 100
nsteps = [100, 100]

mcmc = pyfstat.MCMCSearch(
    label=label,
    outdir=outdir,
    sftfilepattern=os.path.join(outdir, "*{}*sft".format(label)),
    theta_prior=theta_prior,
    tref=mid_time,
    minStartTime=data_parameters["tstart"],
    maxStartTime=tend,
    nsteps=nsteps,
    nwalkers=nwalkers,
    ntemps=ntemps,
    log10beta_min=log10beta_min,
)
mcmc.setup_initialisation(100, scatter_val=1e-10)
mcmc.run(
    walker_plot_args={"plot_det_stat": True, "injection_parameters": signal_parameters}
)
mcmc.print_summary()
mcmc.plot_corner(add_prior=True, truths=signal_parameters)
mcmc.plot_prior_posterior(injection_parameters=signal_parameters)
