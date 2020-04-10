import pyfstat
import numpy as np
import os

label = os.path.splitext(os.path.basename(__file__))[0]
outdir = os.path.join("example_data", label)

# Properties of the GW data
data_parameters = {
    "sqrtSX": 1e-23,
    "tstart": 1000000000,
    "duration": 100 * 86400,
}
tend = data_parameters["tstart"] + data_parameters["duration"]
mid_time = 0.5 * (data_parameters["tstart"] + tend)

# Properties of the signal
depth = 0.1
signal_parameters = {
    "F0": 30.0,
    "F1": 0,
    "F2": 0,
    "Alpha": 0.15,
    "Delta": 0.15,
    "orbitTp": mid_time,
    "orbitArgp": 0.0,
    "orbitasini": 10.0,
    "orbitEcc": 1e-3,
    "orbitPeriod": 45 * 24 * 3600.0,
    "tref": mid_time,
    "h0": data_parameters["sqrtSX"] / depth,
}

data = pyfstat.BinaryModulatedWriter(
    label=label, outdir=outdir, **data_parameters, **signal_parameters
)
data.make_data()

DeltaF0 = 1e-7
DeltaF1 = 1e-13
VF0 = (np.pi * data_parameters["duration"] * DeltaF0) ** 2 / 3.0
VF1 = (np.pi * data_parameters["duration"] ** 2 * DeltaF1) ** 2 * 4 / 45.0
print("\nV={:1.2e}, VF0={:1.2e}, VF1={:1.2e}\n".format(VF0 * VF1, VF0, VF1))

theta_prior = {
    "F0": {
        "type": "unif",
        "lower": signal_parameters["F0"] - DeltaF0 / 2.0,
        "upper": signal_parameters["F0"] + DeltaF0 / 2,
    },
    "F1": signal_parameters["F1"],
    "F2": signal_parameters["F2"],
    "Alpha": {"type": "unif", "lower": 0.1, "upper": 0.2},
    "Delta": {"type": "unif", "lower": 0.1, "upper": 0.2},
    "asini": {"type": "unif", "lower": 5, "upper": 15},
    "period": {"type": "unif", "lower": 40 * 24 * 3600.0, "upper": 50 * 24 * 3600.0},
    "ecc": {"type": "unif", "lower": 0.0, "upper": 1e-2},
    "tp": {"type": "unif", "lower": 0.9 * mid_time, "upper": 1.1 * mid_time,},
    "argp": {"type": "unif", "lower": 0.0, "upper": 2 * np.pi},
}

ntemps = 1
log10beta_min = -1
nwalkers = 100
nsteps = [300, 300]

mcmc = pyfstat.MCMCSemiCoherentSearch(
    label=label,
    outdir=outdir,
    nsegs=10,
    sftfilepattern=os.path.join(outdir, "*{}*sft".format(label)),
    theta_prior=theta_prior,
    tref=signal_parameters["tref"],
    minStartTime=data_parameters["tstart"],
    maxStartTime=tend,
    nsteps=nsteps,
    nwalkers=nwalkers,
    ntemps=ntemps,
    log10beta_min=log10beta_min,
    binary=True,
)
mcmc.run()
mcmc.plot_corner(add_prior=True)
mcmc.print_summary()
