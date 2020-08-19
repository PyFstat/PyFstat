import pyfstat
import numpy as np
import matplotlib.pyplot as plt
import os

label = os.path.splitext(os.path.basename(__file__))[0]
outdir = os.path.join("PyFstat_example_data", label)

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
depth = 40
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
print("Predicted twoF value: {}\n".format(twoF))

# Search
VF0 = VF1 = 1e5
DeltaF0 = np.sqrt(VF0) * np.sqrt(3) / (np.pi * data_parameters["duration"])
DeltaF1 = np.sqrt(VF1) * np.sqrt(180) / (np.pi * data_parameters["duration"] ** 2)
theta_prior = {
    "F0": {
        "type": "unif",
        "lower": signal_parameters["F0"] - DeltaF0 / 2.0,
        "upper": signal_parameters["F0"] + DeltaF0 / 2,
    },
    "F1": {
        "type": "unif",
        "lower": signal_parameters["F1"] - DeltaF1 / 2.0,
        "upper": signal_parameters["F1"] + DeltaF1 / 2,
    },
}
for key in "F2", "Alpha", "Delta":
    theta_prior[key] = signal_parameters[key]


ntemps = 3
log10beta_min = -0.5
nwalkers = 100
nsteps = [100, 100]

mcmc = pyfstat.MCMCFollowUpSearch(
    label=label,
    outdir=outdir,
    sftfilepattern=os.path.join(outdir, "*{}*sft".format(label)),
    theta_prior=theta_prior,
    tref=mid_time,
    minStartTime=data_parameters["tstart"],
    maxStartTime=tend,
    nwalkers=nwalkers,
    nsteps=nsteps,
    ntemps=ntemps,
    log10beta_min=log10beta_min,
)

NstarMax = 1000
Nsegs0 = 100
fig, axes = plt.subplots(nrows=2, figsize=(3.4, 3.5))
mcmc.run(
    NstarMax=NstarMax,
    Nsegs0=Nsegs0,
    walker_plot_args={
        "labelpad": 0.01,
        "plot_det_stat": False,
        "fig": fig,
        "axes": axes,
    },
)

mcmc.plot_corner(add_prior=True)
mcmc.plot_prior_posterior()
mcmc.print_summary()
