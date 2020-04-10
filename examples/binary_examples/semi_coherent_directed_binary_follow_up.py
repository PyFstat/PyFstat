import pyfstat
import numpy as np
import matplotlib.pyplot as plt
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

ntemps = 3
log10beta_min = -0.5
nwalkers = 100
nsteps = [100, 100]

mcmc = pyfstat.MCMCFollowUpSearch(
    label=label,
    outdir=outdir,
    sftfilepattern=os.path.join(outdir, "*{}*sft".format(label)),
    theta_prior=theta_prior,
    tref=signal_parameters["tref"],
    minStartTime=data_parameters["tstart"],
    maxStartTime=tend,
    nwalkers=nwalkers,
    nsteps=nsteps,
    ntemps=ntemps,
    log10beta_min=log10beta_min,
    binary=True,
)

NstarMax = 1000
Nsegs0 = 100
fig, axes = plt.subplots(nrows=10, figsize=(3.4, 3.5))
mcmc.run(
    NstarMax=NstarMax,
    Nsegs0=Nsegs0,
    walker_plot_args={
        "labelpad": 0.01,
        "plot_det_stat": True,
        "fig": fig,
        "axes": axes,
    },
)
if hasattr(mcmc, "walkers_fig") and hasattr(mcmc, "walkers_axes"):
    # walkers figure is only returned on first run, not when saved data is reused
    for ax in mcmc.walkers_axes:
        ax.grid()
        ax.set_xticks(np.arange(0, 600, 100))
        ax.set_xticklabels([str(s) for s in np.arange(0, 700, 100)])
    mcmc.walkers_axes[-1].set_xlabel(r"Number of steps", labelpad=0.1)
    mcmc.walkers_fig.tight_layout()
    mcmc.walkers_fig.savefig(
        os.path.join(mcmc.outdir, mcmc.label + "_walkers.png"), dpi=400
    )
    mcmc.plot_corner(add_prior=True)
    mcmc.print_summary()
