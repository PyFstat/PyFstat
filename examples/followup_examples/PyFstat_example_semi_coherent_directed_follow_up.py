import pyfstat
import numpy as np
import matplotlib.pyplot as plt
import os

F0 = 30.0
F1 = -1e-10
F2 = 0
Alpha = np.radians(83.6292)
Delta = np.radians(22.0144)

# Properties of the GW data
sqrtSX = 1e-23
tstart = 1000000000
duration = 100 * 86400
tend = tstart + duration
tref = 0.5 * (tstart + tend)

depth = 40
label = os.path.splitext(os.path.basename(__file__))[0]
outdir = os.path.join("example_data", label)

h0 = sqrtSX / depth

data = pyfstat.Writer(
    label=label,
    outdir=outdir,
    tref=tref,
    tstart=tstart,
    F0=F0,
    F1=F1,
    F2=F2,
    duration=duration,
    Alpha=Alpha,
    Delta=Delta,
    h0=h0,
    sqrtSX=sqrtSX,
)
data.make_data()

# The predicted twoF, given by lalapps_predictFstat can be accessed by
twoF = data.predict_fstat()
print("Predicted twoF value: {}\n".format(twoF))

# Search
VF0 = VF1 = 1e5
DeltaF0 = np.sqrt(VF0) * np.sqrt(3) / (np.pi * duration)
DeltaF1 = np.sqrt(VF1) * np.sqrt(180) / (np.pi * duration ** 2)
theta_prior = {
    "F0": {"type": "unif", "lower": F0 - DeltaF0 / 2.0, "upper": F0 + DeltaF0 / 2},
    "F1": {"type": "unif", "lower": F1 - DeltaF1 / 2.0, "upper": F1 + DeltaF1 / 2},
    "F2": F2,
    "Alpha": Alpha,
    "Delta": Delta,
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
    tref=tref,
    minStartTime=tstart,
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
