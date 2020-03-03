import pyfstat
import numpy as np
import os

# Properties of the GW data
sqrtSX = 1e-23
tstart = 1000000000
duration = 100 * 86400
tend = tstart + duration

# Properties of the signal
F0 = 30.0
F1 = -1e-10
F2 = 0
Alpha = np.radians(83.6292)
Delta = np.radians(22.0144)
tref = 0.5 * (tstart + tend)

depth = 10
h0 = sqrtSX / depth
label = os.path.splitext(os.path.basename(__file__))[0]
outdir = os.path.join("example_data", label)

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

DeltaF0 = 1e-7
DeltaF1 = 1e-13
VF0 = (np.pi * duration * DeltaF0) ** 2 / 3.0
VF1 = (np.pi * duration ** 2 * DeltaF1) ** 2 * 4 / 45.0
print("\nV={:1.2e}, VF0={:1.2e}, VF1={:1.2e}\n".format(VF0 * VF1, VF0, VF1))

theta_prior = {
    "F0": {"type": "unif", "lower": F0 - DeltaF0 / 2.0, "upper": F0 + DeltaF0 / 2.0},
    "F1": {"type": "unif", "lower": F1 - DeltaF1 / 2.0, "upper": F1 + DeltaF1 / 2.0},
    "F2": F2,
    "Alpha": Alpha,
    "Delta": Delta,
}

ntemps = 1
log10beta_min = -1
nwalkers = 100
nsteps = [100, 100]

mcmc = pyfstat.MCMCSearch(
    label=label,
    outdir=outdir,
    sftfilepattern=os.path.join(outdir, "*{}*sft".format(label)),
    theta_prior=theta_prior,
    tref=tref,
    minStartTime=tstart,
    maxStartTime=tend,
    nsteps=nsteps,
    nwalkers=nwalkers,
    ntemps=ntemps,
    log10beta_min=log10beta_min,
)
mcmc.setup_initialisation(100, scatter_val=1e-10)
mcmc.run()
mcmc.plot_corner(add_prior=True)
mcmc.print_summary()
