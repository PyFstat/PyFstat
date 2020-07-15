import numpy as np
import pyfstat
import gridcorner
import time
from PyFstat_example_make_data_for_search_on_1_glitch import (
    tstart,
    duration,
    tref,
    F0,
    F1,
    F2,
    Alpha,
    Delta,
    delta_F0,
    dtglitch,
    outdir,
)
import os

outdir = os.path.join("PyFstat_example_data", "PyFstat_example_glitch_robust_search")
label = "semicoherent_glitch_robust_directed_MCMC_search_on_1_glitch"

Nstar = 1000
F0_width = np.sqrt(Nstar) * np.sqrt(12) / (np.pi * duration)
F1_width = np.sqrt(Nstar) * np.sqrt(180) / (np.pi * duration ** 2)

theta_prior = {
    "F0": {"type": "unif", "lower": F0 - F0_width / 2.0, "upper": F0 + F0_width / 2.0},
    "F1": {"type": "unif", "lower": F1 - F1_width / 2.0, "upper": F1 + F1_width / 2.0},
    "F2": F2,
    "delta_F0": {"type": "unif", "lower": 0, "upper": 1e-5},
    "delta_F1": 0,
    "tglitch": {
        "type": "unif",
        "lower": tstart + 0.1 * duration,
        "upper": tstart + 0.9 * duration,
    },
    "Alpha": Alpha,
    "Delta": Delta,
}

ntemps = 3
log10beta_min = -0.5
nwalkers = 100
nsteps = [250, 250]

mcmc = pyfstat.MCMCGlitchSearch(
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
    nglitch=1,
)
mcmc.transform_dictionary["F0"] = dict(
    subtractor=F0, multiplier=1e6, symbol="$f-f_\mathrm{s}$"
)
mcmc.unit_dictionary["F0"] = "$\mu$Hz"
mcmc.transform_dictionary["F1"] = dict(
    subtractor=F1, multiplier=1e12, symbol="$\dot{f}-\dot{f}_\mathrm{s}$"
)
mcmc.unit_dictionary["F1"] = "$p$Hz/s"
mcmc.transform_dictionary["delta_F0"] = dict(
    multiplier=1e6, subtractor=delta_F0, symbol="$\delta f-\delta f_\mathrm{s}$"
)
mcmc.unit_dictionary["delta_F0"] = "$\mu$Hz/s"
mcmc.transform_dictionary["tglitch"]["subtractor"] = tstart + dtglitch
mcmc.transform_dictionary["tglitch"][
    "label"
] = "$t^\mathrm{g}-t^\mathrm{g}_\mathrm{s}$\n[d]"

t1 = time.time()
mcmc.run(save_loudest=False)  # uses CFSv2 which doesn't support glitch parameters
dT = time.time() - t1
fig_and_axes = gridcorner._get_fig_and_axes(4, 2, 0.05)
mcmc.plot_corner(
    label_offset=0.25,
    truths=[0, 0, 0, 0],
    fig_and_axes=fig_and_axes,
    quantiles=(0.16, 0.84),
    hist_kwargs=dict(lw=1.5, zorder=-1),
    truth_color="C3",
)

mcmc.print_summary()

print(("Prior widths =", F0_width, F1_width))
print(("Actual run time = {}".format(dT)))
