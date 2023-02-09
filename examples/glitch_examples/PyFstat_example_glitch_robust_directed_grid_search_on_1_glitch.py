"""
Glitch robust grid search
=========================

Grid search employing a signal hypothesis allowing for a glitch to
be present in the data. The setup corresponds to a targeted search,
and the simulated signal contains a single glitch.
"""

import os
import time

import numpy as np
from PyFstat_example_make_data_for_search_on_1_glitch import (
    F0,
    F1,
    F2,
    Alpha,
    Delta,
    delta_F0,
    dtglitch,
    duration,
    outdir,
    tref,
    tstart,
)

import pyfstat

label = "PyFstatExampleGlitchRobustDirectedGridSearchOn1Glitch"
logger = pyfstat.set_up_logger(label=label, outdir=outdir)

Nstar = 1000
F0_width = np.sqrt(Nstar) * np.sqrt(12) / (np.pi * duration)
F1_width = np.sqrt(Nstar) * np.sqrt(180) / (np.pi * duration**2)
N = 20
F0s = [F0 - F0_width / 2.0, F0 + F0_width / 2.0, F0_width / N]
F1s = [F1 - F1_width / 2.0, F1 + F1_width / 2.0, F1_width / N]
F2s = [F2]
Alphas = [Alpha]
Deltas = [Delta]

max_delta_F0 = 1e-5
tglitchs = [tstart + 0.1 * duration, tstart + 0.9 * duration, 0.8 * float(duration) / N]
delta_F0s = [0, max_delta_F0, max_delta_F0 / N]
delta_F1s = [0]


t1 = time.time()
search = pyfstat.GridGlitchSearch(
    label,
    outdir,
    os.path.join(outdir, "*1glitch*sft"),
    F0s=F0s,
    F1s=F1s,
    F2s=F2s,
    Alphas=Alphas,
    Deltas=Deltas,
    tref=tref,
    minStartTime=tstart,
    maxStartTime=tstart + duration,
    tglitchs=tglitchs,
    delta_F0s=delta_F0s,
    delta_F1s=delta_F1s,
)
search.run()
dT = time.time() - t1

F0_vals = np.unique(search.data["F0"]) - F0
F1_vals = np.unique(search.data["F1"]) - F1
delta_F0s_vals = np.unique(search.data["delta_F0"]) - delta_F0
tglitch_vals = np.unique(search.data["tglitch"])
tglitch_vals_days = (tglitch_vals - tstart) / 86400.0 - dtglitch / 86400.0

logger.info("Making gridcorner plot...")
twoF = search.data["twoF"].reshape(
    (len(F0_vals), len(F1_vals), len(delta_F0s_vals), len(tglitch_vals))
)
xyz = [F0_vals * 1e6, F1_vals * 1e12, delta_F0s_vals * 1e6, tglitch_vals_days]
labels = [
    "$f - f_\\mathrm{s}$\n[$\\mu$Hz]",
    "$\\dot{f} - \\dot{f}_\\mathrm{s}$\n[$p$Hz/s]",
    "$\\delta f-\\delta f_\\mathrm{s}$\n[$\\mu$Hz]",
    "$t^\\mathrm{g} - t^\\mathrm{g}_\\mathrm{s}$\n[d]",
    "$t^\\mathrm{g} - t^\\mathrm{g}_\\mathrm{s}$\n[d]",
    "$\\widehat{2\\mathcal{F}}$",
]
fig, axes = pyfstat.gridcorner(
    twoF,
    xyz,
    projection="log_mean",
    labels=labels,
    showDvals=False,
    lines=[0, 0, 0, 0],
    label_offset=0.25,
    max_n_ticks=4,
)
fig.savefig("{}/{}_projection_matrix.png".format(outdir, label), bbox_inches="tight")


logger.info(f"Prior widths = {F0_width}, {F1_width}")
logger.info(f"Actual run time = {dT} s")
logger.info(f"Actual number of grid points = {search.data.shape[0]}")
