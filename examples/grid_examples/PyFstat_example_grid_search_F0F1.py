"""
Directed grid search: Linear spindown
==========================================

Search for CW signal including one spindown parameter
using a parameter space grid (i.e. no MCMC).
"""

import os

import numpy as np

import pyfstat

label = "PyFstatExampleGridSearchF0F1"
outdir = os.path.join("PyFstat_example_data", label)
logger = pyfstat.set_up_logger(label=label, outdir=outdir)

# Properties of the GW data
sqrtSX = 1e-23
tstart = 1000000000
duration = 10 * 86400
tend = tstart + duration
tref = 0.5 * (tstart + tend)
IFOs = "H1"

# parameters for injected signals
depth = 20
inj = {
    "tref": tref,
    "F0": 30.0,
    "F1": -1e-10,
    "F2": 0,
    "Alpha": 1.0,
    "Delta": 1.5,
    "h0": sqrtSX / depth,
    "cosi": 0.0,
}

data = pyfstat.Writer(
    label=label,
    outdir=outdir,
    tstart=tstart,
    duration=duration,
    sqrtSX=sqrtSX,
    detectors=IFOs,
    **inj,
)
data.make_data()

m = 0.01
dF0 = np.sqrt(12 * m) / (np.pi * duration)
dF1 = np.sqrt(180 * m) / (np.pi * duration**2)
dF2 = 1e-17
N = 100
DeltaF0 = N * dF0
DeltaF1 = N * dF1
F0s = [inj["F0"] - DeltaF0 / 2.0, inj["F0"] + DeltaF0 / 2.0, dF0]
F1s = [inj["F1"] - DeltaF1 / 2.0, inj["F1"] + DeltaF1 / 2.0, dF1]
F2s = [inj["F2"]]
Alphas = [inj["Alpha"]]
Deltas = [inj["Delta"]]
search = pyfstat.GridSearch(
    label=label,
    outdir=outdir,
    sftfilepattern=data.sftfilepath,
    F0s=F0s,
    F1s=F1s,
    F2s=F2s,
    Alphas=Alphas,
    Deltas=Deltas,
    tref=tref,
    minStartTime=tstart,
    maxStartTime=tend,
)
search.run()

# report details of the maximum point
max_dict = search.get_max_twoF()
logger.info(
    "max2F={:.4f} from GridSearch, offsets from injection: {:s}.".format(
        max_dict["twoF"],
        ", ".join(
            [
                "{:.4e} in {:s}".format(max_dict[key] - inj[key], key)
                for key in max_dict.keys()
                if not key == "twoF"
            ]
        ),
    )
)
search.generate_loudest()

logger.info("Plotting 2F(F0)...")
search.plot_1D(xkey="F0", xlabel="freq [Hz]", ylabel="$2\\mathcal{F}$")
logger.info("Plotting 2F(F1)...")
search.plot_1D(xkey="F1")
logger.info("Plotting 2F(F0,F1)...")
search.plot_2D(xkey="F0", ykey="F1", colorbar=True)

logger.info("Making gridcorner plot...")
F0_vals = np.unique(search.data["F0"]) - inj["F0"]
F1_vals = np.unique(search.data["F1"]) - inj["F1"]
twoF = search.data["twoF"].reshape((len(F0_vals), len(F1_vals)))
xyz = [F0_vals, F1_vals]
labels = [
    "$f - f_0$",
    "$\\dot{f} - \\dot{f}_0$",
    "$\\widetilde{2\\mathcal{F}}$",
]
fig, axes = pyfstat.gridcorner(
    twoF, xyz, projection="log_mean", labels=labels, whspace=0.1, factor=1.8
)
fig.savefig(os.path.join(outdir, label + "_projection_matrix.png"))
