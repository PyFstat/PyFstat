"""
Directed grid search: Monochromatic source
==========================================

Search for a monochromatic (no spindown) signal using
a parameter space grid (i.e. no MCMC).
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import pyfstat

label = "PyFstatExampleGridSearchF0"
outdir = os.path.join("PyFstat_example_data", label)
logger = pyfstat.set_up_logger(label=label, outdir=outdir)

# Properties of the GW data
sqrtS = "1e-23"
IFOs = "H1"
# IFOs = "H1,L1"
sqrtSX = ",".join(np.repeat(sqrtS, len(IFOs.split(","))))
tstart = 1000000000
duration = 100 * 86400
tend = tstart + duration
tref = 0.5 * (tstart + tend)

# parameters for injected signals
depth = 70
inj = {
    "tref": tref,
    "F0": 30.0,
    "F1": 0,
    "F2": 0,
    "Alpha": 1.0,
    "Delta": 1.5,
    "h0": float(sqrtS) / depth,
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

m = 0.001
dF0 = np.sqrt(12 * m) / (np.pi * duration)
DeltaF0 = 800 * dF0
F0s = [inj["F0"] - DeltaF0 / 2.0, inj["F0"] + DeltaF0 / 2.0, dF0]
F1s = [inj["F1"]]
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
fig, ax = plt.subplots()
frequencies = search.data["F0"]
twoF = search.data["twoF"]
# mismatch = np.sign(x-inj["F0"])*(duration * np.pi * (x - inj["F0"]))**2 / 12.0
ax.plot(frequencies, twoF, "k", lw=1)
DeltaF = frequencies - inj["F0"]
sinc = np.sin(np.pi * DeltaF * duration) / (np.pi * DeltaF * duration)
A = np.abs((np.max(twoF) - 4) * sinc**2 + 4)
ax.plot(frequencies, A, "-r", lw=1)
ax.set_ylabel("$\\widetilde{2\\mathcal{F}}$")
ax.set_xlabel("Frequency")
ax.set_xlim(F0s[0], F0s[1])
dF0 = np.sqrt(12 * 1) / (np.pi * duration)
xticks = [inj["F0"] - 10 * dF0, inj["F0"], inj["F0"] + 10 * dF0]
ax.set_xticks(xticks)
xticklabels = ["$f_0 {-} 10\\Delta f$", "$f_0$", "$f_0 {+} 10\\Delta f$"]
ax.set_xticklabels(xticklabels)
plt.tight_layout()
fig.savefig(os.path.join(outdir, label + "_1D.png"), dpi=300)
