import pyfstat
import numpy as np
import matplotlib.pyplot as plt
import os

try:
    from gridcorner import gridcorner
except ImportError:
    raise ImportError(
        "Python module 'gridcorner' not found, please install from "
        "https://gitlab.aei.uni-hannover.de/GregAshton/gridcorner"
    )

label = os.path.splitext(os.path.basename(__file__))[0]
outdir = os.path.join("PyFstat_example_data", label)

F0 = 30.0
F1 = 1e-10
F2 = 0
Alpha = 1.0
Delta = 1.5

# Properties of the GW data
sqrtSX = 1e-23
tstart = 1000000000
duration = 10 * 86400
tend = tstart + duration
tref = 0.5 * (tstart + tend)
IFOs = "H1"

depth = 20

h0 = sqrtSX / depth
cosi = 0

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
    cosi=cosi,
    sqrtSX=sqrtSX,
    detectors=IFOs,
)
data.make_data()

m = 0.01
dF0 = np.sqrt(12 * m) / (np.pi * duration)
dF1 = np.sqrt(180 * m) / (np.pi * duration ** 2)
dF2 = 1e-17
N = 100
DeltaF0 = N * dF0
DeltaF1 = N * dF1
F0s = [F0 - DeltaF0 / 2.0, F0 + DeltaF0 / 2.0, dF0]
F1s = [F1 - DeltaF1 / 2.0, F1 + DeltaF1 / 2.0, dF1]
F2s = [F2]
Alphas = [Alpha]
Deltas = [Delta]
search = pyfstat.GridSearch(
    label, outdir, data.sftfilepath, F0s, F1s, F2s, Alphas, Deltas, tref, tstart, tend,
)
search.run()

search.plot_1D(xkey="F0", xlabel="freq [Hz]", ylabel="$2\mathcal{F}$")
search.plot_1D(xkey="F1")
search.plot_2D(xkey="F0", ykey="F1", colorbar=True)

F0_vals = np.unique(search.data[:, 2]) - F0
F1_vals = np.unique(search.data[:, 3]) - F1
twoF = search.data[:, -1].reshape((len(F0_vals), len(F1_vals)))
xyz = [F0_vals, F1_vals]
labels = [
    "$f - f_0$",
    "$\dot{f} - \dot{f}_0$",
    "$\widetilde{2\mathcal{F}}$",
]
fig, axes = gridcorner(
    twoF, xyz, projection="log_mean", labels=labels, whspace=0.1, factor=1.8
)
fig.savefig(os.path.join(outdir, label + "_projection_matrix.png"))
