import pyfstat
import numpy as np
import matplotlib.pyplot as plt
import os

label = os.path.splitext(os.path.basename(__file__))[0]
outdir = os.path.join("PyFstat_example_data", label)

F0 = 30.0
F1 = 0
F2 = 0
Alpha = 1.0
Delta = 1.5

# Properties of the GW data
depth = 70
sqrtS = "1e-23"
h0 = float(sqrtS) / depth
cosi = 0
IFOs = "H1"
# IFOs = "H1,L1"
sqrtSX = ",".join(np.repeat(sqrtS, len(IFOs.split(","))))
tstart = 1000000000
duration = 100 * 86400
tend = tstart + duration
tref = 0.5 * (tstart + tend)

example_name = os.path.splitext(os.path.basename(__file__))[0]

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

m = 0.001
dF0 = np.sqrt(12 * m) / (np.pi * duration)
DeltaF0 = 800 * dF0
F0s = [F0 - DeltaF0 / 2.0, F0 + DeltaF0 / 2.0, dF0]
F1s = [F1]
F2s = [F2]
Alphas = [Alpha]
Deltas = [Delta]
search = pyfstat.GridSearch(
    example_name,
    outdir,
    os.path.join(outdir, "*" + label + "*sft"),
    F0s,
    F1s,
    F2s,
    Alphas,
    Deltas,
    tref,
    tstart,
    tend,
    BSGL=False,
)
search.run()

fig, ax = plt.subplots()
xidx = search.keys.index("F0")
frequencies = np.unique(search.data[:, xidx])
twoF = search.data[:, -1]

# mismatch = np.sign(x-F0)*(duration * np.pi * (x - F0))**2 / 12.0
ax.plot(frequencies, twoF, "k", lw=1)
DeltaF = frequencies - F0
sinc = np.sin(np.pi * DeltaF * duration) / (np.pi * DeltaF * duration)
A = np.abs((np.max(twoF) - 4) * sinc ** 2 + 4)
ax.plot(frequencies, A, "-r", lw=1)
ax.set_ylabel("$\widetilde{2\mathcal{F}}$")
ax.set_xlabel("Frequency")
ax.set_xlim(F0s[0], F0s[1])
dF0 = np.sqrt(12 * 1) / (np.pi * duration)
xticks = [F0 - 10 * dF0, F0, F0 + 10 * dF0]
ax.set_xticks(xticks)
xticklabels = ["$f_0 {-} 10\Delta f$", "$f_0$", "$f_0 {+} 10\Delta f$"]
ax.set_xticklabels(xticklabels)
plt.tight_layout()
fig.savefig(os.path.join(outdir, label + "_1D.png"), dpi=300)
