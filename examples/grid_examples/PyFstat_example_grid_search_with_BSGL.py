"""
Targeted grid search with line-robust BSGL statistic
====================================================

Search for a monochromatic (no spindown) signal using
a parameter space grid (i.e. no MCMC)
and the line-robust BSGL statistic
to distinguish an astrophysical signal from an artifact in a single detector.
"""

import pyfstat
import numpy as np
import os

label = "PyFstat_example_grid_search_BSGL"
outdir = os.path.join("PyFstat_example_data", label)

F0 = 30.0
F1 = 0
F2 = 0
Alpha = 1.0
Delta = 1.5

# Properties of the GW data - first we make data for two detectors,
# both including Gaussian noise and a coherent 'astrophysical' signal.
depth = 70
sqrtS = "1e-23"
h0 = float(sqrtS) / depth
cosi = 0
IFOs = "H1,L1"
sqrtSX = ",".join(np.repeat(sqrtS, len(IFOs.split(","))))
tstart = 1000000000
duration = 100 * 86400
tend = tstart + duration
tref = 0.5 * (tstart + tend)

data = pyfstat.Writer(
    label=label,
    outdir=outdir,
    tref=tref,
    tstart=tstart,
    duration=duration,
    F0=F0,
    F1=F1,
    F2=F2,
    Alpha=Alpha,
    Delta=Delta,
    h0=h0,
    cosi=cosi,
    sqrtSX=sqrtSX,
    detectors=IFOs,
    SFTWindowType="tukey",
    SFTWindowBeta=0.001,
    Band=1,
)
data.make_data()

# Now we add an additional single-detector artifact to H1 only.
# For simplicity, this is modelled here as a fully modulated CW-like signal,
# just restricted to the single detector.
SFTs_H1 = data.sftfilepath.split(";")[0]
extra_writer = pyfstat.Writer(
    label=label,
    outdir=outdir,
    tref=tref,
    F0=F0 + 0.01,
    F1=F1,
    F2=F2,
    Alpha=Alpha,
    Delta=Delta,
    h0=10 * h0,
    cosi=cosi,
    sqrtSX=0,  # don't add yet another set of Gaussian noise
    noiseSFTs=SFTs_H1,
    SFTWindowType="tukey",
    SFTWindowBeta=0.001,
)
extra_writer.make_data()

# set up search parameter ranges
dF0 = 0.0001
DeltaF0 = 1000 * dF0
F0s = [F0 - DeltaF0 / 2.0, F0 + DeltaF0 / 2.0, dF0]
F1s = [F1]
F2s = [F2]
Alphas = [Alpha]
Deltas = [Delta]

# first search: standard F-statistic
# This should show a weak peak from the coherent signal
# and a larger one from the "line artifact" at higher frequency.
searchF = pyfstat.GridSearch(
    label + "_twoF",
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
)
searchF.run()

print("Plotting 2F(F0)...")
searchF.plot_1D(xkey="F0")

# second search: line-robust statistic BSGL activated
searchBSGL = pyfstat.GridSearch(
    label + "_BSGL",
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
    BSGL=True,
)
searchBSGL.run()

# The actual output statistic is log10BSGL.
# The peak at the higher frequency from the "line artifact" should now
# be massively suppressed.
print("Plotting log10BSGL(F0)...")
searchBSGL.plot_1D(xkey="F0")
