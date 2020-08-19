import pyfstat
import numpy as np
import os

label = os.path.splitext(os.path.basename(__file__))[0]
outdir = os.path.join("PyFstat_example_data", label)

# Properties of the GW data
sqrtSX = 1e-23
tstart = 1000000000
duration = 100 * 86400
tend = tstart + duration
detectors = "H1"

# Properties of the signal
F0 = 30.0
F1 = -1e-10
F2 = 0
Alpha = np.radians(83.6292)
Delta = np.radians(22.0144)
tref = 0.5 * (tstart + tend)
cosi = 0

depth = 60
h0 = sqrtSX / depth
cosi = 1.0

data = pyfstat.Writer(
    label=label,
    outdir=outdir,
    tref=tref,
    tstart=tstart,
    F0=F0,
    F1=F1,
    F2=F2,
    duration=duration,
    detectors=detectors,
    Alpha=Alpha,
    Delta=Delta,
    h0=h0,
    cosi=cosi,
    sqrtSX=sqrtSX,
    detectors="H1",
)
data.make_data()

DeltaF0 = 1e-5
search = pyfstat.FrequencySlidingWindow(
    label=label,
    outdir=outdir,
    sftfilepattern=os.path.join(outdir, "*{}*sft".format(label)),
    F0s=[F0 - DeltaF0, F0 + DeltaF0, DeltaF0 / 100.0],
    F1=F1,
    F2=0,
    Alpha=Alpha,
    Delta=Delta,
    tref=tref,
    minStartTime=tstart,
    maxStartTime=tend,
    window_size=25 * 86400,
    window_delta=1 * 86400,
)
search.run()
search.plot_sliding_window()
