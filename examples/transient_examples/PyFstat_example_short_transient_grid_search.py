#!/usr/bin/env python

import pyfstat
import os
import numpy as np

outdir = os.path.join("PyFstat_example_data", "PyFstat_example_short_transient_search")
if not os.path.isdir(outdir) or not np.any(
    [f.endswith(".sft") for f in os.listdir(outdir)]
):
    raise RuntimeError(
        "Please first run PyFstat_example_make_data_for_short_transient_search.py !"
    )

F0 = 30.0
F1 = -1e-10
F2 = 0
Alpha = 0.5
Delta = 1

minStartTime = 1000000000
maxStartTime = minStartTime + 2 * 86400
Tspan = maxStartTime - minStartTime
tref = minStartTime

Tsft = 1800

m = 0.001
dF0 = np.sqrt(12 * m) / (np.pi * Tspan)
DeltaF0 = 100 * dF0
F0s = [F0 - DeltaF0 / 2.0, F0 + DeltaF0 / 2.0, dF0]
F1s = [F1]
F2s = [F2]
Alphas = [Alpha]
Deltas = [Delta]

print("Standard CW search:")
search1 = pyfstat.GridSearch(
    label="CW",
    outdir=outdir,
    sftfilepattern=os.path.join(outdir, "*simulated_transient_signal*sft"),
    F0s=F0s,
    F1s=F1s,
    F2s=F2s,
    Alphas=Alphas,
    Deltas=Deltas,
    tref=tref,
    minStartTime=minStartTime,
    maxStartTime=maxStartTime,
    BSGL=False,
)
search1.run()
search1.print_max_twoF()
search1.save_array_to_disk(search1.data)

search1.plot_1D(xkey="F0", xlabel="freq [Hz]", ylabel="$2\mathcal{F}$")

print("with t0,tau bands:")
search2 = pyfstat.TransientGridSearch(
    label="tCW",
    outdir=outdir,
    sftfilepattern=os.path.join(outdir, "*simulated_transient_signal*sft"),
    F0s=F0s,
    F1s=F1s,
    F2s=F2s,
    Alphas=Alphas,
    Deltas=Deltas,
    tref=tref,
    minStartTime=minStartTime,
    maxStartTime=maxStartTime,
    transientWindowType="rect",
    t0Band=Tspan - 2 * Tsft,
    tauBand=Tspan,
    BSGL=False,
    outputTransientFstatMap=True,
    tCWFstatMapVersion="lal",
)
search2.run()
search2.print_max_twoF()
search2.save_array_to_disk(search2.data)

search2.plot_1D(xkey="F0", xlabel="freq [Hz]", ylabel="$2\mathcal{F}$")
