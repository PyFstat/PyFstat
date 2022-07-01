"""
Short transient grid search
===========================

An example grid-based search for a short transient signal.

This is also ready to use on a GPU,
if you have one available and `pycuda` installed.
Just change to `tCWFstatMapVersion = "pycuda"`.
"""

import os

import numpy as np
import PyFstat_example_make_data_for_short_transient_search as data

import pyfstat

tCWFstatMapVersion = "lal"

if __name__ == "__main__":

    if not os.path.isdir(data.outdir) or not np.any(
        [f.endswith(".sft") for f in os.listdir(data.outdir)]
    ):
        raise RuntimeError(
            "Please first run PyFstat_example_make_data_for_short_transient_search.py !"
        )

    maxStartTime = data.tstart + data.duration

    m = 0.001
    dF0 = np.sqrt(12 * m) / (np.pi * data.duration)
    DeltaF0 = 100 * dF0
    F0s = [data.F0 - DeltaF0 / 2.0, data.F0 + DeltaF0 / 2.0, dF0]
    F1s = [data.F1]
    F2s = [data.F2]
    Alphas = [data.Alpha]
    Deltas = [data.Delta]

    BSGL = False
    BtSG = False

    print("Standard CW search:")
    search1 = pyfstat.GridSearch(
        label="CW" + ("_BSGL" if BSGL else ""),
        outdir=data.outdir,
        sftfilepattern=os.path.join(data.outdir, "*simulated_transient_signal*sft"),
        F0s=F0s,
        F1s=F1s,
        F2s=F2s,
        Alphas=Alphas,
        Deltas=Deltas,
        tref=data.tref,
        BSGL=BSGL,
    )
    search1.run()
    search1.print_max_twoF()
    search1.plot_1D(xkey="F0", xlabel="freq [Hz]", ylabel="$2\\mathcal{F}$")

    # Now we do the transient-enabled search.
    # We write this in "context manager" style (with the "with" keyword)
    # to be GPU-ready: if tCWFstatMapVersion=="pycuda",
    # this style is necessary to ensure proper cleanup of the cuda device context.
    print("with t0,tau bands:")
    label = f"tCW{'_BSGL' if BSGL else ''}_FstatMap_{tCWFstatMapVersion}"
    with pyfstat.TransientGridSearch(
        label=label,
        outdir=data.outdir,
        sftfilepattern=os.path.join(data.outdir, "*simulated_transient_signal*sft"),
        F0s=F0s,
        F1s=F1s,
        F2s=F2s,
        Alphas=Alphas,
        Deltas=Deltas,
        tref=data.tref,
        transientWindowType="rect",
        t0Band=data.duration - 2 * data.Tsft,
        tauBand=data.duration,
        outputTransientFstatMap=True,
        tCWFstatMapVersion=tCWFstatMapVersion,
        BSGL=BSGL,
        BtSG=BtSG,
    ) as search2:
        search2.run()
        search2.print_max_twoF()
        search2.plot_1D(xkey="F0", xlabel="freq [Hz]", ylabel="$2\\mathcal{F}$")
