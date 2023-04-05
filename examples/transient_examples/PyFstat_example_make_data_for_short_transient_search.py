"""
Short transient search examples: Make data
==========================================

An example to generate data with a short transient signal.

This can be run either stand-alone (will just generate SFT files and nothing else);
or it is also being imported from
PyFstat_example_short_transient_grid_search.py
and
PyFstat_example_short_transient_MCMC_search.py
"""

import os

import pyfstat

label = "PyFstatExampleShortTransientSearch"
outdir = os.path.join("PyFstat_example_data", label)
logger = pyfstat.set_up_logger(outdir=outdir, label=label)


F0 = 30.0
F1 = -1e-10
F2 = 0
Alpha = 0.5
Delta = 1

tstart = 1000000000
duration = 2 * 86400

transient_tstart = tstart + 0.25 * duration
transient_duration = 0.5 * duration
transientWindowType = "rect"
tref = tstart

h0 = 1e-23
cosi = 0
psi = 0
phi = 0
sqrtSX = 1e-22
detectors = "H1,L1"

Tsft = 1800

if __name__ == "__main__":
    transient = pyfstat.Writer(
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
        detectors=detectors,
        sqrtSX=sqrtSX,
        transientStartTime=transient_tstart,
        transientTau=transient_duration,
        transientWindowType=transientWindowType,
        Tsft=Tsft,
        Band=0.1,
    )
    transient.make_data()
    logger.info(f"Predicted 2F from injection Writer: {transient.predict_fstat()}")
