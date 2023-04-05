"""
Long transient search examples: Make data
=========================================

An example to generate data with a long transient signal.

This can be run either stand-alone (will just generate SFT files and nothing else);
or it is also being imported from
PyFstat_example_long_transient_MCMC_search.py
"""

import os

import pyfstat

label = "PyFstatExampleLongTransientSearch"
outdir = os.path.join("PyFstat_example_data", label)
logger = pyfstat.set_up_logger(outdir=outdir, label=label)

F0 = 30.0
F1 = -1e-10
F2 = 0
Alpha = 0.5
Delta = 1

tstart = 1000000000
duration = 200 * 86400

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
    )
    transient.make_data()
    logger.info(f"Predicted 2F from injection Writer: {transient.predict_fstat()}")
