#!/usr/bin/env python

import pyfstat
import os

outdir = os.path.join("PyFstat_example_data", "PyFstat_example_simple_mcmc_vs_grid")

F0 = 30.0
F1 = -1e-10
F2 = 0
Alpha = 0.5
Delta = 1

minStartTime = 1000000000
duration = 2 * 86400
maxStartTime = minStartTime + duration
tref = minStartTime

h0 = 1e-23
cosi = 1.0
sqrtSX = 1e-22
detectors = "H1,L1"

Tsft = 1800

transient = pyfstat.Writer(
    label="simulated_signal",
    outdir=outdir,
    tref=tref,
    tstart=minStartTime,
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
    Tsft=Tsft,
)
transient.make_data()
