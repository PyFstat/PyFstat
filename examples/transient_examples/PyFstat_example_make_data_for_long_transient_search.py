#!/usr/bin/env python

import pyfstat
import os

outdir = os.path.join("PyFstat_example_data", "PyFstat_example_long_transient_search")

F0 = 30.0
F1 = -1e-10
F2 = 0
Alpha = 0.5
Delta = 1

tstart = 1000000000
duration = 200 * 86400

transient_tstart = tstart + 0.25 * duration
transient_duration = 0.5 * duration
tref = tstart

h0 = 1e-23
cosi = 0
sqrtSX = 1e-22
detectors = "H1,L1"

transient = pyfstat.Writer(
    label="simulated_transient_signal",
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
    transientWindowType="rect",
)
transient.make_data()
