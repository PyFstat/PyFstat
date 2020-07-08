#!/usr/bin/env python

import pyfstat
import os

F0 = 30.0
F1 = -1e-10
F2 = 0
Alpha = 0.5
Delta = 1

minStartTime = 1000000000
maxStartTime = minStartTime + 200 * 86400

transient_tstart = minStartTime + 50 * 86400
transient_duration = 100 * 86400
tref = minStartTime

h0 = 1e-23
sqrtSX = 1e-22

outdir = os.path.join("example_data", "long_transient")

transient = pyfstat.Writer(
    label="simulated_transient_signal",
    outdir=outdir,
    tref=tref,
    tstart=transient_tstart,
    F0=F0,
    F1=F1,
    F2=F2,
    duration=transient_duration,
    Alpha=Alpha,
    Delta=Delta,
    h0=h0,
    sqrtSX=sqrtSX,
    minStartTime=minStartTime,
    maxStartTime=maxStartTime,
    transientWindowType="rect",
)
transient.make_data()
