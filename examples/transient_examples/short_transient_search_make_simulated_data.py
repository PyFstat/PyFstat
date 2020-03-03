#!/usr/bin/env python

import pyfstat
import os

F0 = 30.0
F1 = -1e-10
F2 = 0
Alpha = 0.5
Delta = 1

minStartTime = 1000000000
maxStartTime = minStartTime + 2 * 86400

transient_tstart = minStartTime + 0.5 * 86400
transient_duration = 1 * 86400
tref = minStartTime

h0 = 1e-23
sqrtSX = 1e-22
detectors = "H1,L1"

Tsft = 1800

outdir = os.path.join("example_data", "short_transient")

transient = pyfstat.Writer(
    label="simulated_transient_signal",
    outdir=outdir,
    tref=tref,
    tstart=transient_tstart,
    duration=transient_duration,
    F0=F0,
    F1=F1,
    F2=F2,
    Alpha=Alpha,
    Delta=Delta,
    h0=h0,
    detectors=detectors,
    sqrtSX=sqrtSX,
    minStartTime=minStartTime,
    maxStartTime=maxStartTime,
    transientWindowType="rect",
    Tsft=Tsft,
)
transient.make_data()
