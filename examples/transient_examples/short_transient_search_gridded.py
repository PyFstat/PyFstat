#!/usr/bin/env python

import pyfstat
import os
import numpy as np
import matplotlib.pyplot as plt

datadir = 'data_s'

F0 = 30.0
F1 = -1e-10
F2 = 0
Alpha = 0.5
Delta = 1

minStartTime = 1000000000
maxStartTime = minStartTime + 2*86400
Tspan = maxStartTime - minStartTime
tref = minStartTime

Tsft = 1800

m = 0.001
dF0 = np.sqrt(12*m)/(np.pi*Tspan)
DeltaF0 = 100*dF0
F0s = [F0-DeltaF0/2., F0+DeltaF0/2., dF0]
F1s = [F1]
F2s = [F2]
Alphas = [Alpha]
Deltas = [Delta]

print('Standard CW search:')
search1 = pyfstat.TransientGridSearch(
    label='CW', outdir=datadir,
    sftfilepattern=os.path.join(datadir,'*simulated_transient_signal*sft'),
    F0s=F0s, F1s=F1s, F2s=F2s, Alphas=Alphas, Deltas=Deltas, tref=tref,
    minStartTime=minStartTime, maxStartTime=maxStartTime,
    BSGL=False,
    outputTransientFstatMap=True)
search1.run()
search1.print_max_twoF()

search1.plot_1D(xkey='F0',
               xlabel='freq [Hz]', ylabel='$2\mathcal{F}$')

print('with t0,tau bands:')
search2 = pyfstat.TransientGridSearch(
    label='tCW', outdir=datadir,
    sftfilepattern=os.path.join(datadir,'*simulated_transient_signal*sft'),
    F0s=F0s, F1s=F1s, F2s=F2s, Alphas=Alphas, Deltas=Deltas, tref=tref,
    minStartTime=minStartTime, maxStartTime=maxStartTime,
    transientWindowType='rect', t0Band=Tspan-2*Tsft, tauBand=Tspan,
    BSGL=False,
    outputTransientFstatMap=True)
search2.run()
search2.print_max_twoF()

search2.plot_1D(xkey='F0',
               xlabel='freq [Hz]', ylabel='$2\mathcal{F}$')
