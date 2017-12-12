#!/usr/bin/env python

import pyfstat
import numpy as np
import matplotlib.pyplot as plt

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

search = pyfstat.GridSearch(
    label='grid_search', outdir='data_s',
    sftfilepattern='data_s/*simulated_transient_signal*sft',
    F0s=F0s, F1s=F1s, F2s=F2s, Alphas=Alphas, Deltas=Deltas, tref=tref,
    minStartTime=minStartTime, maxStartTime=maxStartTime,
#    transientWindowType='rect', t0Band=Tspan-2*Tsft, tauBand=Tspan,
    BSGL=False)
search.run()
search.print_max_twoF()

search.plot_1D(xkey='F0',
               xlabel='freq [Hz]', ylabel='$2\mathcal{F}$')

search.plot_2D(xkey='F0', ykey='F1')
