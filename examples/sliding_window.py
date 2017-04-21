import pyfstat
import numpy as np

# Properties of the GW data
sqrtSX = 1e-23
tstart = 1000000000
duration = 100*86400
tend = tstart + duration

# Properties of the signal
F0 = 30.0
F1 = -1e-10
F2 = 0
Alpha = np.radians(83.6292)
Delta = np.radians(22.0144)
tref = .5*(tstart+tend)

depth = 60
h0 = sqrtSX / depth
data_label = 'sliding_window'

data = pyfstat.Writer(
    label=data_label, outdir='data', tref=tref,
    tstart=tstart, F0=F0, F1=F1, F2=F2, duration=duration, Alpha=Alpha,
    Delta=Delta, h0=h0, sqrtSX=sqrtSX)
data.make_data()

DeltaF0 = 1e-5
search = pyfstat.FrequencySlidingWindow(
        label='sliding_window', outdir='data', sftfilepath='data/*sliding_window*sft',
        F0s=[F0-DeltaF0, F0+DeltaF0, DeltaF0/100.], F1=F1, F2=0,
        Alpha=Alpha, Delta=Delta, tref=tref, minStartTime=tstart,
        maxStartTime=tend, window_size=25*86400, window_delta=1*86400)
search.run()
search.plot_sliding_window()
