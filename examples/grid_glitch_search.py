from pyfstat import GridGlitchSearch, Writer
import numpy as np

phi = 0
F0 = 2.4343
F1 = -1e-10
F2 = 0
delta_phi = 0
delta_F0 = 1e-7
delta_F1 = 0
duration = 100*86400
dtglitch = 100*43200
Alpha = 5e-3
Delta = 6e-2
tstart = 700000000
tend = tstart+duration
tglitch = tstart + dtglitch
tref = tglitch

glitch_data = Writer(label='glitch', outdir='data', delta_phi=delta_phi,
                     delta_F0=delta_F0, tref=tref, tstart=tstart,
                     delta_F1=delta_F1, phi=phi, F0=F0, F1=F1, F2=F2,
                     duration=duration, dtglitch=dtglitch, Alpha=Alpha,
                     Delta=Delta)
glitch_data.make_data()

F0s = [F0]
F1s = [F1]
F2s = [F2]
m = 1e-3
dF = np.sqrt(12 * m) / (np.pi * duration)
delta_F0s = [-1e-6*F0, 1e-6*F0, dF]
delta_F1s = [delta_F1]
dT = duration / 10.
tglitchs = [tstart+dT, tend-dT, duration/100.]
Alphas = [Alpha]
Deltas = [Delta]
grid = GridGlitchSearch('grid', 'data', glitch_data.label, glitch_data.outdir,
                        F0s, F1s, F2s, delta_F0s, delta_F1s, tglitchs, Alphas,
                        Deltas, tref, tstart, tend)
grid.run()
grid.plot_2D('delta_F0', 'tglitch')
print grid.get_max_twoF()
