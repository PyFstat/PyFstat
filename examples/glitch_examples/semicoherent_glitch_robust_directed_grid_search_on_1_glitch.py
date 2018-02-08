import pyfstat
import numpy as np
import matplotlib.pyplot as plt
from make_simulated_data import tstart, duration, tref, F0, F1, F2, Alpha, Delta, outdir

try:
    from gridcorner import gridcorner
except ImportError:
    raise ImportError(
        "Python module 'gridcorner' not found, please install from "
        "https://gitlab.aei.uni-hannover.de/GregAshton/gridcorner")

label = 'semicoherent_glitch_robust_directed_grid_search_on_1_glitch'

plt.style.use('paper')

Nstar = 1000
F0_width = np.sqrt(Nstar)*np.sqrt(12)/(np.pi*duration)
F1_width = np.sqrt(Nstar)*np.sqrt(180)/(np.pi*duration**2)
N = 30
F0s = [F0-F0_width/2., F0+F0_width/2., F0_width/N]
F1s = [F1-F1_width/2., F1+F1_width/2., F1_width/N]
F2s = [F2]
Alphas = [Alpha]
Deltas = [Delta]

max_delta_F0 = 1e-5
tglitchs = [tstart+0.1*duration, tstart+0.9*duration, 0.8*float(duration)/N]
delta_F0s = [0, max_delta_F0, max_delta_F0/N]
delta_F1s = [0]

search = pyfstat.GridGlitchSearch(
    label, outdir, 'data/*1_glitch*sft', F0s=F0s, F1s=F1s, F2s=F2s,
    Alphas=Alphas, Deltas=Deltas, tref=tref, minStartTime=tstart,
    maxStartTime=tstart+duration, tglitchs=tglitchs, delta_F0s=delta_F0s,
    delta_F1s=delta_F1s)
search.run()

F0_vals = np.unique(search.data[:, 0]) - F0
F1_vals = np.unique(search.data[:, 1]) - F1
delta_F0s_vals = np.unique(search.data[:, 5])
tglitch_vals = np.unique(search.data[:, 7])
tglitch_vals_days = (tglitch_vals-tstart) / 86400.

twoF = search.data[:, -1].reshape((len(F0_vals), len(F1_vals),
                                   len(delta_F0s_vals), len(tglitch_vals)))
xyz = [F0_vals, F1_vals, delta_F0s_vals, tglitch_vals_days]
labels = ['$f - f_0$\n[Hz]', '$\dot{f} - \dot{f}_0$\n[Hz/s]',
          '$\delta f$\n[Hz]', '$t^g_0$\n[days]', '$\widehat{2\mathcal{F}}$']
fig, axes = gridcorner(
    twoF, xyz, projection='log_mean', whspace=0.1, factor=1.2, labels=labels)
fig.savefig('{}/{}_projection_matrix.png'.format(outdir, label),
            bbox_inches='tight')
