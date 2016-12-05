import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from oct2py import octave
import glob

filenames = glob.glob("CollectedOutput/*.txt")

plt.style.use('paper')

Tspan = 100 * 86400


def Recovery(Tspan, Depth, twoFstar=60, detectors='H1,L1'):
    numDetectors = len(detectors.split(','))
    cmd = ("DetectionProbabilityStackSlide('Nseg', 1, 'Tdata', {},"
           "'misHist', createDeltaHist(0), 'avg2Fth', {}, 'detectors', '{}',"
           "'Depth', {})"
           ).format(numDetectors*Tspan, twoFstar, detectors, Depth)
    return octave.eval(cmd, verbose=False)


def binomialConfidenceInterval(N, K, confidence=0.95):
    cmd = '[fLow, fUpper] = binomialConfidenceInterval({}, {}, {})'.format(
        N, K, confidence)
    [l, u] =  octave.eval(cmd, verbose=False, return_both=True)[0].split('\n')
    return float(l.split('=')[1]), float(u.split('=')[1])

df_list = []
for fn in filenames:
    df = pd.read_csv(
        fn, sep=' ', names=['depth', 'h0', 'dF0', 'dF1', 'twoF_predicted',
                            'twoF', 'runTime'])
    df['CLUSTER_ID'] = fn.split('_')[1]
    df_list.append(df)
df = pd.concat(df_list)

twoFstar = 60
depths = np.unique(df.depth.values)
recovery_fraction = []
recovery_fraction_CI = []
for d in depths:
    twoFs = df[df.depth == d].twoF.values
    N = len(twoFs)
    K = np.sum(twoFs > twoFstar)
    print d, N, K
    recovery_fraction.append(K/float(N))
    [fLower, fUpper] = binomialConfidenceInterval(N, K)
    recovery_fraction_CI.append([fLower, fUpper])

yerr = np.abs(recovery_fraction - np.array(recovery_fraction_CI).T)
fig, ax = plt.subplots()
ax.errorbar(depths, recovery_fraction, yerr=yerr, fmt='sk', marker='s', ms=2,
            capsize=1, capthick=0.5, elinewidth=0.5,
            label='Monte-Carlo result')

fname = 'analytic_data.txt'
if os.path.isfile(fname):
    depths_smooth, recovery_analytic = np.loadtxt(fname)
else:
    depths_smooth = np.linspace(10, 550, 100)
    recovery_analytic = []
    for d in tqdm(depths_smooth):
        recovery_analytic.append(Recovery(Tspan, d, twoFstar))
    np.savetxt(fname, np.array([depths_smooth, recovery_analytic]))
depths_smooth = np.concatenate(([0], depths_smooth))
recovery_analytic = np.concatenate(([1], recovery_analytic))
ax.plot(depths_smooth, recovery_analytic, '-k', label='Theoretical maximum')


ax.set_ylim(0, 1.05)
ax.set_xlabel(r'Signal depth', size=10)
ax.set_ylabel(r'Recovered fraction', size=10)
ax.legend(loc=1, frameon=False)

fig.tight_layout()
fig.savefig('allsky_recovery.png')


fig, ax = plt.subplots()
ax.hist(df.runTime, bins=20)
ax.set_xlabel('runTime per follow-up [s]')
fig.savefig('runTimeHist.png')

