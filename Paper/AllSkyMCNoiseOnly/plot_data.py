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


def maxtwoFinNoise(twoF, Ntrials):
    F = twoF/2.0
    alpha = (1 + F)*np.exp(-F)
    a = Ntrials/2.0*F*np.exp(-F)
    b = (1 - alpha)**(Ntrials-1)
    return a*b


df_list = []
for fn in filenames:
    df = pd.read_csv(
        fn, sep=' ', names=['dF0', 'dF1', 'twoF', 'runTime'])
    df['CLUSTER_ID'] = fn.split('_')[1]
    df_list.append(df)
df = pd.concat(df_list)
print 'Number of samples = ', len(df)

fig, ax = plt.subplots()
ax.hist(df.twoF, bins=50, histtype='step', color='k', normed=True, linewidth=1)
twoFsmooth = np.linspace(0, df.twoF.max(), 100)
# ax.plot(twoFsmooth, maxtwoFinNoise(twoFsmooth, 8e5), '-r')
ax.set_xlabel('$\widetilde{2\mathcal{F}}$')
ax.set_xlim(0, 60)
fig.tight_layout()
fig.savefig('allsky_noise_twoF_histogram.png')

from latex_macro_generator import write_to_macro
write_to_macro('AllSkyMCNoiseOnlyMaximum', '{:1.1f}'.format(np.max(df.twoF)),
               '../macros.tex')
write_to_macro('AllSkyMCNoiseN', len(df), '../macros.tex')
