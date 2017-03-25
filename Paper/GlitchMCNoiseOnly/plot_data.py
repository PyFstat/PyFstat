import pyfstat
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from oct2py import octave
import glob
from scipy.stats import rv_continuous, chi2
from latex_macro_generator import write_to_macro

filenames = glob.glob("CollectedOutput/*.txt")

plt.style.use('paper')

Tspan = 100 * 86400

df_list = []
for fn in filenames:
    df = pd.read_csv(
        fn, sep=' ', names=['nglitches', 'dF0', 'dF1', 'twoF', 'runTime'])
    df['CLUSTER_ID'] = fn.split('_')[1]
    df_list.append(df)
df = pd.concat(df_list)

fig, ax = plt.subplots()
for ng in np.unique(df.nglitches.values):
    print 'ng={}'.format(ng)
    Nsamples = len(df[df.nglitches == ng])
    MaxtwoF = df[df.nglitches == ng].twoF.max()
    print 'Number of samples = ', Nsamples
    print 'Max twoF', MaxtwoF

    ax.hist(df[df.nglitches == ng].twoF, bins=40, histtype='step', normed=True,
            linewidth=1, label='$N_\mathrm{{glitches}}={}$'.format(ng))

    write_to_macro('DirectedMC{}GlitchNoiseOnlyMaximum'.format(ng),
                   '{:1.1f}'.format(MaxtwoF), '../macros.tex')
    write_to_macro('DirectedMC{}GlitchNoiseN'.format(ng),
                   '{:1.0f}'.format(Nsamples), '../macros.tex')

ax.set_xlabel('$\widehat{2\mathcal{F}}$')
ax.set_xlim(0, 90)
ax.legend(frameon=False, fontsize=6)
fig.tight_layout()
fig.savefig('glitch_noise_twoF_histogram.png')


