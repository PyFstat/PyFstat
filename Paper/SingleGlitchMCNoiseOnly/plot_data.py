import pyfstat
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from oct2py import octave
import glob
from scipy.stats import rv_continuous, chi2

filenames = glob.glob("CollectedOutput/*.txt")

plt.style.use('paper')

Tspan = 100 * 86400

df_list = []
for fn in filenames:
    df = pd.read_csv(
        fn, sep=' ', names=['dF0', 'dF1', 'R', 'delta_F0', 'delta_F1',
                            'twoF', 'runTime'])
    df['CLUSTER_ID'] = fn.split('_')[1]
    df_list.append(df)
df = pd.concat(df_list)
print 'Number of samples = ', len(df)
print 'Max twoF', df.twoF.max()

fig, ax = plt.subplots()
ax.hist(df.twoF, bins=50, histtype='step', color='k', normed=True, linewidth=1,
        label='Monte-Carlo histogram')

ax.set_xlabel('$\widetilde{2\mathcal{F}}$')
ax.set_xlim(0, 90)
ax.legend(frameon=False, fontsize=6)
fig.tight_layout()
fig.savefig('single_glitch_noise_twoF_histogram.png')

#from latex_macro_generator import write_to_macro
#write_to_macro('DirectedMCNoiseOnlyMaximum', '{:1.1f}'.format(np.max(df.twoF)),
#               '../macros.tex')
#write_to_macro('DirectedMCNoiseN', len(df), '../macros.tex')
