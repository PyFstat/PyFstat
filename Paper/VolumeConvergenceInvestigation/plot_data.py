import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from oct2py import octave
import glob
import scipy.stats

filenames = glob.glob("CollectedOutput/*.txt")

plt.style.use('paper')

Tspan = 100 * 86400

df_list = []
for fn in filenames:
    if '3356003' in fn:
        continue
    else:
        df = pd.read_csv(
            fn, sep=' ', names=['V', 'dF0', 'dF1', 'F0_std', 'F1_std', 'DeltaF0',
                                'DeltaF1', 'MaxtwoF', 'twoF_PM', 'cdF0', 'cdF1', 'runTime'])
        df['mismatch'] = (df.twoF_PM - df.MaxtwoF)/df.twoF_PM
        df['F0_fraction'] = df.F0_std / df.DeltaF0
        df['F1_fraction'] = df.F1_std / df.DeltaF1
        df['CLUSTER_ID'] = fn.split('_')[1]
        df['RUN_ID'] = fn.split('_')[2]
        df_list.append(df)

df = pd.concat(df_list)

df.sort_values('V', inplace=True)

df['cd_ave'] = df[['cdF0', 'cdF1']].mean(axis=1)
df['fraction_ave'] = df[['F0_fraction', 'F1_fraction']].mean(axis=1)
df = df[df.V <= 500]

fig, (ax2, ax3) = plt.subplots(nrows=2, figsize=(3.2, 4), sharex=True)

print df.groupby('V')['dF0'].count()


Vs = df['V'].unique()
mismatch_ave = df.groupby('V')['mismatch'].mean()
ave_cd_ave = df.groupby('V')['cd_ave'].apply(scipy.stats.mstats.gmean)
cdF0_ave = df.groupby('V')['cdF0'].mean()
cdF1_ave = df.groupby('V')['cdF1'].mean()
ave_fraction_ave = df.groupby('V')['fraction_ave'].median()
F0_fraction_ave = df.groupby('V')['F0_fraction'].mean()
F1_fraction_ave = df.groupby('V')['F1_fraction'].mean()

ax2.plot(df.V, df.cd_ave, 'o', color='k', alpha=0.5)
#ax2.plot(Vs, ave_cd_ave, color='r')
ax2.set_ylim(0, 10)
ax2.set_ylabel(r'$\langle \textrm{PSRF} \rangle$')

ax3.plot(df.V, df.fraction_ave, 'o', color='k', alpha=0.2)
#ax3.plot(Vs, ave_fraction_ave, color='r')
ax3.set_ylabel(r'$\langle \textrm{posterior std. / prior width} \rangle$')
ax3.set_xlabel(
    r'$\mathcal{V}_\mathrm{PE}^{(0)}=\mathcal{V}_\mathrm{PE}^{(1)}=\sqrt{\mathcal{V}}$')
ax3.set_xlim(0, 525)

ax3.set_xticks(Vs)

fig.tight_layout()
fig.savefig('VolumeConvergenceInvestigation.png')
