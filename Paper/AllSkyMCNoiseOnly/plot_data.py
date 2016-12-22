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


class maxtwoFinNoise_gen(rv_continuous):
    def _pdf(self, twoF, Ntrials):
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

maxtwoFinNoise = maxtwoFinNoise_gen(a=0)
Ntrials_effective, loc, scale = maxtwoFinNoise.fit(df.twoF.values, floc=0, fscale=1)
print 'Ntrials effective = {:1.2e}'.format(Ntrials_effective)
twoFsmooth = np.linspace(0, df.twoF.max(), 1000)
best_fit_pdf = maxtwoFinNoise.pdf(twoFsmooth, Ntrials_effective)
ax.plot(twoFsmooth, best_fit_pdf, '-r')

pval = 1e-6
twoFsmooth_HD = np.linspace(
    twoFsmooth[np.argmax(best_fit_pdf)], df.twoF.max(), 100000)
best_fit_pdf_HD = maxtwoFinNoise.pdf(twoFsmooth_HD, Ntrials_effective)
spacing = twoFsmooth_HD[1]-twoFsmooth_HD[0]
print twoFsmooth_HD[np.argmin(np.abs(best_fit_pdf_HD - pval))], spacing

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
