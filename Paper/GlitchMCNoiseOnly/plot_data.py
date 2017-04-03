import pyfstat
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from oct2py import octave
import glob
from scipy.stats import rv_continuous, chi2
from scipy.special import gammaincc
from latex_macro_generator import write_to_macro


def CF_twoFmax_integrand(theta, twoFmax, Nt):
    Fmax = twoFmax/2.0
    return np.exp(1j*theta*twoFmax)*Nt/2.0*Fmax*np.exp(-Fmax)*(1-(1+Fmax)*np.exp(-Fmax))**(Nt-1.)


def pdf_twoFhat(twoFhat, Ng, Nts, twoFtildemax=100, dtwoF=0.05):
    if np.ndim(Nts) == 0:
        Nts = np.zeros(Ng+1) + Nts

    twoFtildemax_int = np.arange(0, twoFtildemax, dtwoF)
    theta_int = np.arange(-1./dtwoF, 1./dtwoF, 1./twoFtildemax)

    CF_twoFtildemax_theta = np.array(
        [[np.trapz(CF_twoFmax_integrand(t, twoFtildemax_int, Nt), twoFtildemax_int)
          for t in theta_int]
         for Nt in Nts])

    CF_twoFhat_theta = np.prod(CF_twoFtildemax_theta, axis=0)
    print CF_twoFhat_theta.shape, theta_int.shape
    pdf = (1/(2*np.pi)) * np.array(
        [np.trapz(np.exp(-1j*theta_int*twoFhat_val)*CF_twoFhat_theta,
                  theta_int) for twoFhat_val in twoFhat])
    print np.trapz(pdf.real, x=twoFhat)
    return pdf

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

colors = ['C0', 'C1']
fig, ax = plt.subplots()
handles = []
labels = []
for ng, c in zip(np.unique(df.nglitches.values), colors):
    print 'ng={}'.format(ng)
    df_temp = df[df.nglitches == ng]
    #df_temp = df_temp[[str(x).isalpha() for x in df_temp.CLUSTER_ID.values]]
    print df_temp.tail()
    Nsamples = len(df_temp)
    MaxtwoF = df_temp.twoF.max()
    print 'Number of samples = ', Nsamples
    print 'Max twoF', MaxtwoF
    print np.any(np.isnan(df_temp.twoF.values))
    ax.hist(df_temp.twoF, bins=40, histtype='stepfilled',
            normed=True, align='mid', alpha=0.5,
            linewidth=1, label='$N_\mathrm{{glitches}}={}$'.format(ng),
            color=c)

    write_to_macro('DirectedMC{}GlitchNoiseOnlyMaximum'.format(ng),
                   '{:1.1f}'.format(MaxtwoF), '../macros.tex')
    write_to_macro('DirectedMC{}GlitchNoiseN'.format(ng),
                   '{:1.0f}'.format(Nsamples), '../macros.tex')

    twoFmax = np.linspace(0, 100, 200)
    ax.plot(twoFmax, pdf_twoFhat(twoFmax, ng, Nsamples,
                                 twoFtildemax=2*MaxtwoF, dtwoF=0.1),
            color=c, label='$N_\mathrm{{glitches}}={}$ predicted'.format(ng))

ax.set_xlabel('$\widehat{2\mathcal{F}}$')
ax.set_xlim(0, 90)
handles, labels = ax.get_legend_handles_labels()
idxs = np.argsort(labels)
ax.legend(np.array(handles)[idxs], np.array(labels)[idxs], frameon=False,
          fontsize=6)
fig.tight_layout()
fig.savefig('glitch_noise_twoF_histogram.png')


