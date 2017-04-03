import pyfstat
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('paper')

F0 = 30.0
F1 = 0
F2 = 0
Alpha = 1.0
Delta = 1.5

# Properties of the GW data
sqrtSX = 1e-23
tstart = 1000000000
duration = 100*86400
tend = tstart+duration
tref = .5*(tstart+tend)

depth = 70
data_label = 'grided_frequency_depth_{:1.0f}'.format(depth)

h0 = sqrtSX / depth

data = pyfstat.Writer(
    label=data_label, outdir='data', tref=tref,
    tstart=tstart, F0=F0, F1=F1, F2=F2, duration=duration, Alpha=Alpha,
    Delta=Delta, h0=h0, sqrtSX=sqrtSX)
data.make_data()

m = 1
dF0 = np.sqrt(12*m)/(np.pi*duration)
DeltaF0 = 30*dF0
F0s = [F0-DeltaF0/2., F0+DeltaF0/2., 1e-2*dF0]
F1s = [F1]
F2s = [F2]
Alphas = [Alpha]
Deltas = [Delta]
search = pyfstat.GridSearch(
    'grided_frequency_search', 'data', 'data/*'+data_label+'*sft', F0s, F1s,
    F2s, Alphas, Deltas, tref, tstart, tend)
search.run()

fig, ax = plt.subplots()
xidx = search.keys.index('F0')
frequencies = np.unique(search.data[:, xidx])
twoF = search.data[:, -1]

#mismatch = np.sign(x-F0)*(duration * np.pi * (x - F0))**2 / 12.0
ax.plot(frequencies, twoF, 'k', lw=0.8)
DeltaF = frequencies - F0
sinc = np.sin(np.pi*DeltaF*duration)/(np.pi*DeltaF*duration)
A = np.abs((np.max(twoF)-4)*sinc**2 + 4)
ax.plot(frequencies, A, 'r', lw=1.2, dashes=(0.2, 0.2))
ax.set_ylabel('$\widetilde{2\mathcal{F}}$')
ax.set_xlabel('Frequency')
ax.set_xlim(F0s[0], F0s[1])
dF0 = np.sqrt(12*1)/(np.pi*duration)
xticks = [F0-10*dF0, F0, F0+10*dF0]
ax.set_xticks(xticks)
xticklabels = [r'$f_0 {-} 10\Delta f(\tilde{\mu}=1)$', '$f_0$',
               r'$f_0 {+} 10\Delta f(\tilde{\mu}=1)$']
ax.set_xticklabels(xticklabels)
plt.tight_layout()
fig.savefig('{}/{}_1D.png'.format(search.outdir, search.label), dpi=300)
