import pyfstat
import numpy as np
import matplotlib.pyplot as plt
from latex_macro_generator import write_to_macro

plt.style.use('paper')

F0 = 100.0
F1 = 0
F2 = 0
Alpha = 5.98
Delta = -0.1

# Properties of the GW data
sqrtSX = 1e-23
tstart = 1000000000
duration = 30 * 60 * 60
tend = tstart+duration
tref = .5*(tstart+tend)

psi = 2.25
phi = 0.2
cosi = 0.5

depth = 2
data_label = 'grided_frequency_depth_{:1.0f}'.format(depth)

h0 = sqrtSX / depth

data = pyfstat.Writer(
    label=data_label, outdir='data', tref=tref,
    tstart=tstart, F0=F0, F1=F1, F2=F2, duration=duration, Alpha=Alpha,
    Delta=Delta, h0=h0, sqrtSX=sqrtSX, detectors='H1', cosi=cosi, phi=phi,
    psi=psi)
data.make_data()

DeltaF0 = 1.5e-4
F0s = [F0-DeltaF0/2., F0+DeltaF0/2., DeltaF0/2000]
F1s = [F1]
F2s = [F2]
Alphas = [Alpha]
Deltas = [Delta]
search = pyfstat.GridSearch(
    'grided_frequency_search', 'data', 'data/*'+data_label+'-*.sft', F0s, F1s,
    F2s, Alphas, Deltas, tref, tstart, tend)
search.run()

fig, ax = plt.subplots()
xidx = search.keys.index('F0')
frequencies = np.unique(search.data[:, xidx])
twoF = search.data[:, -1]

#mismatch = np.sign(x-F0)*(duration * np.pi * (x - F0))**2 / 12.0
ax.plot(frequencies-F0, twoF, 'k', lw=0.5)

DeltaF = frequencies - F0
ax.set_ylabel('$\widetilde{2\mathcal{F}}$')
ax.set_xlabel('Template frequency')
dF0 = np.sqrt(12*1)/(np.pi*duration)
xticks = [-4/86400., -3/86400., -2/86400., -86400, 0, 
          1/86400., 2/86400., 3/86400., 4/86400.]
#ax.set_xticks(xticks)
#xticklabels = [r'$f_0 {-} \frac{4}{1\textrm{-day}}$', '$f_0$',
#               r'$f_0 {+} \frac{4}{1\textrm{-day}}$']
#ax.set_xticklabels(xticklabels)
ax.grid(linewidth=0.1)
plt.tight_layout()
fig.savefig('{}/{}_1D.png'.format(search.outdir, search.label), dpi=300)

write_to_macro('GridedFrequencySqrtSx', '{}'.format(
    pyfstat.helper_functions.texify_float(sqrtSX)),
               '../macros.tex')
write_to_macro('GridedFrequencyh0', '{}'.format(
    pyfstat.helper_functions.texify_float(h0)),
               '../macros.tex')
write_to_macro('GridedFrequencyT', '{}'.format(int(duration/86400)),
               '../macros.tex')
