import pyfstat
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

F0 = 30.0
F1 = -1e-10
F2 = 0
Alpha = 1.0
Delta = 0.5

# Properties of the GW data
sqrtSX = 1e-23
tstart = 1000000000
duration = 100*86400
tend = tstart+duration
tref = .5*(tstart+tend)

depth = 50
data_label = 'follow_up'

h0 = sqrtSX / depth

data = pyfstat.Writer(
    label=data_label, outdir='data', tref=tref,
    tstart=tstart, F0=F0, F1=F1, F2=F2, duration=duration, Alpha=Alpha,
    Delta=Delta, h0=h0, sqrtSX=sqrtSX)
data.make_data()

# The predicted twoF, given by lalapps_predictFstat can be accessed by
twoF = data.predict_fstat()
print 'Predicted twoF value: {}\n'.format(twoF)

# Search
VF0 = VF1 = 500
DeltaF0 = VF0 * np.sqrt(3)/(np.pi*duration)
DeltaF1 = VF1 * np.sqrt(45/4.)/(np.pi*duration**2)
DeltaAlpha = 1e-1
DeltaDelta = 1e-1
theta_prior = {'F0': {'type': 'unif', 'lower': F0-DeltaF0/2.,
                      'upper': F0+DeltaF0/2},
               'F1': {'type': 'unif', 'lower': F1-DeltaF1/2.,
                      'upper': F1+DeltaF1/2},
               'F2': F2,
               'Alpha': {'type': 'unif', 'lower': Alpha-DeltaAlpha,
                         'upper': Alpha+DeltaAlpha},
               'Delta': {'type': 'unif', 'lower': Delta-DeltaDelta,
                         'upper': Delta+DeltaDelta},
               }

ntemps = 3
log10temperature_min = -0.5
nwalkers = 100
scatter_val = 1e-10
nsteps = [200, 200]

mcmc = pyfstat.MCMCFollowUpSearch(
    label='follow_up', outdir='data',
    sftfilepath='data/*'+data_label+'*sft', theta_prior=theta_prior, tref=tref,
    minStartTime=tstart, maxStartTime=tend, nwalkers=nwalkers, nsteps=nsteps,
    ntemps=ntemps, log10temperature_min=log10temperature_min,
    scatter_val=scatter_val)

fig, axes = plt.subplots(nrows=2, ncols=2)
fig, axes = mcmc.run(
    R=10, Nsegs0=100, subtractions=[F0, F1, Alpha, Delta], labelpad=0.01,
    fig=fig, axes=axes, plot_det_stat=False, return_fig=True)
axes[3].set_xlabel(r'$\textrm{Number of steps}$', labelpad=0.1)
for ax in axes:
    ax.set_xlim(0, axes[0].get_xlim()[-1])
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
fig.tight_layout()
fig.savefig('{}/{}_walkers.png'.format(mcmc.outdir, mcmc.label), dpi=400)

mcmc.plot_corner(add_prior=True)
mcmc.print_summary()
