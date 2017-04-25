import pyfstat
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('thesis')

F0 = 30.0
F1 = -1e-10
F2 = 0
Alpha = 5e-3
Delta = 6e-2

tstart = 1000000000
duration = 100*86400
data_tstart = tstart - duration
data_tend = data_tstart + 3*duration
tref = .5*(data_tstart+data_tend)

h0 = 4e-24
sqrtSX = 1e-22

transient = pyfstat.Writer(
    label='transient', outdir='data', tref=tref, tstart=tstart, F0=F0, F1=F1,
    F2=F2, duration=duration, Alpha=Alpha, Delta=Delta, h0=h0, sqrtSX=sqrtSX,
    minStartTime=data_tstart, maxStartTime=data_tend)
transient.make_data()
print transient.predict_fstat()

DeltaF0 = 1e-7
DeltaF1 = 1e-13
VF0 = (np.pi * duration * DeltaF0)**2 / 3.0
VF1 = (np.pi * duration**2 * DeltaF1)**2 * 4/45.
print '\nV={:1.2e}, VF0={:1.2e}, VF1={:1.2e}\n'.format(VF0*VF1, VF0, VF1)

theta_prior = {'F0': {'type': 'unif',
                      'lower': F0-DeltaF0/2.,
                      'upper': F0+DeltaF0/2.},
               'F1': {'type': 'unif',
                      'lower': F1-DeltaF1/2.,
                      'upper': F1+DeltaF1/2.},
               'F2': F2,
               'Alpha': Alpha,
               'Delta': Delta
               }

ntemps = 3
log10temperature_min = -1
nwalkers = 100
nsteps = [100, 100]

mcmc = pyfstat.MCMCSearch(
    label='transient_search_initial_stage', outdir='data',
    sftfilepath='data/*transient*sft', theta_prior=theta_prior, tref=tref,
    minStartTime=data_tstart, maxStartTime=data_tend, nsteps=nsteps,
    nwalkers=nwalkers, ntemps=ntemps,
    log10temperature_min=log10temperature_min)
mcmc.run()
fig, ax = plt.subplots()
mcmc.write_par()
mcmc.generate_loudest()
mcmc.plot_cumulative_max(ax=ax)
ax.set_xlabel('Days from $t_\mathrm{start}$')
ax.legend_.remove()
fig.savefig('data/transient_search_initial_stage_twoFcumulative')
mcmc.print_summary()

theta_prior = {'F0': {'type': 'unif',
                      'lower': F0-DeltaF0/2.,
                      'upper': F0+DeltaF0/2.},
               'F1': {'type': 'unif',
                      'lower': F1-DeltaF1/2.,
                      'upper': F1+DeltaF1/2.},
               'F2': F2,
               'Alpha': Alpha,
               'Delta': Delta,
               'transient_tstart': {'type': 'unif',
                                    'lower': data_tstart,
                                    'upper': data_tend-0.2*duration},
               'transient_duration': {'type': 'halfnorm',
                                      'loc': 0.01*duration,
                                      'scale': 0.5*duration}
               }

nwalkers = 500
nsteps = [200, 200]

mcmc = pyfstat.MCMCTransientSearch(
    label='transient_search', outdir='data',
    sftfilepath='data/*transient*sft', theta_prior=theta_prior, tref=tref,
    minStartTime=data_tstart, maxStartTime=data_tend, nsteps=nsteps,
    nwalkers=nwalkers, ntemps=ntemps,
    log10temperature_min=log10temperature_min)
mcmc.run()
mcmc.plot_corner()
mcmc.print_summary()
