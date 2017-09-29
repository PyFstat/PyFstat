import pyfstat
import numpy as np
import matplotlib.pyplot as plt

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
data_label = 'weak_signal_follow_up_depth_{:1.0f}'.format(depth)

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
log10beta_min = -0.5
nwalkers = 100
scatter_val = 1e-10
nsteps = [100, 100]

mcmc = pyfstat.MCMCFollowUpSearch(
    label='weak_signal_follow_up', outdir='data',
    sftfilepattern='data/*'+data_label+'*sft', theta_prior=theta_prior, tref=tref,
    minStartTime=tstart, maxStartTime=tend, nwalkers=nwalkers, nsteps=nsteps,
    ntemps=ntemps, log10beta_min=log10beta_min,
    scatter_val=scatter_val)

fig, axes = plt.subplots(nrows=2, ncols=2)
mcmc.run(
    R=10, Nsegs0=100, subtractions=[F0, F1, Alpha, Delta], context='paper',
    fig=fig, axes=axes, plot_det_stat=False, return_fig=True)

mcmc.plot_corner(add_prior=True)
mcmc.print_summary()
