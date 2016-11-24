import pyfstat

F0 = 30.0
F1 = 0
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
theta_prior = {'F0': {'type': 'unif', 'lower': F0*(1-1e-6),
                      'upper': F0*(1+1e-6)},
               'F1': F1, #{'type': 'unif', 'lower': F1*(1+1e-2),
                      #'upper': F1*(1-1e-2)},
               'F2': F2,
               'Alpha': {'type': 'unif', 'lower': Alpha-1e-2,
                         'upper': Alpha+1e-2},
               'Delta': {'type': 'unif', 'lower': Delta-1e-2,
                         'upper': Delta+1e-2},
               }

ntemps = 3
log10temperature_min = -0.5
nwalkers = 100
scatter_val = 1e-10
nsteps = [100, 100]

mcmc = pyfstat.MCMCFollowUpSearch(
    label='weak_signal_follow_up', outdir='data',
    sftfilepath='data/*'+data_label+'*sft', theta_prior=theta_prior, tref=tref,
    minStartTime=tstart, maxStartTime=tend, nwalkers=nwalkers, nsteps=nsteps,
    ntemps=ntemps, log10temperature_min=log10temperature_min,
    scatter_val=scatter_val)
mcmc.run(R0=10, Vmin=100, subtractions=[F0, Alpha, Delta], context='paper')
mcmc.plot_corner(add_prior=True)
mcmc.print_summary()
