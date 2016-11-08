import pyfstat

# Define parameters of the Crab pulsar as an example
F0 = 30.0
F1 = -1e-10
F2 = 0
Alpha = 5e-3
Delta = 6e-2
tref = 362750407.0

# Properties of the GW data
sqrtSX = 1e-23
tstart = 1000000000
duration = 100*86400
tend = tstart+duration

depth = 50

h0 = sqrtSX / depth

data = pyfstat.Writer(
    label='depth_{:1.0f}'.format(depth), outdir='data', tref=tref,
    tstart=tstart, F0=F0, F1=F1, F2=F2, duration=duration, Alpha=Alpha,
    Delta=Delta, h0=h0, sqrtSX=sqrtSX)
data.make_data()

# The predicted twoF, given by lalapps_predictFstat can be accessed by
twoF = data.predict_fstat()
print 'Predicted twoF value: {}\n'.format(twoF)

# Search
theta_prior = {'F0': {'type': 'unif', 'lower': F0*(1-1e-4),
                      'upper': F0*(1+1e-4)},
               'F1': {'type': 'unif', 'lower': F1*(1+1e-2),
                      'upper': F1*(1-1e-2)},
               'F2': F2,
               'Alpha': {'type': 'unif', 'lower': Alpha-1e-2,
                         'upper': Alpha+1e-2},
               'Delta': {'type': 'unif', 'lower': Delta-5e-2,
                         'upper': Delta+5e-2},
               }

ntemps = 1
log10temperature_min = -1
nwalkers = 100
run_setup = [(1000, 50),
             (1000, 30),
             (1000, 20),
             (1000, 15),
             (1000, 10),
             (1000, 5),
             (1000, 1),
             ((1000, 1000), 1, True)]

mcmc = pyfstat.MCMCFollowUpSearch(
    label='weak_signal_follow_up', outdir='data',
    sftfilepath='data/*depth*sft', theta_prior=theta_prior, tref=tref,
    minStartTime=tstart, maxStartTime=tend, nwalkers=nwalkers,
    ntemps=ntemps, log10temperature_min=log10temperature_min)
mcmc.run(run_setup)
mcmc.plot_corner(add_prior=True)
mcmc.print_summary()
mcmc.generate_loudest()
