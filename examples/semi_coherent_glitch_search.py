import pyfstat

F0 = 30.0
F1 = -1e-10
F2 = 0
Alpha = 5e-3
Delta = 6e-2
tref = 362750407.0

tstart = 1000000000
duration = 100*86400
tend = tstart + duration

theta_prior = {'F0': {'type': 'norm', 'loc': F0, 'scale': abs(1e-6*F0)},
               'F1': {'type': 'norm', 'loc': F1, 'scale': abs(1e-6*F1)},
               'F2': F2,
               'Alpha': Alpha,
               'Delta': Delta,
               'delta_F0': {'type': 'halfnorm', 'loc': 0,
                            'scale': 1e-5*F0},
               'delta_F1': 0,
               'tglitch': {'type': 'unif',
                           'lower': tstart+0.1*duration,
                           'upper': tstart+0.9*duration},
               }

nwalkers = 500
nsteps = [1000, 1000, 1000]

mcmc = pyfstat.MCMCGlitchSearch(
    'semi_coherent_glitch_search', 'data', sftlabel='glitch', sftdir='data',
    theta_prior=theta_prior, tref=tref, tstart=tstart, tend=tend,
    nsteps=nsteps, nwalkers=nwalkers, scatter_val=1e-10, nglitch=1)

mcmc.run()
mcmc.plot_corner(add_prior=True)
mcmc.print_summary()
