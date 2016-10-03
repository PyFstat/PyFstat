from pyfstat import MCMCSearch

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
               'Delta': Delta
               }

ntemps = 1
nwalkers = 100
nsteps = [100, 500, 1000]

mcmc = MCMCSearch('fully_coherent', 'data', sftfilepath='data/*basic*sft',
                  theta_prior=theta_prior, tref=tref, tstart=tstart, tend=tend,
                  nsteps=nsteps, nwalkers=nwalkers, ntemps=ntemps,
                  scatter_val=1e-10)
mcmc.run()
mcmc.plot_corner()
mcmc.print_summary()
