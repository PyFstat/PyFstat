import numpy as np
from pyfstat import MCMCSearch

F0 = 30.0
F1 = -1e-10
F2 = 0
Alpha = np.radians(83.6292)
Delta = np.radians(22.0144)

tref = 362750407.0

tstart = 1000000000
duration = 100*86400
tend = tstart + duration

theta_prior = {'F0': {'type': 'unif', 'lower': F0-1e-4, 'upper': F0+1e-4},
               'F1': {'type': 'unif', 'lower': F1*(1+1e-3), 'upper': F1*(1-1e-3)},
               'F2': F2,
               'Alpha': Alpha,
               'Delta': Delta
               }

ntemps = 2
log10beta_min = -0.01
nwalkers = 100
nsteps = [500, 500]

mcmc = MCMCSearch('fully_coherent_search_using_MCMC_on_glitching_data', 'data',
                  sftfilepattern='data/*_glitch*.sft',
                  theta_prior=theta_prior, tref=tref, minStartTime=tstart, maxStartTime=tend,
                  nsteps=nsteps, nwalkers=nwalkers, ntemps=ntemps,
                  log10beta_min=log10beta_min)
mcmc.run()
mcmc.plot_corner(add_prior=True)
mcmc.print_summary()
