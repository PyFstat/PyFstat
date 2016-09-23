from pyfstat import MCMCSearch
import numpy as np

F0 = 30.0
F1 = -1e-10
F2 = 0
Alpha = 5e-3
Delta = 6e-2
tref = 362750407.0

tstart = 1000000000
duration = 100*86400
tend = tstart = duration

theta_prior = {'F0': {'type': 'unif', 'lower': F0-5e-5,
                      'upper': F0+5e-5},
               'F1': {'type': 'norm', 'loc': F1, 'scale': abs(1e-6*F1)},
               'F2': F2,
               'Alpha': Alpha,
               'Delta': Delta
               }

ntemps = 10
betas = np.logspace(0, -30, ntemps)
nwalkers = 500
nsteps = [100, 100, 100]

mcmc = MCMCSearch('fully_coherent_on_glitching_data', 'data',
                  sftlabel='glitch', sftdir='data',
                  theta_prior=theta_prior, tref=tref, tstart=tstart, tend=tend,
                  nsteps=nsteps, nwalkers=nwalkers, ntemps=ntemps, betas=betas,
                  scatter_val=1e-6)
mcmc.run()
mcmc.plot_corner(add_prior=True)
mcmc.print_summary()
