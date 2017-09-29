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

theta_prior = {'F0': {'type': 'unif', 'lower': F0*(1-1e-6), 'upper': F0*(1+1e-5)},
               'F1': {'type': 'unif', 'lower': F1*(1+1e-2), 'upper': F1*(1-1e-2)},
               'F2': F2,
               'Alpha': Alpha,
               'Delta': Delta
               }

ntemps = 1
log10beta_min = -1
nwalkers = 100
run_setup = [(1000, 50), (1000, 25), (1000, 1, False),
             ((500, 500), 1, True)]

mcmc = pyfstat.MCMCFollowUpSearch(
    label='follow_up', outdir='data',
    sftfilepattern='data/*basic*sft', theta_prior=theta_prior, tref=tref,
    minStartTime=tstart, maxStartTime=tend, nwalkers=nwalkers,
    ntemps=ntemps, log10beta_min=log10beta_min)
mcmc.run(run_setup, gen_tex_table=True)
#mcmc.run(Nsegs0=50)
mcmc.plot_corner(add_prior=True)
mcmc.print_summary()
