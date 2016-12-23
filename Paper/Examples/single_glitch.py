import pyfstat
import numpy as np
import matplotlib.pyplot as plt

sqrtSX = 1e-22
tstart = 1000000000
duration = 100*86400
tend = tstart+duration

# Define parameters of the Crab pulsar as an example
tref = .5*(tstart + tend)
F0 = 30.0
F1 = -1e-10
F2 = 0
Alpha = np.radians(83.6292)
Delta = np.radians(22.0144)

# Signal strength
depth = 10
h0 = sqrtSX / depth

VF0 = VF1 = 200
dF0 = np.sqrt(3)/(np.pi*duration)
dF1 = np.sqrt(45/4.)/(np.pi*duration**2)
DeltaF0 = VF0 * dF0
DeltaF1 = VF1 * dF1

# Next, taking the same signal parameters, we include a glitch half way through
dtglitch = duration/2.0
delta_F0 = 0.25*DeltaF0
delta_F1 = -0.1*DeltaF1

glitch_data = pyfstat.Writer(
    label='single_glitch', outdir='data', tref=tref, tstart=tstart, F0=F0,
    F1=F1, F2=F2, duration=duration, Alpha=Alpha, Delta=Delta, h0=h0,
    sqrtSX=sqrtSX, dtglitch=dtglitch, delta_F0=delta_F0, delta_F1=delta_F1)
glitch_data.make_data()

F0s = [F0-DeltaF0/2., F0+DeltaF0/2., 1*dF0]
F1s = [F1-DeltaF1/2., F1+DeltaF1/2., 1*dF1]
F2s = [F2]
Alphas = [Alpha]
Deltas = [Delta]
search = pyfstat.GridSearch(
    'single_glitch_F0F1_grid', 'data', 'data/*single_glitch*sft', F0s, F1s,
    F2s, Alphas, Deltas, tref, tstart, tend)
search.run()
search.plot_2D('F0', 'F1')

theta_prior = {'F0': {'type': 'unif', 'lower': F0-DeltaF0/2.,
                      'upper': F0+DeltaF0/2},
               'F1': {'type': 'unif', 'lower': F1-DeltaF1/2.,
                      'upper': F1+DeltaF1/2},
               'F2': F2,
               'Alpha': Alpha,
               'Delta': Delta
               }
ntemps = 3
log10temperature_min = -0.05
nwalkers = 100
nsteps = [500, 500]

mcmc = pyfstat.MCMCSearch(
    'single_glitch', 'data', sftfilepath='data/*_single_glitch*.sft',
    theta_prior=theta_prior, tref=tref, minStartTime=tstart, maxStartTime=tend,
    nsteps=nsteps, nwalkers=nwalkers, ntemps=ntemps,
    log10temperature_min=log10temperature_min)

mcmc.run()
mcmc.plot_corner(figsize=(3.2, 3.2))
mcmc.print_summary()


theta_prior = {'F0': {'type': 'unif', 'lower': F0-DeltaF0/2.,
                      'upper': F0+DeltaF0/2},
               'F1': {'type': 'unif', 'lower': F1-DeltaF1/2.,
                      'upper': F1+DeltaF1/2},
               'F2': F2,
               'Alpha': Alpha,
               'Delta': Delta,
               'tglitch': {'type': 'unif', 'lower': tstart+0.1*duration,
                           'upper': tend-0.1*duration},
               'delta_F0': {'type': 'halfnorm', 'loc': 0, 'scale': DeltaF0},
               'delta_F1': {'type': 'norm', 'loc': 0, 'scale': DeltaF1},
               }
ntemps = 3
log10temperature_min = -0.1
nwalkers = 100
nsteps = [1000, 1000]
glitch_mcmc = pyfstat.MCMCGlitchSearch(
    'single_glitch_glitchSearch', 'data',
    sftfilepath='data/*_single_glitch*.sft', theta_prior=theta_prior,
    tref=tref, minStartTime=tstart, maxStartTime=tend, nsteps=nsteps,
    nwalkers=nwalkers, ntemps=ntemps,
    log10temperature_min=log10temperature_min)
glitch_mcmc.run()
glitch_mcmc.plot_corner(figsize=(6, 6))
glitch_mcmc.print_summary()

