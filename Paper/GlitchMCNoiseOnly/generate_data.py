import pyfstat
import numpy as np
import os
import sys
import time

nglitch = 2

ID = sys.argv[1]
outdir = sys.argv[2]

label = 'run_{}'.format(ID)
data_label = '{}_data'.format(label)
results_file_name = '{}/NoiseOnlyMCResults_{}.txt'.format(outdir, ID)

# Properties of the GW data
sqrtSX = 1e-23
tstart = 1000000000
Tspan = 100*86400
tend = tstart + Tspan

# Fixed properties of the signal
F0_center = 30
F1_center = -1e-10
F2 = 0
Alpha = np.radians(83.6292)
Delta = np.radians(22.0144)
tref = .5*(tstart+tend)

VF0 = VF1 = 200
dF0 = np.sqrt(3)/(np.pi*Tspan)
dF1 = np.sqrt(45/4.)/(np.pi*Tspan**2)
DeltaF0 = VF0 * dF0
DeltaF1 = VF1 * dF1

nsteps = 25
run_setup = [((nsteps, 0), 20, False),
             ((nsteps, 0), 7, False),
             ((nsteps, 0), 2, False),
             ((nsteps, nsteps), 1, False)]

h0 = 0
F0 = F0_center + np.random.uniform(-0.5, 0.5)*DeltaF0
F1 = F1_center + np.random.uniform(-0.5, 0.5)*DeltaF1

psi = np.random.uniform(-np.pi/4, np.pi/4)
phi = np.random.uniform(0, 2*np.pi)
cosi = np.random.uniform(-1, 1)

# Next, taking the same signal parameters, we include a glitch half way through
dtglitch = Tspan/2.0
delta_F0 = 0.25*DeltaF0
delta_F1 = -0.1*DeltaF1

glitch_data = pyfstat.Writer(
    label=data_label, outdir=outdir, tref=tref, tstart=tstart, F0=F0,
    F1=F1, F2=F2, duration=Tspan, Alpha=Alpha, Delta=Delta, h0=h0,
    sqrtSX=sqrtSX, dtglitch=dtglitch, delta_F0=delta_F0, delta_F1=delta_F1)
glitch_data.make_data()


startTime = time.time()
theta_prior = {'F0': {'type': 'unif', 'lower': F0-DeltaF0/2., # PROBLEM
                      'upper': F0+DeltaF0/2},
               'F1': {'type': 'unif', 'lower': F1-DeltaF1/2.,
                      'upper': F1+DeltaF1/2},
               'F2': F2,
               'Alpha': Alpha,
               'Delta': Delta,
               'tglitch': {'type': 'unif', 'lower': tstart+0.1*Tspan,
                           'upper': tend-0.1*Tspan},
               'delta_F0': {'type': 'halfnorm', 'loc': 0, 'scale': DeltaF0},
               'delta_F1': {'type': 'norm', 'loc': 0, 'scale': DeltaF1},
               }
ntemps = 2
log10temperature_min = -0.1
nwalkers = 100
nsteps = [500, 500]
glitch_mcmc = pyfstat.MCMCGlitchSearch(
    label=label, outdir=outdir,
    sftfilepath='{}/*{}*sft'.format(outdir, data_label),
    theta_prior=theta_prior,
    tref=tref, minStartTime=tstart, maxStartTime=tend, nsteps=nsteps,
    nwalkers=nwalkers, ntemps=ntemps, nglitch=nglitch,
    log10temperature_min=log10temperature_min)
glitch_mcmc.run(run_setup=run_setup, create_plots=False, log_table=False,
                gen_tex_table=False)
glitch_mcmc.print_summary()
d, maxtwoF = glitch_mcmc.get_max_twoF()
dF0 = F0 - d['F0']
dF1 = F1 - d['F1']
#tglitch = d['tglitch']
#R = (tglitch - tstart) / Tspan
#delta_F0 = d['delta_F0']
#delta_F1 = d['delta_F1']
runTime = time.time() - startTime
with open(results_file_name, 'a') as f:
    f.write('{} {:1.8e} {:1.8e} {:1.8e} {:1.1f}\n'
            .format(nglitch, dF0, dF1, maxtwoF, runTime))
os.system('rm {}/*{}*'.format(outdir, label))
