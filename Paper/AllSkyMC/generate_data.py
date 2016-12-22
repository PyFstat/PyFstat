import pyfstat
import numpy as np
import os
import sys
import time

ID = sys.argv[1]
outdir = sys.argv[2]

label = 'run_{}'.format(ID)
data_label = '{}_data'.format(label)
results_file_name = '{}/MCResults_{}.txt'.format(outdir, ID)

# Properties of the GW data
sqrtSX = 1e-23
tstart = 1000000000
Tspan = 100*86400
tend = tstart + Tspan

# Fixed properties of the signal
F0_center = 30
F1_center = -1e-10
F2 = 0
tref = .5*(tstart+tend)


VF0 = VF1 = 100
DeltaF0 = VF0 * np.sqrt(3)/(np.pi*Tspan)
DeltaF1 = VF1 * np.sqrt(45/4.)/(np.pi*Tspan**2)

DeltaAlpha = 0.02
DeltaDelta = 0.02

depths = np.linspace(100, 400, 9)
depths = [118.75, 156.25]

nsteps = 50
run_setup = [((nsteps, 0), 20, False),
             ((nsteps, 0), 11, False),
             ((nsteps, 0), 6, False),
             ((nsteps, 0), 3, False),
             ((nsteps, nsteps), 1, False)]


for depth in depths:
    h0 = sqrtSX / float(depth)
    F0 = F0_center + np.random.uniform(-0.5, 0.5)*DeltaF0
    F1 = F1_center + np.random.uniform(-0.5, 0.5)*DeltaF1
    Alpha_center = np.random.uniform(DeltaAlpha, 2*np.pi-DeltaAlpha)
    Delta_center = np.arccos(2*np.random.uniform(0, 1)-1)-np.pi/2
    Alpha = Alpha_center + np.random.uniform(-0.5, 0.5)*DeltaAlpha
    Delta = Delta_center + np.random.uniform(-0.5, 0.5)*DeltaDelta
    psi = np.random.uniform(-np.pi/4, np.pi/4)
    phi = np.random.uniform(0, 2*np.pi)
    cosi = np.random.uniform(-1, 1)

    data = pyfstat.Writer(
        label=data_label, outdir=outdir, tref=tref,
        tstart=tstart, F0=F0, F1=F1, F2=F2, duration=Tspan, Alpha=Alpha,
        Delta=Delta, h0=h0, sqrtSX=sqrtSX, psi=psi, phi=phi, cosi=cosi,
        detector='H1,L1')
    data.make_data()

    startTime = time.time()
    theta_prior = {'F0': {'type': 'unif',
                          'lower': F0_center-DeltaF0,
                          'upper': F0_center+DeltaF0},
                   'F1': {'type': 'unif',
                          'lower': F1_center-DeltaF1,
                          'upper': F1_center+DeltaF1},
                   'F2': F2,
                   'Alpha': {'type': 'unif',
                             'lower': Alpha_center-DeltaAlpha,
                             'upper': Alpha_center+DeltaAlpha},
                   'Delta': {'type': 'unif',
                             'lower': Delta_center-DeltaDelta,
                             'upper': Delta_center+DeltaDelta},
                   }

    ntemps = 2
    log10temperature_min = -1
    nwalkers = 100

    mcmc = pyfstat.MCMCFollowUpSearch(
        label=label, outdir=outdir,
        sftfilepath='{}/*{}*sft'.format(outdir, data_label),
        theta_prior=theta_prior,
        tref=tref, minStartTime=tstart, maxStartTime=tend,
        nwalkers=nwalkers, ntemps=ntemps,
        log10temperature_min=log10temperature_min)
    mcmc.run(run_setup=run_setup, create_plots=False, log_table=False,
             gen_tex_table=False)
    d, maxtwoF = mcmc.get_max_twoF()
    dF0 = F0 - d['F0']
    dF1 = F1 - d['F1']
    runTime = time.time() - startTime
    with open(results_file_name, 'a') as f:
        f.write('{} {:1.8e} {:1.8e} {:1.8e} {:1.8e} {}\n'
                .format(depth, h0, dF0, dF1, maxtwoF, runTime))
    os.system('rm {}/*{}*'.format(outdir, label))
