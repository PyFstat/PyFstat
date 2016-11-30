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
sqrtSX = 2e-23
tstart = 1000000000
Tspan = 100*86400
tend = tstart + Tspan

# Fixed properties of the signal
F0_center = 30
F1_center = 1e-10
F2 = 0
Alpha = 5e-3
Delta = 6e-2
tref = .5*(tstart+tend)


VF0 = VF1 = 100
DeltaF0 = VF0 * np.sqrt(3)/(np.pi*Tspan)
DeltaF1 = VF1 * np.sqrt(45/4.)/(np.pi*Tspan**2)

depths = np.linspace(100, 400, 7)
depths = [125, 175]

nsteps = 20
run_setup = [((nsteps, 0), 20, False),
             ((nsteps, 0), 7, False),
             ((nsteps, 0), 2, False),
             ((nsteps, nsteps), 1, False)]

for depth in depths:
    startTime = time.time()
    h0 = sqrtSX / float(depth)
    r = np.random.uniform(0, 1)
    theta = np.random.uniform(0, 2*np.pi)
    F0 = F0_center + np.random.uniform(-0.5, 0.5)*DeltaF0
    F1 = F1_center + np.random.uniform(-0.5, 0.5)*DeltaF1

    psi = np.random.uniform(-np.pi/4, np.pi/4)
    phi = np.random.uniform(0, 2*np.pi)
    cosi = np.random.uniform(-1, 1)

    data = pyfstat.Writer(
        label=data_label, outdir=outdir, tref=tref,
        tstart=tstart, F0=F0, F1=F1, F2=F2, duration=Tspan, Alpha=Alpha,
        Delta=Delta, h0=h0, sqrtSX=sqrtSX, psi=psi, phi=phi, cosi=cosi,
        detector='H1,L1')
    data.make_data()
    predicted_twoF = data.predict_fstat()

    theta_prior = {'F0': {'type': 'unif',
                          'lower': F0-DeltaF0/2.,
                          'upper': F0+DeltaF0/2.},
                   'F1': {'type': 'unif',
                          'lower': F1-DeltaF1/2.,
                          'upper': F1+DeltaF1/2.},
                   'F2': F2,
                   'Alpha': Alpha,
                   'Delta': Delta
                   }

    ntemps = 1
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
        f.write('{} {:1.8e} {:1.8e} {:1.8e} {:1.8e} {:1.8e} {}\n'
                .format(depth, h0, dF0, dF1, predicted_twoF, maxtwoF, runTime))
    os.system('rm {}/*{}*'.format(outdir, label))
