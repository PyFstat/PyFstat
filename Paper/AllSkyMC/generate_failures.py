import pyfstat
import numpy as np
import os
import time

outdir = 'data'

label = 'run_failures'
data_label = '{}_data'.format(label)
results_file_name = '{}/MCResults_failures.txt'.format(outdir)

# Properties of the GW data
sqrtSX = 2e-23
tstart = 1000000000
Tspan = 100*86400
tend = tstart + Tspan

# Fixed properties of the signal
F0_center = 30
F1_center = 1e-10
F2 = 0
tref = .5*(tstart+tend)


VF0 = VF1 = 100
DeltaF0 = VF0 * np.sqrt(3)/(np.pi*Tspan)
DeltaF1 = VF1 * np.sqrt(45/4.)/(np.pi*Tspan**2)

DeltaAlpha = 0.02
DeltaDelta = 0.02

depths = [140]

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
    Alpha_center = np.random.uniform(0, 2*np.pi)
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
    predicted_twoF = data.predict_fstat()

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
    theta_prior = {'F0': {'upper': 30.000006381121477, 'lower': 29.999993618878523, 'type': 'unif'}, 'F1': {'upper': 1.0143020701400378e-10, 'lower': 9.8569792985996225e-11, 'type': 'unif'}, 'F2': 0, 'Delta': {'upper': -0.20155527961896461, 'lower': -0.24155527961896459, 'type': 'unif'}, 'Alpha': {'upper': 2.8924321897264367, 'lower': 2.8524321897264366, 'type': 'unif'}}

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
    mcmc.run(run_setup=run_setup, create_plots=True, log_table=False,
             gen_tex_table=False)
    d, maxtwoF = mcmc.get_max_twoF()
    print 'MaxtwoF = {}'.format(maxtwoF)
