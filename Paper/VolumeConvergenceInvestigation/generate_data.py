import pyfstat
import numpy as np
import os
import sys
import time

ID = sys.argv[1]
outdir = sys.argv[2]

label = 'VCrun_{}'.format(ID)
data_label = 'VCrunData_{}_'.format(ID)
results_file_name = '{}/VolumeConvergenceResults_{}.txt'.format(outdir, ID)

# Properties of the GW data
sqrtSX = 1e-23
tstart = 1000000000
Tspan = 100*86400
tend = tstart + Tspan

# Fixed properties of the signal
F0 = 30
F1 = -1e-10
F2 = 0
Alpha = np.radians(83.6292)
Delta = np.radians(22.0144)
tref = .5*(tstart+tend)

depth = 100

h0 = sqrtSX / float(depth)

psi = np.random.uniform(-np.pi/4, np.pi/4)
phi = np.random.uniform(0, 2*np.pi)
cosi = np.random.uniform(-1, 1)

data = pyfstat.Writer(
    label=data_label, outdir=outdir, tref=tref,
    tstart=tstart, F0=F0, F1=F1, F2=F2, duration=Tspan, Alpha=Alpha,
    Delta=Delta, h0=h0, sqrtSX=sqrtSX, psi=psi, phi=phi, cosi=cosi,
    detectors='L1')
data.make_data()
twoF_PM = data.predict_fstat()

nsteps = [500, 500]

Vs = np.arange(50, 550, 50)
Vs = [50, 150, 200, 250, 350, 400, 450]

for V in Vs:

    DeltaF0 = V * np.sqrt(3)/(np.pi*Tspan)
    DeltaF1 = V * np.sqrt(45/4.)/(np.pi*Tspan**2)

    startTime = time.time()
    theta_prior = {'F0': {'type': 'unif',
                          'lower': F0-DeltaF0/2.0,
                          'upper': F0+DeltaF0/2.0},
                   'F1': {'type': 'unif',
                          'lower': F1-DeltaF1/2.0,
                          'upper': F1+DeltaF1/2.0},
                   'F2': F2,
                   'Alpha': Alpha,
                   'Delta': Delta
                   }

    ntemps = 3
    log10temperature_min = -0.5
    nwalkers = 100

    mcmc = pyfstat.MCMCSearch(
        label=label, outdir=outdir, nsteps=nsteps,
        sftfilepath='{}/*{}*sft'.format(outdir, data_label),
        theta_prior=theta_prior,
        tref=tref, minStartTime=tstart, maxStartTime=tend,
        nwalkers=nwalkers, ntemps=ntemps,
        log10temperature_min=log10temperature_min)
    mcmc.setup_convergence_testing(
        convergence_period=10, convergence_length=10,
        convergence_burnin_fraction=0.1, convergence_threshold_number=5,
        convergence_threshold=1.1, convergence_early_stopping=False)
    mcmc.run(create_plots=False,
             log_table=False, gen_tex_table=False
             )
    mcmc.print_summary()
    cdF0, cdF1 = mcmc.convergence_diagnostic[-1]
    d, maxtwoF = mcmc.get_max_twoF()
    d_med_std = mcmc.get_median_stds()
    dF0 = F0 - d['F0']
    dF1 = F1 - d['F1']
    F0_std = d_med_std['F0_std']
    F1_std = d_med_std['F1_std']
    runTime = time.time() - startTime
    with open(results_file_name, 'a') as f:
        f.write('{} {:1.8e} {:1.8e} {:1.8e} {:1.8e} {:1.8e} {:1.8e} {:1.8e} {:1.8e} {:1.5e} {:1.5e} {}\n'
                .format(V, dF0, dF1, F0_std, F1_std, DeltaF0, DeltaF1, maxtwoF, twoF_PM, cdF0, cdF1,
                        runTime))
    os.system('rm {}/*{}*'.format(outdir, label))

os.system('rm {}/*{}*'.format(outdir, data_label))
