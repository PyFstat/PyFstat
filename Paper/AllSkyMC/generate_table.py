import pyfstat
import numpy as np

outdir = 'data'

label = 'AllSky'
data_label = '{}_data'.format(label)

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

depths = np.linspace(100, 400, 7)

run_setup = [((100, 0), 27, False),
             ((100, 0), 15, False),
             ((100, 0), 8, False),
             ((100, 0), 4, False),
             ((50, 50), 1, False)]

DeltaAlpha = 0.05
DeltaDelta = 0.05

depth = 100

h0 = sqrtSX / float(depth)
F0 = F0_center
F1 = F1_center
Alpha = 0
Delta = 0
Alpha_min = Alpha - DeltaAlpha/2
Alpha_max = Alpha + DeltaAlpha/2
Delta_min = Delta - DeltaDelta/2
Delta_max = Delta + DeltaDelta/2

psi = 0
phi = 0
cosi = 0

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
               'Alpha': {'type': 'unif',
                         'lower': Alpha_min,
                         'upper': Alpha_max},
               'Delta': {'type': 'unif',
                         'lower': Delta_min,
                         'upper': Delta_max},
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
mcmc.run(run_setup=run_setup)
