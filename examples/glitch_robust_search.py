import numpy as np
import pyfstat

outdir = 'data'
label = 'glitch_robust_search'

# Properties of the GW data
tstart = 1000000000
Tspan = 60 * 86400

# Fixed properties of the signal
F0s = 30
F1s = -1e-8
F2s = 0
Alpha = np.radians(83.6292)
Delta = np.radians(22.0144)

tref = tstart + .5 * Tspan

sftfilepath = 'data/*glitching_signal*sft'

F0_width = np.sqrt(3)/(np.pi*Tspan)
F1_width = np.sqrt(45/4.)/(np.pi*Tspan**2)
DeltaF0 = 50 * F0_width
DeltaF1 = 50 * F1_width

theta_prior = {'F0': {'type': 'unif',
                      'lower': F0s-DeltaF0,
                      'upper': F0s+DeltaF0},
               'F1': {'type': 'unif',
                      'lower': F1s-DeltaF1,
                      'upper': F1s+DeltaF1},
               'F2': F2s,
               'delta_F0': {'type': 'unif',
                            'lower': 0,
                            'upper': 1e-5},
               'delta_F1': {'type': 'unif',
                            'lower': -1e-11,
                            'upper': 1e-11},
               'tglitch': {'type': 'unif',
                           'lower': tstart+0.1*Tspan,
                           'upper': tstart+0.9*Tspan},
               'Alpha': Alpha,
               'Delta': Delta,
               }

search = pyfstat.MCMCGlitchSearch(
    label=label, outdir=outdir, sftfilepath=sftfilepath,
    theta_prior=theta_prior, nglitch=1, tref=tref, nsteps=[500, 500],
    ntemps=3, log10temperature_min=-0.5, minStartTime=tstart,
    maxStartTime=tstart+Tspan)
search.run()
search.plot_corner(label_offset=0.8, add_prior=True)
search.print_summary()
