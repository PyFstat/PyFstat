import numpy as np
import pyfstat

outdir = 'data'
label = 'simulated_glitching_signal'

# Properties of the GW data
tstart = 1000000000
Tspan = 60 * 86400

tref = tstart + .5 * Tspan

# Fixed properties of the signal
F0s = 30
F1s = -1e-8
F2s = 0
Alpha = np.radians(83.6292)
Delta = np.radians(22.0144)
h0 = 1e-25
sqrtSX = 1e-24
psi = -0.1
phi = 0
cosi = 0.5

# Glitch properties
dtglitch = 0.45 * Tspan  # time (in secs) after minStartTime
dF0 = 5e-6
dF1 = 1e-12


detectors = 'H1'

glitch_data = pyfstat.Writer(
    label=label, outdir=outdir, tref=tref, tstart=tstart,
    F0=F0s, F1=F1s, F2=F2s, duration=Tspan, Alpha=Alpha,
    Delta=Delta, sqrtSX=sqrtSX, dtglitch=dtglitch,
    h0=h0, cosi=cosi, phi=phi, psi=psi,
    delta_F0=dF0, delta_F1=dF1, add_noise=True)

glitch_data.make_data()
