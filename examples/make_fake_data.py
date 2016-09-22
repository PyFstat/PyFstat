from pyfstat import Writer

# First, we generate data with a reasonably strong smooth signal

# Define parameters of the Crab pulsar
F0 = 30.0
F1 = -1e-10
F2 = 0
Alpha = 5e-3
Delta = 6e-2
tref = 362750407.0

# Signal strength
h0 = 1e-23

# Properties of the GW data
sqrtSX = 1e-22
tstart = 1000000000
duration = 100*86400
tend = tstart+duration

data = Writer(
    label='basic', outdir='data', tref=tref, tstart=tstart, F0=F0, F1=F1,
    F2=F2, duration=duration, Alpha=Alpha, Delta=Delta, h0=h0, sqrtSX=sqrtSX)
data.make_data()


# Next, taking the same signal parameters, we include a glitch half way through
dtglitch = duration/2.0
delta_F0 = 1e-6 * F0
delta_F1 = 1e-5 * F1

glitch_data = Writer(
    label='glitch', outdir='data', tref=tref, tstart=tstart, F0=F0, F1=F1,
    F2=F2, duration=duration, Alpha=Alpha, Delta=Delta, h0=h0, sqrtSX=sqrtSX,
    dtglitch=dtglitch, delta_F0=delta_F0, delta_F1=delta_F1)
glitch_data.make_data()

