from pyfstat import Writer

# First, we generate data with a reasonably strong smooth signal

# Define parameters of the Crab pulsar as an example
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

# The predicted twoF, given by lalapps_predictFstat can be accessed by
twoF = data.predict_fstat()
print 'Predicted twoF value: {}\n'.format(twoF)

# Next, taking the same signal parameters, we include a glitch half way through
dtglitch = duration/2.0
delta_F0 = 4e-5
delta_F1 = 0

glitch_data = Writer(
    label='glitch', outdir='data', tref=tref, tstart=tstart, F0=F0, F1=F1,
    F2=F2, duration=duration, Alpha=Alpha, Delta=Delta, h0=h0, sqrtSX=sqrtSX,
    dtglitch=dtglitch, delta_F0=delta_F0, delta_F1=delta_F1)
glitch_data.make_data()

# Making data with two glitches

dtglitch = [duration/4.0, 4*duration/5.0]
delta_phi = [0, 0]
delta_F0 = [4e-6, 3e-7]
delta_F1 = [0, 0]
delta_F2 = [0, 0]

two_glitch_data = Writer(
    label='twoglitch', outdir='data', tref=tref, tstart=tstart, F0=F0, F1=F1,
    F2=F2, duration=duration, Alpha=Alpha, Delta=Delta, h0=h0, sqrtSX=sqrtSX,
    dtglitch=dtglitch, delta_phi=delta_phi, delta_F0=delta_F0,
    delta_F1=delta_F1, delta_F2=delta_F2)
two_glitch_data.make_data()


# Making transient data in the middle third
data_tstart = tstart - duration
data_duration = 3 * duration
transient = Writer(
    label='transient', outdir='data', tref=tref, tstart=tstart, F0=F0, F1=F1,
    F2=F2, duration=duration, Alpha=Alpha, Delta=Delta, h0=h0, sqrtSX=sqrtSX,
    data_tstart=data_tstart, data_duration=data_duration)
transient.make_data()

