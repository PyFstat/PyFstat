# Making fake data

Here, we describe the steps required to generate fake data which will be used
throughout the other examples. We will generate data based on the properties of
the Crab pulsar, first as a smooth CW signal and then as a CW signal which
contains a glitch. This document is based on the file
[make_fake_data.py](../examples/make_fake_data.py).

## Smooth signal

In the following code segment, we import the `Writer` class used to generate
fake data, define the Crab parameters and create an instant of the `Writer`
called `data`

```
from pyfstat import Writer

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
```

We can now use the `data` object to create `.sft` files which contain a smooth
signal in Gaussian noise. In detail, the process consists first in calling

```
data.make_cff()
```
which generates a file `data/basic.cff` (notice the directory and file name
are defined by the `outdir` and `label` arguments given to `Writer`). This
file contains instructions which will be passed to `lalapps_MakeFakedata_v5`,
namely

```
[TS0]
Alpha = 5.000000000000000104e-03
Delta = 5.999999999999999778e-02
h0 = 9.999999999999999604e-24
cosi = 0.000000000000000000e+00
psi = 0.000000000000000000e+00
phi0 = 0.000000000000000000e+00
Freq = 3.000000000000000000e+01
f1dot = -1.000000000000000036e-10
f2dot = 0.000000000000000000e+00
refTime = 362750407.000000
transientWindowType=rect
transientStartTime=1000000000.000
transientTauDays=100.000
```

Finally, we generate the `.sft` files by calling

```
data.run_makefakedata()
```

In fact, the previous two commands are wrapped together by a single call to
`data.make_data()` which we will use from now on.

## Glitching signal

We now want to generate a set of data which contains a *glitching signal*. We
start with a simple case in which the glitch occurs half way through the
observation span. We define the properties of this signal, create
another `Writer` instance called `glitch_data`, and then run `make_data()`

```
dtglitch = duration/2.0
delta_F0 = 1e-6 * F0
delta_F1 = 1e-5 * F1

glitch_data = Writer(
    label='glitch', outdir='data', tref=tref, tstart=tstart, F0=F0, F1=F1,
    F2=F2, duration=duration, Alpha=Alpha, Delta=Delta, h0=h0, sqrtSX=sqrtSX,
    dtglitch=dtglitch, delta_F0=delta_F0, delta_F1=delta_F1)
glitch_data.make_data()
```

It is worth noting the difference between the config file for the non-glitching
signal and this config file (`data/glitch.cff`) which reads

```
[TS0]
Alpha = 5.000000000000000104e-03
Delta = 5.999999999999999778e-02
h0 = 9.999999999999999604e-24
cosi = 0.000000000000000000e+00
psi = 0.000000000000000000e+00
phi0 = 0.000000000000000000e+00
Freq = 3.000000000000000000e+01
f1dot = -1.000000000000000036e-10
f2dot = 0.000000000000000000e+00
refTime = 362750407.000000
transientWindowType=rect
transientStartTime=1000000000.000
transientTauDays=50.000
[TS1]
Alpha = 5.000000000000000104e-03
Delta = 5.999999999999999778e-02
h0 = 9.999999999999999604e-24
cosi = 0.000000000000000000e+00
psi = 0.000000000000000000e+00
phi0 = -1.222261350197196007e+05
Freq = 3.000003064156959098e+01
f1dot = -1.000009999999999993e-10
f2dot = 0.000000000000000000e+00
refTime = 362750407.000000
transientWindowType=rect
transientStartTime=1004320000.000
transientTauDays=50.000
```

The glitch config file uses transient windows to create two non-overlapping,
but continuous signals.


