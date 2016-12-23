# Making fake data

Here, we describe the steps required to generate fake data which will be used
throughout the other examples. We will generate data based on the properties of
the Crab pulsar, first as a smooth CW signal, then as a CW signal which
contains one glitch, and finally as a signal with two glitches. This document
is based on the file [make_fake_data.py](../examples/make_fake_data.py).

## Smooth signal

In the following code segment, we import the `Writer` class used to generate
fake data, define the Crab parameters and create an instant of the `Writer`
called `data`

```python
import numpy as np
from pyfstat import Writer

# Define parameters of the Crab pulsar
F0 = 30.0
F1 = -1e-10
F2 = 0
Alpha = np.radians(83.6292)
Delta = np.radians(22.0144)
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

```python
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

```python
data.run_makefakedata()
```

In fact, the previous two commands are wrapped together by a single call to
`data.make_data()` which we will use from now on.


## Glitching signal

We now want to generate a set of data which contains a *glitching signal*. We
start with a simple case in which the glitch occurs half way through the
observation span. We define the properties of this signal, create
another `Writer` instance called `glitch_data`, and then run `make_data()`

```python
dtglitch = duration/2.0
delta_F0 = 0.4e-5
delta_F1 = 0

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
phi0 = -1.612440256772935390e+04
Freq = 3.000000400000000056e+01
f1dot = -1.000000000000000036e-10
f2dot = 0.000000000000000000e+00
refTime = 362750407.000000
transientWindowType=rect
transientStartTime=1004320000.000
transientTauDays=50.000
```

The glitch config file uses transient windows to create two non-overlapping,
but continuous signals.

## Expected twoF

Finally, the `Writer` class also provides a wrapper of `lalapps_PredictFstat`.
So calling

```python
>>> print data.predict_fstat()
1721.1
```

Notice that the predicted value will be the same for both sets of data.

## Making data with multiple glitches

Finally, one can also use the `Writer` to generate data with multiple glitches.
To do this, simply pass in `dtglitch`, `delta_phi`, `delta_F0`, `delta_F1`, and
`delta_F2` as arrays  (with a length equal to the number of glitches). Note
that all these must be of equal length. Moreover, the glitches are applied
sequentially and additively as implemented
`pyfstat.BaseSearchClass.calculate_thetas`. Here is an example with two
glitches, one a quarter of the way through and the other a fifth from the end.

```python
dtglitch = [duration/4.0, 4*duration/5.0]
delta_phi = [0, 0]
delta_F0 = [0.4e-5, 0.3e-6]
delta_F1 = [0, 0]
delta_F2 = [0, 0]

two_glitch_data = Writer(
    label='two_glitch', outdir='data', tref=tref, tstart=tstart, F0=F0, F1=F1,
    F2=F2, duration=duration, Alpha=Alpha, Delta=Delta, h0=h0, sqrtSX=sqrtSX,
    dtglitch=dtglitch, delta_phi=delta_phi, delta_F0=delta_F0,
    delta_F1=delta_F1, delta_F2=delta_F2)
two_glitch_data.make_data()
```

So, having run `$ python make_fake_data.py` (from the `examples` directory), we
will see that in the sub-directory `examples/data/` there are three `.sft`
files.  These will be used throughout the other examples.

