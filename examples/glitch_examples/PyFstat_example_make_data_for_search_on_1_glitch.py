"""
Glitch examples: Make data
==========================

Generate the data to run examples on glitch-robust searches.
"""

import os

import numpy as np

from pyfstat import GlitchWriter, Writer

label = "PyFstatExampleGlitchRobustSearch"
outdir = os.path.join("PyFstat_example_data", label)

# First, we generate data with a reasonably strong smooth signal

# Define parameters of the Crab pulsar as an example
F0 = 30.0
F1 = -1e-10
F2 = 0
Alpha = np.radians(83.6292)
Delta = np.radians(22.0144)

# Signal strength
h0 = 5e-24
cosi = 0

# Properties of the GW data
sqrtSX = 1e-22
tstart = 1000000000
duration = 50 * 86400
tend = tstart + duration
tref = tstart + 0.5 * duration
IFO = "H1"

data = Writer(
    label=label + "0glitch",
    outdir=outdir,
    tref=tref,
    tstart=tstart,
    F0=F0,
    F1=F1,
    F2=F2,
    duration=duration,
    Alpha=Alpha,
    Delta=Delta,
    h0=h0,
    cosi=cosi,
    sqrtSX=sqrtSX,
    detectors=IFO,
)
data.make_data()

# Next, taking the same signal parameters, we include a glitch half way through
dtglitch = duration / 2.0
delta_F0 = 5e-6
delta_F1 = 0

glitch_data = GlitchWriter(
    label=label + "1glitch",
    outdir=outdir,
    tref=tref,
    tstart=tstart,
    F0=F0,
    F1=F1,
    F2=F2,
    duration=duration,
    Alpha=Alpha,
    Delta=Delta,
    h0=h0,
    cosi=cosi,
    sqrtSX=sqrtSX,
    detectors=IFO,
    dtglitch=dtglitch,
    delta_F0=delta_F0,
    delta_F1=delta_F1,
)
glitch_data.make_data()

# Making data with two glitches

dtglitch_2 = [duration / 4.0, 4 * duration / 5.0]
delta_phi_2 = [0, 0]
delta_F0_2 = [4e-6, 3e-7]
delta_F1_2 = [0, 0]
delta_F2_2 = [0, 0]

two_glitch_data = GlitchWriter(
    label=label + "2glitch",
    outdir=outdir,
    tref=tref,
    tstart=tstart,
    F0=F0,
    F1=F1,
    F2=F2,
    duration=duration,
    Alpha=Alpha,
    Delta=Delta,
    h0=h0,
    cosi=cosi,
    sqrtSX=sqrtSX,
    detectors=IFO,
    dtglitch=dtglitch_2,
    delta_phi=delta_phi_2,
    delta_F0=delta_F0_2,
    delta_F1=delta_F1_2,
    delta_F2=delta_F2_2,
)
two_glitch_data.make_data()
