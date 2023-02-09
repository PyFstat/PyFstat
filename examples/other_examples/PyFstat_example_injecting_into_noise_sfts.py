"""
Software injection into pre-existing data files
===============================================

Add a software injection into a set of SFTs.

In this case, the set of SFTs is generated using Makefakedata_v5,
but the same procedure can be applied to any other set of SFTs
(including real detector data).
"""

import os

import numpy as np

import pyfstat

label = "PyFstatExampleInjectionIntoNoiseSFTs"
outdir = os.path.join("PyFstat_example_data", label)
logger = pyfstat.set_up_logger(label=label, outdir=outdir)

tstart = 1269734418
duration_Tsft = 100
Tsft = 1800
randSeed = 69420
IFO = "H1"
h0 = 1000
cosi = 0
F0 = 30
Alpha = 0
Delta = 0

Band = 2.0

# create sfts with a strong signal in them
# window options are optional here
noise_and_signal_writer = pyfstat.Writer(
    label="PyFstatTestSFTsNoiseAndSignal",
    outdir=outdir,
    h0=h0,
    cosi=cosi,
    F0=F0,
    Alpha=Alpha,
    Delta=Delta,
    tstart=tstart,
    duration=duration_Tsft * Tsft,
    Tsft=Tsft,
    Band=Band,
    detectors=IFO,
    randSeed=randSeed,
    SFTWindowType="tukey",
    SFTWindowParam=0.001,
)
noise_and_signal_writer.make_data()

# compute Fstat
coherent_search = pyfstat.ComputeFstat(
    tref=noise_and_signal_writer.tref,
    sftfilepattern=noise_and_signal_writer.sftfilepath,
    minCoverFreq=-0.5,
    maxCoverFreq=-0.5,
)
FS_1 = coherent_search.get_fullycoherent_twoF(
    noise_and_signal_writer.F0,
    noise_and_signal_writer.F1,
    noise_and_signal_writer.F2,
    noise_and_signal_writer.Alpha,
    noise_and_signal_writer.Delta,
)

# create noise sfts
# window options are again optional for this step
noise_writer = pyfstat.Writer(
    label="PyFstatTestSFTsOnlyNoise",
    outdir=outdir,
    h0=0,
    F0=F0,
    tstart=tstart,
    duration=duration_Tsft * Tsft,
    Tsft=Tsft,
    Band=Band,
    detectors=IFO,
    randSeed=randSeed,
    SFTWindowType="tukey",
    SFTWindowParam=0.001,
)
noise_writer.make_data()

# then inject a strong signal
# window options *must* match those previously used for the noiseSFTs
add_signal_writer = pyfstat.Writer(
    label="PyFstatTestSFTsWithAddedSignal",
    outdir=outdir,
    F0=F0,
    Alpha=Alpha,
    Delta=Delta,
    h0=h0,
    cosi=cosi,
    tstart=tstart,
    duration=duration_Tsft * Tsft,
    Tsft=Tsft,
    Band=Band,
    detectors=IFO,
    sqrtSX=0,
    noiseSFTs=noise_writer.sftfilepath,
    SFTWindowType="tukey",
    SFTWindowParam=0.001,
)
add_signal_writer.make_data()

# compute Fstat
coherent_search = pyfstat.ComputeFstat(
    tref=add_signal_writer.tref,
    sftfilepattern=add_signal_writer.sftfilepath,
    minCoverFreq=-0.5,
    maxCoverFreq=-0.5,
)
FS_2 = coherent_search.get_fullycoherent_twoF(
    add_signal_writer.F0,
    add_signal_writer.F1,
    add_signal_writer.F2,
    add_signal_writer.Alpha,
    add_signal_writer.Delta,
)

logger.info("Base case Fstat: {}".format(FS_1))
logger.info("Noise + Signal Fstat: {}".format(FS_2))
logger.info("Relative Difference: {}".format(np.abs(FS_2 - FS_1) / FS_1))
