import os
import sys
import numpy as np
import lalpulsar
import pyfstat

label = os.path.splitext(os.path.basename(__file__))[0]
outdir = os.path.join("PyFstat_example_data", label)

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
    label="test_noiseSFTs_noise_and_signal",
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
    SFTWindowBeta=0.001,
)
sftfilepattern = os.path.join(
    noise_and_signal_writer.outdir,
    "*{}*{}*sft".format(duration_Tsft, noise_and_signal_writer.label),
)

noise_and_signal_writer.make_data()

# compute Fstat
coherent_search = pyfstat.ComputeFstat(
    tref=noise_and_signal_writer.tref,
    sftfilepattern=sftfilepattern,
    minCoverFreq=-0.5,
    maxCoverFreq=-0.5,
)
FS_1 = coherent_search.get_fullycoherent_twoF(
    noise_and_signal_writer.tstart,
    noise_and_signal_writer.tend,
    noise_and_signal_writer.F0,
    noise_and_signal_writer.F1,
    noise_and_signal_writer.F2,
    noise_and_signal_writer.Alpha,
    noise_and_signal_writer.Delta,
)

# create noise sfts
# window options are again optional for this step
noise_writer = pyfstat.Writer(
    label="test_noiseSFTs_only_noise",
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
    SFTWindowBeta=0.001,
)
noise_writer.make_data()

# then inject a strong signal
# window options *must* match those previously used for the noiseSFTs
add_signal_writer = pyfstat.Writer(
    label="test_noiseSFTs_add_signal",
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
    noiseSFTs=os.path.join(
        noise_writer.outdir, "*{}*{}*sft".format(duration_Tsft, noise_writer.label)
    ),
    SFTWindowType="tukey",
    SFTWindowBeta=0.001,
)
sftfilepattern = os.path.join(
    add_signal_writer.outdir,
    "*{}*{}*sft".format(duration_Tsft, add_signal_writer.label),
)
add_signal_writer.make_data()

# compute Fstat
coherent_search = pyfstat.ComputeFstat(
    tref=add_signal_writer.tref,
    sftfilepattern=sftfilepattern,
    minCoverFreq=-0.5,
    maxCoverFreq=-0.5,
)
FS_2 = coherent_search.get_fullycoherent_twoF(
    add_signal_writer.tstart,
    add_signal_writer.tend,
    add_signal_writer.F0,
    add_signal_writer.F1,
    add_signal_writer.F2,
    add_signal_writer.Alpha,
    add_signal_writer.Delta,
)

print("Base case Fstat: {}".format(FS_1))
print("Noise + Signal Fstat: {}".format(FS_2))
print("Relative Difference: {}".format(np.abs(FS_2 - FS_1) / FS_1))
