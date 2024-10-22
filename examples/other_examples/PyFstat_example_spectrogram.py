"""
Compute a spectrogram
==========================

Compute the spectrogram of a set of SFTs. This is useful to produce
visualizations of the Doppler modulation of a CW signal.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import pyfstat

# not github-action compatible
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.size"] = 18
# plt.rcParams["text.usetex"] = True

# workaround deprecation warning
# see https://github.com/matplotlib/matplotlib/issues/21723
plt.rcParams["axes.grid"] = False

label = "PyFstatExampleSpectrogramNormPower"
outdir = os.path.join("PyFstat_example_data", label)
logger = pyfstat.set_up_logger(label=label, outdir=outdir)

depth = 5

gap_duration = 10 * 86400
Tsft = 1800

segments = [  # Define the tstart and duration of each segment of data
    {"tstart": 1000000000, "duration": 120 * 86400},
    {"tstart": 1000000000 + 120 * 86400 + gap_duration, "duration": 300 * 86400},
    {"tstart": 1000000000 + 420 * 86400 + 2 * gap_duration, "duration": 120 * 86400},
]

timestamps = {
    "H1": np.concatenate(
        [  # Generate timestamps for each segment and concatenate them
            segment["tstart"] + Tsft * np.arange(segment["duration"] // Tsft)
            for segment in segments
        ]
    )
}

data_parameters = {
    "sqrtSX": 1e-23,
    "timestamps": timestamps,
    "detectors": "H1",
    "Tsft": 1800,
}

signal_parameters = {
    "F0": 100.0,
    "F1": 0,
    "F2": 0,
    "Alpha": 0.0,
    "Delta": 0.5,
    "tp": segments[0]["tstart"],
    "asini": 25.0,
    "period": 50 * 86400,
    "tref": segments[0]["tstart"],
    "h0": data_parameters["sqrtSX"] / depth,
    "cosi": 1.0,
}

# making data
data = pyfstat.BinaryModulatedWriter(
    label=label, outdir=outdir, **data_parameters, **signal_parameters
)
data.make_data()

ax = pyfstat.utils.plot_spectrogram(
    sftfilepattern=data.sftfilepath,
    detector=data_parameters["detectors"],
    sqrtSX=data_parameters["sqrtSX"],
    quantity="normpower",
    savefig=True,
    outdir=outdir,
    label=label,
)
