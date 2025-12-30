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

signal = True  # turn off for pure noise spectrogram
binary = True  # turn off for simpler isolated neutron star signal

# this sets how strong the signal is compared to the noise (higher value -> weaker)
depth = 5

# Define the tstart and duration of each segment of data,
# including some gaps like in real observing runs.
gap_duration = 10 * 86400
Tsft = 1800
segments = [
    {"tstart": 1000000000, "duration": 120 * 86400},
    {"tstart": 1000000000 + 120 * 86400 + gap_duration, "duration": 300 * 86400},
    {"tstart": 1000000000 + 420 * 86400 + 2 * gap_duration, "duration": 120 * 86400},
]

# Generate timestamps for each segment and concatenate them.
timestamps = {
    "H1": np.concatenate(
        [
            segment["tstart"] + Tsft * np.arange(segment["duration"] // Tsft)
            for segment in segments
        ]
    )
}

# general parameters to configure the data to simulate
data_parameters = {
    "sqrtSX": 1e-23,
    "timestamps": timestamps,
    "detectors": "H1",
    "Tsft": 1800,
}
# For pure noise, we need to select the frequency range ourselves.
# For signals, PyFstat auto-estimates this.
if not signal:
    data_parameters.update(
        {
            "F0": 100.0,
            "Band": 0.1,
        }
    )

# parameters for a signal to inject (if signal==True)
signal_parameters = {
    "F0": 100.0,
    "F1": 0,
    "F2": 0,
    "Alpha": 0.0,
    "Delta": 0.5,
    "tref": segments[0]["tstart"],
    "h0": data_parameters["sqrtSX"] / depth if signal else 0.0,
    "cosi": 1.0,
}
# optionally add binary orbital parameters
if binary:
    signal_parameters.update(
        {
            "tp": segments[0]["tstart"],
            "asini": 25.0,
            "period": 50 * 86400,
        }
    )

# making data
data = pyfstat.Writer(
    label=label,
    outdir=outdir,
    **data_parameters,
    signal_parameters=signal_parameters,
)
data.make_data()

# make the plot
ax = pyfstat.utils.plot_spectrogram(
    sftfilepattern=data.sftfilepath,
    detector=data_parameters["detectors"],
    sqrtSX=data_parameters["sqrtSX"],
    quantity="normpower",
    savefig=True,
    outdir=outdir,
    label=label,
)
