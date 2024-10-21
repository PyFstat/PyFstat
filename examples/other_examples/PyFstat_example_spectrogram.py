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

label = "PyFstatExampleSpectrogramImag"
outdir = os.path.join("PyFstat_example_data", label)
logger = pyfstat.set_up_logger(label=label, outdir=outdir)

depth = 5

# timestamps = {"H1": 1238166018 + 1800 * np.array([0, 2, 4, 8])}
timestamps = {
    "H1": np.array(
        [
            1000000000,
            1000000000 + 1800,
            1000000000 + 3600,
            1000000000 + 10000,
            1000000000 + 11800,
        ]
    )
}

data_parameters = {
    "sqrtSX": 1e-23,
    # "tstart": 1000000000,
    # "duration": 2 * 365 * 86400,
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
    # "tp": data_parameters["tstart"],
    # "asini": 25.0,
    # "period": 50 * 86400,
    # "tref": data_parameters["tstart"],
    "tref": 1000000000,
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
    quantity="imag",
    savefig=True,
    outdir=outdir,
    label=label,
)
