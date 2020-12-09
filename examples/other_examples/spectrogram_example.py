import os
import shutil
import sys

import matplotlib.pyplot as plt

import pyfstat

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 18
plt.rcParams["text.usetex"] = True

label = "track_example"
outdir = os.path.join(sys.path[0], "tmp")

depth = 1

data_parameters = {
    "sqrtSX": 1e-23,
    "tstart": 1000000000,
    "duration": 2 * 365 * 86400,
    "detectors": "H1",
    "Tsft": 1800,
}

signal_parameters = {
    "F0": 100.0,
    "F1": 0,
    "F2": 0,
    "Alpha": 0.0,
    "Delta": 0.5,
    "tp": data_parameters["tstart"],
    "asini": 25.0,
    "period": 50 * 86400,
    "tref": data_parameters["tstart"],
    "h0": data_parameters["sqrtSX"] / depth,
    "cosi": 1.0,
}

data = pyfstat.BinaryModulatedWriter(
    label=label, outdir=outdir, **data_parameters, **signal_parameters
)
data.make_data()

times, freqs, sft_data = pyfstat.helper_functions.get_sft_array(data.sftfilepath)

fig, ax = plt.subplots(figsize=(0.8 * 16, 0.8 * 9))
ax.grid(which="both")
ax.set(xlabel="Time [days]", ylabel="Frequency [Hz]", ylim=(99.98, 100.02))
c = ax.pcolormesh(
    (times - times[0]) / 86400,
    freqs,
    (sft_data / data_parameters["sqrtSX"])[:-1, :-1],
    cmap="inferno_r",
)
fig.colorbar(c, label="Normalized Power")
plt.tight_layout()
fig.savefig("spectrogram_example.png")

shutil.rmtree(outdir)
