"""
Compute a spectrogram
==========================

Compute the spectrogram of a set of SFTs. This is useful to produce
visualizations of the Doppler modulation of a CW signal.
"""

import os
import matplotlib.pyplot as plt

import pyfstat

# not github-action compatible
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.size"] = 18
# plt.rcParams["text.usetex"] = True

label = "PyFstat_example_spectrogram"
outdir = os.path.join("PyFstat_example_data", label)

depth = 5

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

# making data
data = pyfstat.BinaryModulatedWriter(
    label=label, outdir=outdir, **data_parameters, **signal_parameters
)
data.make_data()

print("Loading SFT data and computing normalized power...")
times, freqs, sft_data = pyfstat.helper_functions.get_sft_array(data.sftfilepath)
normalized_power = (
    2 * sft_data ** 2 / (data_parameters["Tsft"] * data_parameters["sqrtSX"] ** 2)
)

plotfile = os.path.join(outdir, label + ".png")
print(f"Plotting to file: {plotfile}")
fig, ax = plt.subplots(figsize=(0.8 * 16, 0.8 * 9))
ax.grid(which="both")
ax.set(xlabel="Time [days]", ylabel="Frequency [Hz]", ylim=(99.98, 100.02))
c = ax.pcolormesh(
    (times - times[0]) / 86400,
    freqs,
    normalized_power,
    cmap="inferno_r",
    shading="nearest",
)
fig.colorbar(c, label="Normalized Power")
plt.tight_layout()
fig.savefig(plotfile)
