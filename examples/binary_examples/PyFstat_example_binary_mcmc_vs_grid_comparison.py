#!/usr/bin/env python

import pyfstat
import os
import numpy as np
import matplotlib.pyplot as plt

# Set to false to include eccentricity
circular_orbit = False

label = os.path.splitext(os.path.basename(__file__))[0]
outdir = os.path.join("PyFstat_example_data", label)

# Parameters to generate a data set
data_parameters = {
    "sqrtSX": 1e-22,
    "tstart": 1000000000,
    "duration": 60 * 86400,
    "detectors": "H1,L1",
    "Tsft": 1800,
    "Band": 4,
}

# Injected signal parameters
tend = data_parameters["tstart"] + data_parameters["duration"]
mid_time = 0.5 * (data_parameters["tstart"] + tend)
depth = 10.0
signal_parameters = {
    "tref": data_parameters["tstart"],
    "F0": 100.0,
    "F1": 0,
    "F2": 0,
    "Alpha": 0.5,
    "Delta": 0.5,
    "period": 5 * 24 * 3600.0,
    "asini": 10.0,
    "tp": mid_time * 1.05,
    "argp": 0.0 if circular_orbit else 0.54,
    "ecc": 0.0 if circular_orbit else 0.7,
    "h0": data_parameters["sqrtSX"] / depth,
    "cosi": 1.0,
}


print("Generating SFTs with injected signal...")
writer = pyfstat.BinaryModulatedWriter(
    label="simulated_signal",
    outdir=outdir,
    **data_parameters,
    **signal_parameters,
)
writer.make_data()
print("")

print("Performing Grid Search...")

# Create ad-hoc grid and compute Fstatistic around injection point
half_points_per_dimension = 2
search_keys = ["period", "asini", "tp"] + ["argp", "ecc"]
search_keys_label = [
    r"$P$ [s]",
    r"$a_p$ [s]",
    r"$t_{p}$ [s]",
    r"$\omega$ [rad]",
    r"$e$",
]
grid_arrays = np.meshgrid(
    *[
        signal_parameters[key]
        * (
            1
            + 0.01
            * np.arange(-half_points_per_dimension, half_points_per_dimension + 1, 1)
        )
        for key in search_keys
    ]
)
grid_points = np.hstack(
    [grid_arrays[i].reshape(-1, 1) for i in range(len(grid_arrays))]
)

compute_f_stat = pyfstat.ComputeFstat(
    sftfilepattern=os.path.join(outdir, "*simulated_signal*sft"),
    tref=signal_parameters["tref"],
    binary=True,
    minCoverFreq=-0.5,
    maxCoverFreq=-0.5,
)
twoF_values = np.zeros(grid_points.shape[0])
for ind in range(grid_points.shape[0]):
    point = grid_points[ind]
    twoF_values[ind] = compute_f_stat.get_fullycoherent_twoF(
        tstart=data_parameters["tstart"],
        tend=tend,
        F0=signal_parameters["F0"],
        F1=signal_parameters["F1"],
        F2=signal_parameters["F2"],
        Alpha=signal_parameters["Alpha"],
        Delta=signal_parameters["Delta"],
        period=point[0],
        asini=point[1],
        tp=point[2],
        argp=point[3],
        ecc=point[4],
    )
print(f"2Fstat computed on {grid_points.shape[0]} points")
print("")
print("Plotting results...")
dim = len(search_keys)
fig, ax = plt.subplots(dim, 1, figsize=(10, 10))
for ind in range(dim):
    a = ax.ravel()[ind]
    a.grid()
    a.set(xlabel=search_keys_label[ind], ylabel=r"$2 \mathcal{F}$", yscale="log")
    a.plot(grid_points[:, ind], twoF_values, "o")
    a.axvline(signal_parameters[search_keys[ind]], label="Injection", color="orange")
plt.tight_layout()
fig.savefig(os.path.join(outdir, "grid_twoF_per_dimension.pdf"))


print("Performing MCMCSearch...")

# Set up priors for the binary parameters
theta_prior = {
    "F0": signal_parameters["F0"],
    "F1": signal_parameters["F1"],
    "F2": signal_parameters["F2"],
    "Alpha": signal_parameters["Alpha"],
    "Delta": signal_parameters["Delta"],
}

for key in search_keys:
    theta_prior.update(
        {
            key: {
                "type": "unif",
                "lower": 0.999 * signal_parameters[key],
                "upper": 1.001 * signal_parameters[key],
            }
        }
    )
if circular_orbit:
    for key in ["ecc", "argp"]:
        theta_prior[key] = 0
        search_keys.remove(key)
# theta_prior["tp"]["lower"] = signal_parameters["tp"] - 0.5 * signal_parameters["period"]
# theta_prior["tp"]["upper"] = (
#    signal_parameters["tp"]
#    + 0.5 * (1 + signal_parameters["argp"] / np.pi) * signal_parameters["period"]
# )

# ptemcee sampler settings - in a real application we might want higher values
ntemps = 3
log10beta_min = -1
nwalkers = 100
nsteps = [200, 200]  # [burnin,production]

mcmcsearch = pyfstat.MCMCSearch(
    label="mcmc_search",
    outdir=outdir,
    sftfilepattern=os.path.join(outdir, "*simulated_signal*sft"),
    theta_prior=theta_prior,
    tref=signal_parameters["tref"],
    nsteps=nsteps,
    nwalkers=nwalkers,
    ntemps=ntemps,
    log10beta_min=log10beta_min,
    binary=True,
)
# walker plot is generated during main run of the search class
mcmcsearch.run(
    plot_walkers=True,
    walker_plot_args={"plot_det_stat": True, "injection_parameters": signal_parameters},
)
mcmcsearch.print_summary()

# call some built-in plotting methods
# these can all highlight the injection parameters, too
print("Making MCMCSearch {:s} corner plot...".format("-".join(search_keys)))
mcmcsearch.plot_corner(truths=signal_parameters)
print("Making MCMCSearch prior-posterior comparison plot...")
mcmcsearch.plot_prior_posterior(injection_parameters=signal_parameters)
print("")

print("*" * 20)
print("Quantitative comparisons:")
print("*" * 20)

# some informative command-line output comparing search results and injection
# get max twoF and binary Doppler parameters
max_grid_index = np.argmax(twoF_values)
max_grid_2F = twoF_values[max_grid_index]
max_grid_parameters = grid_points[max_grid_index]

# same for MCMCSearch, here twoF is separate, and non-sampled parameters are not included either
max_dict_mcmc, max_2F_mcmc = mcmcsearch.get_max_twoF()
print(
    "Grid Search:\n\tmax2F={:.4f}\n\tOffsets from injection parameters (relative error): {:s}.".format(
        max_grid_2F,
        ", ".join(
            [
                "\n\t\t{1:s}: {0:.4e} ({2:.4g})%".format(
                    max_grid_parameters[search_keys.index(key)]
                    - signal_parameters[key],
                    key,
                    100
                    * (
                        max_grid_parameters[search_keys.index(key)]
                        - signal_parameters[key]
                    )
                    / signal_parameters[key],
                )
                for key in search_keys
            ]
        ),
    )
)
print(
    "Max 2F candidate from MCMC Search:\n\tmax2F={:.4f}"
    "\n\tOffsets from injection parameters (relative error): {:s}.".format(
        max_2F_mcmc,
        ", ".join(
            [
                "\n\t\t{1:s}: {0:.4e} ({2:.4g})%".format(
                    max_dict_mcmc[key] - signal_parameters[key],
                    key,
                    100
                    * (max_dict_mcmc[key] - signal_parameters[key])
                    / signal_parameters[key],
                )
                for key in search_keys
            ]
        ),
    )
)
# get additional point and interval estimators
stats_dict_mcmc = mcmcsearch.get_summary_stats()
print(
    "Mean from MCMCSearch:\n\tOffset from injection parameters (relative error): {:s}"
    "\n\tExpressed as fractions of 2sigma intervals: {:s}.".format(
        ", ".join(
            [
                "\n\t\t{1:s}: {0:.4e} ({2:.4g})%".format(
                    stats_dict_mcmc[key]["mean"] - signal_parameters[key],
                    key,
                    100
                    * (stats_dict_mcmc[key]["mean"] - signal_parameters[key])
                    / signal_parameters[key],
                )
                for key in search_keys
            ]
        ),
        ", ".join(
            [
                "\n\t\t{1:s}: {0:.4e}".format(
                    100
                    * np.abs(stats_dict_mcmc[key]["mean"] - signal_parameters[key])
                    / (2 * stats_dict_mcmc[key]["std"]),
                    key,
                )
                for key in search_keys
            ]
        ),
    )
)
print(
    "Median from MCMCSearch:\n\tOffset from injection parameters (relative error): {:s},"
    "\n\tExpressed as fractions of 90% confidence intervals: {:s}.".format(
        ", ".join(
            [
                "\n\t\t{1:s}: {0:.4e} ({2:.4g})%".format(
                    stats_dict_mcmc[key]["median"] - signal_parameters[key],
                    key,
                    100
                    * (stats_dict_mcmc[key]["median"] - signal_parameters[key])
                    / signal_parameters[key],
                )
                for key in search_keys
            ]
        ),
        ", ".join(
            [
                "\n\t\t{1:s}: {0:.4e}".format(
                    100
                    * np.abs(stats_dict_mcmc[key]["median"] - signal_parameters[key])
                    / (
                        stats_dict_mcmc[key]["upper90"]
                        - stats_dict_mcmc[key]["lower90"]
                    ),
                    key,
                )
                for key in search_keys
            ]
        ),
    )
)