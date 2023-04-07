"""
MCMC search with fully coherent BSGL statistic
==============================================

Targeted MCMC search for an isolated CW signal using the
fully coherent line-robust BSGL-statistic.
"""

import os

import numpy as np

import pyfstat

label = "PyFstatExampleFullyCoherentMCMCSearchBSGL"
outdir = os.path.join("PyFstat_example_data", label)
logger = pyfstat.set_up_logger(label=label, outdir=outdir)

# Properties of the GW data - first we make data for two detectors,
# both including Gaussian noise and a coherent 'astrophysical' signal.
data_parameters = {
    "sqrtSX": 1e-23,
    "tstart": 1000000000,
    "duration": 100 * 86400,
    "detectors": "H1,L1",
    "SFTWindowType": "tukey",
    "SFTWindowParam": 0.001,
}
tend = data_parameters["tstart"] + data_parameters["duration"]
mid_time = 0.5 * (data_parameters["tstart"] + tend)

# Properties of the signal
depth = 10
signal_parameters = {
    "F0": 30.0,
    "F1": -1e-10,
    "F2": 0,
    "Alpha": np.radians(83.6292),
    "Delta": np.radians(22.0144),
    "tref": mid_time,
    "h0": data_parameters["sqrtSX"] / depth,
    "cosi": 1.0,
}

data = pyfstat.Writer(
    label=label, outdir=outdir, **data_parameters, **signal_parameters
)
data.make_data()

# The predicted twoF (expectation over noise realizations) can be accessed by
twoF = data.predict_fstat()
logger.info("Predicted twoF value: {}\n".format(twoF))

# Now we add an additional single-detector artifact to H1 only.
# For simplicity, this is modelled here as a fully modulated CW-like signal,
# just restricted to the single detector.
SFTs_H1 = data.sftfilepath.split(";")[0]
data_parameters_line = data_parameters.copy()
signal_parameters_line = signal_parameters.copy()
data_parameters_line["detectors"] = "H1"
data_parameters_line["sqrtSX"] = 0  # don't add yet another set of Gaussian noise
signal_parameters_line["F0"] += 1e-6
signal_parameters_line["h0"] *= 10.0
extra_writer = pyfstat.Writer(
    label=label,
    outdir=outdir,
    **data_parameters_line,
    **signal_parameters_line,
    noiseSFTs=SFTs_H1,
)
extra_writer.make_data()

# use the combined data from both Writers
sftfilepattern = os.path.join(outdir, "*" + label + "*sft")

# MCMC prior ranges
DeltaF0 = 1e-5
DeltaF1 = 1e-13
theta_prior = {
    "F0": {
        "type": "unif",
        "lower": signal_parameters["F0"] - DeltaF0 / 2.0,
        "upper": signal_parameters["F0"] + DeltaF0 / 2.0,
    },
    "F1": {
        "type": "unif",
        "lower": signal_parameters["F1"] - DeltaF1 / 2.0,
        "upper": signal_parameters["F1"] + DeltaF1 / 2.0,
    },
}
for key in "F2", "Alpha", "Delta":
    theta_prior[key] = signal_parameters[key]

# MCMC sampler settings - relatively cheap setup, may not converge perfectly
ntemps = 2
log10beta_min = -0.5
nwalkers = 50
nsteps = [100, 100]

# we'll want to plot results relative to the injection parameters
transform_dict = dict(
    F0=dict(subtractor=signal_parameters["F0"], symbol="$f-f^\\mathrm{s}$"),
    F1=dict(
        subtractor=signal_parameters["F1"], symbol="$\\dot{f}-\\dot{f}^\\mathrm{s}$"
    ),
)

# first search: standard F-statistic
# This should show a weak peak from the coherent signal
# and a larger one from the "line artifact" at higher frequency.
mcmc_F = pyfstat.MCMCSearch(
    label=label + "_twoF",
    outdir=outdir,
    sftfilepattern=sftfilepattern,
    theta_prior=theta_prior,
    tref=mid_time,
    minStartTime=data_parameters["tstart"],
    maxStartTime=tend,
    nsteps=nsteps,
    nwalkers=nwalkers,
    ntemps=ntemps,
    log10beta_min=log10beta_min,
    BSGL=False,
)
mcmc_F.transform_dictionary = transform_dict
mcmc_F.run(
    walker_plot_args={"plot_det_stat": True, "injection_parameters": signal_parameters}
)
mcmc_F.print_summary()
mcmc_F.plot_corner(add_prior=True, truths=signal_parameters)
mcmc_F.plot_prior_posterior(injection_parameters=signal_parameters)

# second search: line-robust statistic BSGL activated
mcmc_F = pyfstat.MCMCSearch(
    label=label + "_BSGL",
    outdir=outdir,
    sftfilepattern=sftfilepattern,
    theta_prior=theta_prior,
    tref=mid_time,
    minStartTime=data_parameters["tstart"],
    maxStartTime=tend,
    nsteps=nsteps,
    nwalkers=nwalkers,
    ntemps=ntemps,
    log10beta_min=log10beta_min,
    BSGL=True,
)
mcmc_F.transform_dictionary = transform_dict
mcmc_F.run(
    walker_plot_args={"plot_det_stat": True, "injection_parameters": signal_parameters}
)
mcmc_F.print_summary()
mcmc_F.plot_corner(add_prior=True, truths=signal_parameters)
mcmc_F.plot_prior_posterior(injection_parameters=signal_parameters)
