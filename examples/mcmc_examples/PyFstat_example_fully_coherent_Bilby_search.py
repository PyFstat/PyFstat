"""
Bilby search: Fully coherent F-statistic
========================================

Directed Bilby search for an isolated CW signal using the PyFstat
fully coherent F-statistic likelihood.

This example mirrors the basic fully coherent MCMC example, but calls
``MCMCSearch.run_bilby()`` instead of ``MCMCSearch.run()``.
"""

import os

import numpy as np

import pyfstat
from pyfstat.utils import get_predict_fstat_parameters_from_dict

label = "PyFstatExampleFullyCoherentBilbySearch"
outdir = os.path.join("PyFstat_example_data", label)
logger = pyfstat.set_up_logger(label=label, outdir=outdir)

# Properties of the GW data
data_parameters = {
    "sqrtSX": 1e-23,
    "tstart": 1000000000,
    "duration": 20 * 86400,
    "detectors": "H1",
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
    label=label,
    outdir=outdir,
    **data_parameters,
    **signal_parameters,
)
data.make_data()

twoF = data.predict_fstat()
logger.info("Predicted twoF value: {}\n".format(twoF))

DeltaF0 = 5e-7
DeltaF1 = 5e-13
VF0 = (np.pi * data_parameters["duration"] * DeltaF0) ** 2 / 3.0
VF1 = (np.pi * data_parameters["duration"] ** 2 * DeltaF1) ** 2 * 4 / 45.0
logger.info("\nV={:1.2e}, VF0={:1.2e}, VF1={:1.2e}\n".format(VF0 * VF1, VF0, VF1))

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

search = pyfstat.MCMCSearch(
    label=label,
    outdir=outdir,
    sftfilepattern=data.sftfilepath,
    theta_prior=theta_prior,
    tref=mid_time,
    minStartTime=data_parameters["tstart"],
    maxStartTime=tend,
)
search.transform_dictionary = dict(
    F0=dict(subtractor=signal_parameters["F0"], symbol="$f-f^\\mathrm{s}$"),
    F1=dict(
        subtractor=signal_parameters["F1"], symbol="$\\dot{f}-\\dot{f}^\\mathrm{s}$"
    ),
)
result = search.run_bilby(
    sampler="dynesty",
    sampler_kwargs={
        "nlive": 400,
        "dlogz": 1.0,
    },
    save_pickle=False,
    save_bilby_logs=True,
    save_bilby_progress=True,
)
logger.info("Bilby result object: {}".format(result))
search.print_summary()
search.plot_corner(add_prior=True, truths=signal_parameters)
search.plot_prior_posterior(injection_parameters=signal_parameters)

search.generate_loudest()

search.print_summary()

# plot cumulative 2F, first building a dict as required for PredictFStat
d, maxtwoF = search.get_max_twoF()
for key, val in search.theta_prior.items():
    if key not in d:
        d[key] = val
d["h0"] = data.h0
d["cosi"] = data.cosi
d["psi"] = data.psi
PFS_input = get_predict_fstat_parameters_from_dict(d)
search.plot_cumulative_max(PFS_input=PFS_input)
