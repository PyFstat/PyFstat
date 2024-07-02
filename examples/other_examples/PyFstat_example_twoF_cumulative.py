"""
Cumulative coherent 2F
======================

Compute the cumulative coherent F-statistic of a signal candidate.
"""

import os

import numpy as np

import pyfstat
from pyfstat.utils import get_predict_fstat_parameters_from_dict

label = "PyFstatExampleTwoFCumulative"
outdir = os.path.join("PyFstat_example_data", label)
logger = pyfstat.set_up_logger(label=label, outdir=outdir)

# Properties of the GW data
gw_data = {
    "sqrtSX": 1e-23,
    "tstart": 1000000000,
    "duration": 10 * 86400,
    "detectors": "H1,L1",
    "Band": 4,
    "Tsft": 1800,
}

# Properties of the signal
depth = 30
phase_parameters = {
    "F0": 30.0,
    "F1": -1e-10,
    "F2": 0,
    "Alpha": np.radians(83.6292),
    "Delta": np.radians(22.0144),
    "tref": gw_data["tstart"],
    "asini": 10,
    "period": 10 * 3600 * 24,
    "tp": gw_data["tstart"] + gw_data["duration"] / 2.0,
    "ecc": 0,
    "argp": 0,
}
amplitude_parameters = {
    "h0": gw_data["sqrtSX"] / depth,
    "cosi": 1,
    "phi": np.pi,
    "psi": np.pi / 8,
}

PFS_input = get_predict_fstat_parameters_from_dict(
    {**phase_parameters, **amplitude_parameters}
)

# Let me grab tref here, since it won't really be needed in phase_parameters
tref = phase_parameters.pop("tref")
data = pyfstat.BinaryModulatedWriter(
    label=label,
    outdir=outdir,
    tref=tref,
    **gw_data,
    **phase_parameters,
    **amplitude_parameters,
)
data.make_data()

# The predicted twoF (expectation over noise realizations) can be accessed by
twoF = data.predict_fstat()
logger.info("Predicted twoF value: {}\n".format(twoF))

# Create a search object for each of the possible SFT combinations
# (H1 only, L1 only, H1 + L1).
ifo_constraints = ["L1", "H1", None]
compute_fstat_per_ifo = [
    pyfstat.ComputeFstat(
        sftfilepattern=os.path.join(
            data.outdir,
            (f"{ifo_constraint[0]}*.sft" if ifo_constraint is not None else "*.sft"),
        ),
        tref=data.tref,
        binary=phase_parameters.get("asini", 0),
        minCoverFreq=-0.5,
        maxCoverFreq=-0.5,
    )
    for ifo_constraint in ifo_constraints
]

for ind, compute_f_stat in enumerate(compute_fstat_per_ifo):
    compute_f_stat.plot_twoF_cumulative(
        label=label + (f"_{ifo_constraints[ind]}" if ind < 2 else "_H1L1"),
        outdir=outdir,
        savefig=True,
        CFS_input=phase_parameters,
        PFS_input=PFS_input,
        custom_ax_kwargs={
            "title": "How does 2F accumulate over time?",
            "label": "Cumulative 2F"
            + (f" {ifo_constraints[ind]}" if ind < 2 else " H1 + L1"),
        },
    )
