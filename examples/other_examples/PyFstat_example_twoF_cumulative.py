import os
import numpy as np
import pyfstat

label = os.path.splitext(os.path.basename(__file__))[0]
outdir = os.path.join("PyFstat_example_data", label)

# Properties of the GW data
gw_data = {
    "sqrtSX": 1e-23,
    "tstart": 1000000000,
    "duration": 100 * 86400,
    "detectors": "H1,L1",
    "Band": 4,
    "Tsft": 1800,
}

# Properties of the signal
depth = 100
signal_parameters = {
    "F0": 30.0,
    "F1": -1e-10,
    "F2": 0,
    "Alpha": np.radians(83.6292),
    "Delta": np.radians(22.0144),
    "tref": gw_data["tstart"],
    "cosi": 1,
    "h0": gw_data["sqrtSX"] / depth,
}

data = pyfstat.Writer(
    label=label,
    outdir=outdir,
    **gw_data,
    **signal_parameters,
)
data.make_data()

# The predicted twoF, given by lalapps_predictFstat can be accessed by
twoF = data.predict_fstat()
print("Predicted twoF value: {}\n".format(twoF))

ifo_constraints = ["L1", "H1", None]
compute_fstat_per_ifo = [
    pyfstat.ComputeFstat(
        sftfilepattern=os.path.join(
            data.outdir,
            (f"{ifo_constraint[0]}*.sft" if ifo_constraint is not None else "*.sft"),
        ),
        tref=signal_parameters["tref"],
        binary=signal_parameters.get("asini", 0),
        minCoverFreq=-0.5,
        maxCoverFreq=-0.5,
    )
    for ifo_constraint in ifo_constraints
]

cumulative_f_stat_params = {
    key: signal_parameters[key]
    for key in signal_parameters
    if key
    in ["F0", "F1", "F2", "Alpha", "Delta", "asini", "period", "tp", "ecc", "argp"]
}
cumulative_f_stat_params.update(
    {"tstart": gw_data["tstart"], "tend": gw_data["tstart"] + gw_data["duration"]}
)

predict_f_stat_params = {
    key: signal_parameters.get(key, 0)
    for key in ["F0", "Alpha", "Delta", "psi", "cosi", "h0"]
}

for ind, compute_f_stat in enumerate(compute_fstat_per_ifo):
    taus, twoF = compute_f_stat.calculate_twoF_cumulative(**cumulative_f_stat_params)
    compute_f_stat.plot_twoF_cumulative(
        **cumulative_f_stat_params,
        label=label + (f"_{ifo_constraints[ind]}" if ind < 2 else "_H1L1"),
        outdir=outdir,
        injection_parameters=predict_f_stat_params,
        plot_label="Cumulative 2F"
        + (f" {ifo_constraints[ind]}" if ind < 2 else " H1 + L1"),
        custom_axis_kwargs={"title": "This is a custom title"},
    )
