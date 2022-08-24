import os
import shutil

import numpy as np
import pytest

import pyfstat


@pytest.fixture
def mfd_parameters():
    return {
        "fMin": 10,
        "Band": 1,
        "sqrtSX": "1e-23",
        "timestamps": 1000000000 + 1800 * np.arange(5),
        "Tsft": 1800,
        "detectors": ["H1", "L1"],
        "randSeed": 314192,
    }


@pytest.fixture
def signal_parameters(mfd_parameters):
    return {
        "F0": mfd_parameters["fMin"] + 0.5 * mfd_parameters["Band"],
        "F1": 0.0,
        "Alpha": 0,
        "Delta": 0,
        "refTime": mfd_parameters["timestamps"][0],
        "h0": 1e-23,
        "cosi": 1.0,
        "psi": 0,
        "phi0": 0,
    }


@pytest.fixture
def consistent_writer(mfd_parameters, signal_parameters):
    w_parameters = {**mfd_parameters, **signal_parameters}

    for remove in ["fMin"]:
        w_parameters.pop(remove)
    for new, old in [("tref", "refTime"), ("phi", "phi0")]:
        w_parameters[new] = w_parameters.pop(old)
    w_parameters["detectors"] = ",".join(w_parameters.pop("detectors"))

    this_writer = pyfstat.Writer(**w_parameters, outdir="MakeFakeDataTest")
    yield this_writer
    if os.path.isdir(this_writer.outdir):
        shutil.rmtree(this_writer.outdir)


def test_MakeFakeData(consistent_writer, mfd_parameters, signal_parameters):
    mfd = pyfstat.make_sfts.MakeFakeData(**mfd_parameters)
    mfd_freq, mfd_timestamps, mfd_amplitudes = mfd.simulate_data(**signal_parameters)

    consistent_writer.make_data()
    w_freq, w_timestamps, w_amplitudes = pyfstat.helper_functions.get_sft_as_arrays(
        consistent_writer.sftfilepath
    )

    np.testing.assert_allclose(mfd_freq, w_freq)
    for ifo in mfd_parameters["detectors"]:
        np.testing.assert_allclose(mfd_timestamps[ifo], w_timestamps[ifo])
        np.testing.assert_allclose(mfd_amplitudes[ifo], w_amplitudes[ifo])
