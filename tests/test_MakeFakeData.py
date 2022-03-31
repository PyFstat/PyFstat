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


def test_MakeFakeData(mfd_parameters, signal_parameters):
    mfd = pyfstat.make_sfts.MakeFakeData(**mfd_parameters)
    data = mfd.simulate_data(**signal_parameters)
    print(data)
