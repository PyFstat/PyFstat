import os
import shutil

import numpy as np
import pytest

import pyfstat


@pytest.fixture
def obsrun_parameters():
    return {
        "fMin": 1.0,
        "Band": 1,
        "sqrtSX": dict.fromkeys(["H1", "L1"], 1e-23),
        "timestamps": dict.fromkeys(["H1", "L1"], 1000000000 + 1800 * np.arange(5)),
        "Tsft": 1800,
        "randSeed": 314192,
    }


@pytest.fixture
def signal_parameters(obsrun_parameters):
    return {
        "F0": obsrun_parameters["F0"] + 0.5 * obsrun_parameters["Band"],
        "F1": 0.0,
        "Alpha": 0,
        "Delta": 0,
        "refTime": obsrun_parameters["timestamps"]["H1"][0],
        "h0": 1e-23,
        "cosi": 1.0,
        "psi": 0,
        "phi0": 0,
    }


@pytest.fixture
def noise_writer(obsrun_parameters, tmp_path):
    return pyfstat.Writer(**obsrun_parameters, outdir=tmp_path, label="NoiseOnly")


@pytest.fixture
def signal_writer(obsrun_parameters, signal_parameters, tmp_path):
    writer_parameters = {**obsrun_parameters, **signal_parameters}

    for remove in ["F0"]:
        writer_parameters.pop(remove)
    for new, old in [("tref", "refTime"), ("phi", "phi0")]:
        writer_parameters[new] = signal_parameters.pop(old)

    return pyfstat.Writer(**writer_parameters, outdir=tmp_path, label="NoiseAndSignal")


def test_MakeFakeData_set_data_params(obsrun_parameters):

    mfd = pyfstat.MakeFakeData()
    mfd.set_data_params(**obsrun_parameters)
    for key in ["randSeed", "Band"]:
        assert getattr(mfd.data_params, key) == obsrun_parameters[key]

    detectors = list(obsrun_parameters["timestamps"].keys())
    num_detectors = len(detectors)

    assert mfd.data_params.multiTimestamps.length == num_detectors

    print("*" * 50)
    print("Let's see if segfault....")
    for ifo_ind, ifo in enumerate(detectors):
        # This fails for some reason...?
        print(mfd.data_params.multiTimestamps.data[0].data[0])
    for ind in range(mfd.data_params.multiTimestamps.data[0].length):
        print(ind)
    print("Good, no segfault")
    # np.testing.assert_equal(
    #    mfd.data_params.multiTimestamps.data[0].data[0],
    #    data_params["timestamps"][detectors[0]],
    # )


def test_MakeFakeData(signal_writer, obsrun_parameters, signal_parameters):
    mfd = pyfstat.make_sfts.MakeFakeData()

    mfd.set_run_parameters(**obsrun_parameters)
    mfd_freq, mfd_timestamps, mfd_amplitudes = mfd.simulate(**signal_parameters)

    signal_writer.make_data()
    w_freq, w_timestamps, w_amplitudes = pyfstat.utils.get_sft_as_arrays(
        signal_writer.sftfilepath
    )

    np.testing.assert_allclose(mfd_freq, w_freq)
    for ifo in mfd_parameters["detectors"]:
        np.testing.assert_allclose(mfd_timestamps[ifo], w_timestamps[ifo])
        np.testing.assert_allclose(mfd_amplitudes[ifo], w_amplitudes[ifo])


def test_MakeFakeData_noiseSFTs(noise_writer, obsrun_parameters, signal_parameters):

    noise_writer.make_data()

    mfd = pyfstat.make_sfts.MakeFakeData()

    mfd.set_from_noise_sfts(
        noise_writer.sftfilepath, noise_writer.SFTWindowType, noise_writer.SFTWindowBeta
    )
    mfd_freq, mfd_timestamps, mfd_amplitudes = mfd.simulate(**signal_parameters)

    writer = pyfstat.Writer(noiseSFTs=noise_writer.sftfilepath, **signal_parameters)
    w_freq, w_timestamps, w_amplitudes = pyfstat.utils.get_sft_as_arrays(
        consistent_writer.sftfilepath
    )

    np.testing.assert_allclose(mfd_freq, w_freq)
    for ifo in mfd_parameters["detectors"]:
        np.testing.assert_allclose(mfd_timestamps[ifo], w_timestamps[ifo])
        np.testing.assert_allclose(mfd_amplitudes[ifo], w_amplitudes[ifo])
