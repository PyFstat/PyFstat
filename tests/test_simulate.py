import os
import shutil

import numpy as np
import pytest

import pyfstat


@pytest.fixture
def data_params():
    return {
        "fMin": 10,
        "Band": 1,
        "sqrtSX": "1e-23",
        "timestamps": dict.fromkeys(["H1", "L1"], 1000000000 + 1800 * np.arange(5)),
        "Tsft": 1800,
        "randSeed": 314192,
    }


# @pytest.fixture
# def signal_parameters(obsrun_parameters):
#    return {
#        "F0": mfd_parameters["fMin"] + 0.5 * mfd_parameters["Band"],
#        "F1": 0.0,
#        "Alpha": 0,
#        "Delta": 0,
#        "refTime": mfd_parameters["timestamps"][0],
#        "h0": 1e-23,
#        "cosi": 1.0,
#        "psi": 0,
#        "phi0": 0,
#    }
#
# @pytest.fixture
# def noise_writer(obsrun_parameters, tmp_path):
#    writer_parameters = {**obsrun_parameters}
#
#    for new, old in [("tref", "refTime")]:
#        writer_parameters[new] = w_parameters.pop(old)
#    writer_parameters["detectors"] = ",".join(writer_parameters.pop("detectors"))
#
#    this_writer = pyfstat.Writer(**writer_parameters, outdir=tmp_path, label="NoiseOnly")
#    return this_writer
#
#
# @pytest.fixture
# def signal_writer(obsrun_parameters, signal_parameters, tmp_path):
#    writer_parameters = {**obsrun_parameters, **signal_parameters}
#
#    for remove in ["fMin"]:
#        writer_parameters.pop(remove)
#    for new, old in [("tref", "refTime"), ("phi", "phi0")]:
#        writer_parameters[new] = w_parameters.pop(old)
#    writer_parameters["detectors"] = ",".join(writer_parameters.pop("detectors"))
#
#    this_writer = pyfstat.Writer(**writer_parameters, outdir=tmp_path, label="NoiseAndSignal")
#    return this_writer


def test_MakeFakeData_set_data_params(data_params):

    mfd = pyfstat.MakeFakeData()
    print(type(data_params["timestamps"]))
    mfd.set_data_params(**data_params)
    for key, val in data_params.items():
        assert getattr(mfd.data_params, key) == value


# def test_MakeFakeData(signal_writer, obsrun_parameters, signal_parameters):
#    mfd = pyfstat.make_sfts.MakeFakeData()
#
#    mfd.set_run_parameters(**obsrun_parameters)
#    mfd_freq, mfd_timestamps, mfd_amplitudes = mfd.simulate(**signal_parameters)
#
#    signal_writer.make_data()
#    w_freq, w_timestamps, w_amplitudes = pyfstat.utils.get_sft_as_arrays(
#        signal_writer.sftfilepath
#    )
#
#    np.testing.assert_allclose(mfd_freq, w_freq)
#    for ifo in mfd_parameters["detectors"]:
#        np.testing.assert_allclose(mfd_timestamps[ifo], w_timestamps[ifo])
#        np.testing.assert_allclose(mfd_amplitudes[ifo], w_amplitudes[ifo])
#
# def test_MakeFakeData_noiseSFTs(noise_writer, obsrun_parameters, signal_parameters):
#
#    noise_writer.make_data()
#
#    mfd = pyfstat.make_sfts.MakeFakeData()
#
#    mfd.set_from_noise_sfts(noise_writer.sftfilepath, noise_writer.SFTWindowType, noise_writer.SFTWindowBeta)
#    mfd_freq, mfd_timestamps, mfd_amplitudes = mfd.simulate(**signal_parameters)
#
#    writer = pyfstat.Writer(noiseSFTs=noise_writer.sftfilepath, **signal_parameters)
#    w_freq, w_timestamps, w_amplitudes = pyfstat.utils.get_sft_as_arrays(
#        consistent_writer.sftfilepath
#    )
#
#    np.testing.assert_allclose(mfd_freq, w_freq)
#    for ifo in mfd_parameters["detectors"]:
#        np.testing.assert_allclose(mfd_timestamps[ifo], w_timestamps[ifo])
#        np.testing.assert_allclose(mfd_amplitudes[ifo], w_amplitudes[ifo])
