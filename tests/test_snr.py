import os
import shutil

import numpy as np
import pytest

import pyfstat


@pytest.fixture
def data_parameters():
    return {
        "sqrtSX": 1,
        "Tsft": 1800,
        "tstart": 700000000,
        "duration": 4 * 1800,
        "detectors": "H1",
        "SFTWindowType": "tukey",
        "SFTWindowBeta": 0.001,
        "randSeed": 42,
    }


@pytest.fixture
def writer(data_parameters):
    data_parameters["label"] = "Test"
    data_parameters["outdir"] = "TestData/"
    data_parameters["F0"] = 10.0
    data_parameters["Band"] = 0.1
    data_parameters["sqrtSX"] = 1e-23

    this_writer = pyfstat.Writer(**data_parameters)
    yield this_writer
    if os.path.isdir(this_writer.outdir):
        shutil.rmtree(this_writer.outdir)


@pytest.fixture
def multi_detector_states(data_parameters):
    ds = pyfstat.snr.DetectorStates()

    Tsft = data_parameters["Tsft"]
    tstart = data_parameters["tstart"]
    ts = np.arange(tstart, tstart + data_parameters["duration"], Tsft)
    detectors = data_parameters["detectors"]

    return ds.get_multi_detector_states(
        timestamps=ts, detectors=detectors, Tsft=Tsft, time_offset=Tsft / 2
    )


@pytest.fixture
def snr_object(data_parameters, multi_detector_states):
    return pyfstat.SignalToNoiseRatio(
        detector_states=multi_detector_states,
        assumeSqrtSX=data_parameters["sqrtSX"],
    )


def test_SignalToNoiseRatio(writer, multi_detector_states):

    params = {
        "h0": 1e-23,
        "cosi": 0,
        "psi": 0,
        "phi": 0,
        "Alpha": 0,
        "Delta": 0,
    }
    params_pfs = params.copy()
    params_pfs.pop("phi")

    # Test compute SNR using assumeSqrtSX
    snr = pyfstat.SignalToNoiseRatio(
        detector_states=multi_detector_states,
        assumeSqrtSX=writer.sqrtSX,
    )
    twoF_from_snr2, twoF_stdev_from_snr2 = snr.compute_twoF(**params)

    predicted_twoF, predicted_stdev_twoF = pyfstat.helper_functions.predict_fstat(
        **params_pfs,
        minStartTime=writer.tstart,
        duration=writer.duration,
        IFOs=writer.detectors,
        assumeSqrtSX=snr.assumeSqrtSX,
    )
    np.testing.assert_allclose(twoF_from_snr2, predicted_twoF, rtol=1e-3)
    np.testing.assert_allclose(twoF_stdev_from_snr2, predicted_stdev_twoF, rtol=1e-3)

    # Test compute SNR using noise SFTs
    writer.make_data()
    snr = pyfstat.SignalToNoiseRatio.from_sfts(
        F0=writer.F0,
        sftfilepath=writer.sftfilepath,
        time_offset=writer.Tsft / 2,
    )
    twoF_from_snr2, twoF_stdev_from_snr2 = snr.compute_twoF(**params)
    predicted_twoF, predicted_stdev_twoF = pyfstat.helper_functions.predict_fstat(
        **params_pfs,
        F0=writer.F0,
        sftfilepattern=writer.sftfilepath,
    )
    np.testing.assert_allclose(twoF_from_snr2, predicted_twoF, rtol=1e-3)
    np.testing.assert_allclose(twoF_stdev_from_snr2, predicted_stdev_twoF, rtol=1e-3)


def test_compute_h0_from_snr2(snr_object):

    params = {
        "h0": 1e-23,
        "cosi": 0,
        "psi": 0,
        "phi": 0,
        "Alpha": 0,
        "Delta": 0,
    }

    params_no_h0 = {key: value for key, value in params.items() if key != "h0"}

    snr2 = snr_object.compute_snr2(**params)
    h0 = snr_object.compute_h0_from_snr2(**params_no_h0, snr2=snr2)

    np.testing.assert_allclose(params["h0"], h0)
