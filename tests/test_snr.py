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
        "detectors": "H1,L1",
        "SFTWindowType": "tukey",
        "SFTWindowParam": 0.001,
        "randSeed": 42,
    }


@pytest.fixture
def writer(data_parameters):
    extra_parameters = {}
    extra_parameters["label"] = "Test"
    extra_parameters["outdir"] = "TestData/"
    extra_parameters["F0"] = 10.0
    extra_parameters["Band"] = 0.1
    extra_parameters["sqrtSX"] = 1e-23

    this_writer = pyfstat.Writer(**{**data_parameters, **extra_parameters})
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


def compare_detector_states_series(dss_0, dss_1):
    """
    Check timestamps, Tsft, detector name and time off-set
    """
    if dss_0.length != dss_1.length:
        raise ValueError(
            f"Detector states series' length don't match: {dss_0.length} vs. {dss_1.length}"
        )
    for attribute in ["length", "system", "deltaT"]:
        attr_0 = getattr(dss_0, attribute)
        attr_1 = getattr(dss_1, attribute)
        if attr_0 != attr_1:
            raise ValueError(
                f"Attribute {attribute} doesn't match: {attr_0} vs. {attr_1}"
            )

    name_0 = dss_0.detector.frDetector.name
    name_1 = dss_1.detector.frDetector.name
    if name_0 != name_1:
        raise ValueError(f"Detector names don't match: {name_0} vs. {name_1}")

    for ind in range(dss_0.length):
        ts_0 = dss_0.data[ind].tGPS
        ts_1 = dss_1.data[ind].tGPS
        if ts_0 != ts_1:
            raise ValueError(f"Timetamps are not consistent: {ts_0} vs. {ts_1}")

    return True


def compare_multi_detector_states_series(mdss_0, mdss_1):
    """
    Loop over individual detector states series
    """
    if mdss_0.length != mdss_1.length:
        raise ValueError(
            f"Multi-detector states series' length don't match: {mdss_0.length} vs. {mdss_1.length}"
        )
    return all(
        compare_detector_states_series(mdss_0.data[ifo], mdss_1.data[ifo])
        for ifo in range(mdss_0.length)
    )


def test_DetectorStates(data_parameters, writer):
    # Test that both input formats of timestamps work consistently
    common_ts = writer.tstart + writer.Tsft * np.arange(writer.duration / writer.Tsft)
    timestamp_options = [
        {"detectors": "H1,L1", "timestamps": common_ts},
        {"timestamps": {key: common_ts for key in ("H1", "L1")}},
    ]

    mds = [
        pyfstat.DetectorStates().get_multi_detector_states(Tsft=writer.Tsft, **ts)
        for ts in timestamp_options
    ]
    assert compare_multi_detector_states_series(*mds)

    # test again with plain list instead of np.array
    mds2 = pyfstat.DetectorStates().get_multi_detector_states(
        Tsft=writer.Tsft, timestamps={key: list(common_ts) for key in ("H1", "L1")}
    )
    assert compare_multi_detector_states_series(mds[0], mds2)

    # test a wrong input that shouldn't work
    with pytest.raises(Exception):
        pyfstat.DetectorStates().get_multi_detector_states(
            Tsft=writer.Tsft, timestamps={"H1,L1": [0]}
        )

    # Test that SFT parsing also works consistently
    writer.make_data()
    from_sfts = pyfstat.DetectorStates().get_multi_detector_states_from_sfts(
        writer.sftfilepath, central_frequency=writer.F0
    )
    assert compare_multi_detector_states_series(from_sfts, mds[0])


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

    predicted_twoF, predicted_stdev_twoF = pyfstat.utils.predict_fstat(
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
    predicted_twoF, predicted_stdev_twoF = pyfstat.utils.predict_fstat(
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
