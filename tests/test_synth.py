import logging

import numpy as np
import pytest

import pyfstat


@pytest.fixture
def timestamps():
    tstart = 700000000
    duration = 86400
    Tsft = 1800
    return np.arange(tstart, tstart + duration, Tsft)


@pytest.mark.parametrize("h0", [0, 1])
@pytest.mark.parametrize("detectors", ["H1", "H1,L1"])
def test_synth_CW(timestamps, h0, detectors, numDraws=1000):

    signal_params = {
        "h0": h0,
        "cosi": 0,
        "psi": 0,
        "phi": 0,
        "Alpha": 0,
        "Delta": 0,
    }

    synth = pyfstat.Synthesizer(
        label="Test",
        outdir="TestData/",
        **signal_params,
        detectors=detectors,
        timestamps=timestamps,
        transientStartTime=0,
        transientTau=timestamps[-1] - timestamps[0],
        tstart=timestamps[0],
        randSeed=0,
    )

    params_pfs = signal_params.copy()
    params_pfs.pop("phi")
    detstates = pyfstat.snr.DetectorStates()
    snr = pyfstat.SignalToNoiseRatio(
        detector_states=detstates.get_multi_detector_states(
            timestamps=timestamps,
            detectors=detectors,
            Tsft=timestamps[1] - timestamps[0],
        ),
        assumeSqrtSX=1,
    )
    twoF_from_snr2, twoF_stdev_from_snr2 = snr.compute_twoF(**signal_params)
    logging.info(f"expected twoF: {twoF_from_snr2}")

    twoF = synth.synth_Fstats(numDraws=1)
    logging.info(f"first draw of 2F: {twoF}")
    assert twoF > 0

    twoF = synth.synth_Fstats(numDraws=numDraws)
    logging.info(f"mean over {numDraws} draws of 2F: {np.mean(twoF)}")
    assert pytest.approx(np.mean(twoF), rel=0.05) == twoF_from_snr2
