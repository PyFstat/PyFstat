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


def test_synth_CW(timestamps):
    synth = pyfstat.Synthesizer(
        label="Test",
        outdir="TestData/",
        Alpha=0,
        Delta=0,
        h0=0,
        cosi=0,
        psi=0,
        phi=0,
        detectors="H1",
        timestamps=timestamps,
        transientStartTime=0,
        transientTau=timestamps[-1] - timestamps[0],
        tstart=timestamps[0],
        randSeed=0,
    )

    twoF = synth.synth_Fstats(numDraws=1)
    logging.info(f"first draw of 2F: {twoF}")
    assert twoF > 0

    twoF = synth.synth_Fstats(numDraws=1000)
    logging.info(f"mean over 1000 draws of 2F: {np.mean(twoF)}")
    assert pytest.approx(np.mean(twoF), rel=0.01) == 4.0
