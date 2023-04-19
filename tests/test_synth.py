import logging
import os
import shutil
from numbers import Number

import numpy as np
import pytest

import pyfstat


@pytest.fixture
def timestamps():
    tstart = 700000000
    duration = 86400
    Tsft = 1800
    return np.arange(tstart, tstart + duration, Tsft)


@pytest.mark.parametrize("amp_priors", ["fixedamp", "PK2009"])
@pytest.mark.parametrize("sky_priors", ["targeted", "allsky"])
@pytest.mark.parametrize("h0", [0, 1])
@pytest.mark.parametrize("detectors", ["H1", "H1,L1"])
def test_synth_CW(timestamps, amp_priors, sky_priors, h0, detectors, numDraws=1000):
    if amp_priors == "fixedamp":
        priors = {
            "h0": h0,
            "cosi": 0,
            "psi": 0,
            "phi": 0,
        }
    elif amp_priors == "PK2009":
        priors = {
            "h0": {"stats.uniform": {"loc": 0.0, "scale": h0}},
            "cosi": {"stats.uniform": {"loc": -1.0, "scale": 2.0}},
            "psi": {"stats.uniform": {"loc": -0.25 * np.pi, "scale": 0.5 * np.pi}},
            "phi": {"stats.uniform": {"loc": 0.0, "scale": 2.0 * np.pi}},
        }
    if sky_priors == "targeted":
        priors["Alpha"] = 0
        priors["Delta"] = 0
    elif sky_priors == "allsky":
        priors["Alpha"] = {"stats.uniform": {"loc": 0.0, "scale": 2 * np.pi}}
        priors["Delta"] = {"uniform_sky_declination": {}}

    detstats = [
        "twoF",
        "maxTwoF",
        "BtSG",
    ]
    if len(detectors.split(",")) >= 2:
        detstats.append({"BSGL": {"Fstar0sc": 15}})
    randSeed = 0

    synth = pyfstat.Synthesizer(
        label="Test",
        outdir="TestData/",
        priors=priors,
        detectors=detectors,
        timestamps=timestamps,
        transientStartTime=0,
        transientTau=timestamps[-1] - timestamps[0],
        tstart=timestamps[0],
        randSeed=randSeed,
        detstats=detstats,
    )

    paramsGen = pyfstat.InjectionParametersGenerator(priors=priors, seed=randSeed)
    detstates = pyfstat.snr.DetectorStates()
    snr = pyfstat.SignalToNoiseRatio(
        detector_states=detstates.get_multi_detector_states(
            timestamps=timestamps,
            detectors=detectors,
            Tsft=timestamps[1] - timestamps[0],
        ),
        assumeSqrtSX=1,
    )
    twoF_from_snr2 = np.zeros(numDraws)
    for n in range(numDraws):
        injParams = paramsGen.draw()
        twoF_from_snr2[n], _ = snr.compute_twoF(**injParams)
    meanTwoF_from_snr2 = np.mean(twoF_from_snr2)
    logging.info(f"expected average twoF: {meanTwoF_from_snr2}")

    cands = synth.synth_candidates(numDraws=numDraws, keep_params=True)
    twoF = cands["maxTwoF"][0]
    logging.info(f"first draw of 2F: {twoF}")
    assert twoF > 0
    meanTwoF = np.mean(cands["maxTwoF"])
    logging.info(f"mean over {numDraws} draws of 2F: {meanTwoF}")
    assert pytest.approx(meanTwoF, rel=0.05) == meanTwoF_from_snr2
    if np.all([isinstance(prior, Number) for prior in priors.values()]):
        logging.info(
            "All parameters fixed, we should get a unique SNR for all draws..."
        )
        assert len(np.unique(cands["snr"])) == 1
    if isinstance(priors["h0"], Number):
        logging.info(f"Fixed h0={h0}, we should get the same back for all draws...")
        assert np.allclose(
            [cands["h0"][n] for n in range(numDraws)],
            h0,
            rtol=1e-9,
            atol=0,
        )
    # FIXME: add more tests of other parameters and detstats

    if os.path.isdir(synth.outdir):
        shutil.rmtree(synth.outdir)
