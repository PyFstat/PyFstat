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


@pytest.mark.parametrize("amp_priors", ["fixedamp", "PK2009", "fixedsnr", "logunisnr"])
@pytest.mark.parametrize("sky_priors", ["targeted", "allsky"])
@pytest.mark.parametrize("h0", [0, 1])  # also used for snr
@pytest.mark.parametrize("detectors", ["H1", "H1,L1"])
def test_synth_CW(timestamps, amp_priors, sky_priors, h0, detectors, numDraws=1000):
    if amp_priors == "fixedamp":
        priors = {
            "h0": h0,
            "cosi": 0,
            "psi": 0,
            "phi": 0,
        }
    else:
        priors = {
            "cosi": {"stats.uniform": {"loc": -1.0, "scale": 2.0}},
            "psi": {"stats.uniform": {"loc": -0.25 * np.pi, "scale": 0.5 * np.pi}},
            "phi": {"stats.uniform": {"loc": 0.0, "scale": 2.0 * np.pi}},
        }
        if amp_priors == "fixedsnr":
            priors["snr"] = h0
        if amp_priors == "logunisnr":
            if h0 == 0:
                pytest.skip()
            priors["snr"] = {"stats.loguniform": {"a": 1, "b": 10}}
        elif amp_priors == "PK2009":
            if h0 == 0:
                pytest.skip()
            priors["h0"] = {"stats.uniform": {"loc": 0.0, "scale": h0}}
    if sky_priors == "targeted":
        priors["Alpha"] = 0
        priors["Delta"] = 0
    elif sky_priors == "allsky":
        priors["Alpha"] = {"stats.uniform": {"loc": 0.0, "scale": 2 * np.pi}}
        priors["Delta"] = {"uniform_sky_declination": {}}

    detstats = ["twoF"]
    if len(detectors.split(",")) >= 2:
        detstats.append("twoFX")
        detstats.append({"BSGL": {"Fstar0sc": 15}})
    randSeed = 1

    # for comparison: predicted 2F from SignalToNoiseRatio
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
    allParams = paramsGen.draw_many(numDraws)
    for n in range(numDraws):
        injParams = {key: val[n] for key, val in allParams.items()}
        if "snr" in priors.keys():
            injSNR = injParams.pop("snr")
            injParams["h0"] = snr.compute_h0_from_snr2(**injParams, snr2=injSNR**2)
        twoF_from_snr2[n], _ = snr.compute_twoF(**injParams)
    meanTwoF_from_snr2 = np.mean(twoF_from_snr2)
    logging.info(f"expected average twoF: {meanTwoF_from_snr2}")

    # the actual synthing
    synth = pyfstat.Synthesizer(
        label="TestSynthCWs",
        outdir="TestData/synth",
        priors=priors,
        detectors=detectors,
        timestamps=timestamps,
        randSeed=randSeed,
        detstats=detstats,
    )
    try:
        cands = synth.synth_candidates(
            numDraws=numDraws,
            returns=["detstats", "parameters"],
            hdf5_outputs=["detstats", "parameters", "atoms"],
        )
    except ImportError:
        logging.warning("hdf5 not available, skipping output tests.")
        cands = synth.synth_candidates(
            numDraws=numDraws,
            returns=["detstats", "parameters"],
            hdf5_outputs=[],
        )

    twoF = cands["twoF"][0]
    logging.info(f"first draw of 2F: {twoF}")
    assert twoF > 0
    meanTwoF = np.mean(cands["twoF"])
    logging.info(f"mean over {numDraws} draws of 2F: {meanTwoF}")
    assert pytest.approx(meanTwoF, rel=0.05) == meanTwoF_from_snr2
    if np.all([isinstance(prior, Number) for prior in priors.values()]):
        logging.info(
            "All parameters fixed, we should get a unique SNR for all draws..."
        )
        assert len(np.unique(cands["snr"])) == 1
    for param in ["h0", "snr"]:
        if param in priors and isinstance(priors[param], Number):
            logging.info(
                f"Fixed {param}={priors[param]}, we should get the same back for all draws..."
            )
            assert np.allclose(
                [cands[param][n] for n in range(numDraws)],
                priors[param],
                rtol=1e-9,
                atol=0,
            )
    # FIXME: add more tests of other parameters and detstats

    if os.path.isdir(synth.outdir):
        shutil.rmtree(synth.outdir)


def test_synth_tCW(timestamps):
    detstats = [
        "twoF",
        "maxTwoF",
        "BtSG",
    ]
    priors = {
        "h0": 0,
        "cosi": 0,
        "psi": 0,
        "phi": 0,
        "Alpha": 0,
        "Delta": 0,
    }
    logging.warning("Transients are not yet implemented!")
    with pytest.raises(NotImplementedError):
        pyfstat.Synthesizer(
            label="TestSynthTransients",
            outdir="TestData/synth",
            priors=priors,
            detectors="H1",
            timestamps=timestamps,
            transientWindowType="rect",
            transientStartTime=0,
            transientTau=timestamps[-1] - timestamps[0],
            detstats=detstats,
        )
