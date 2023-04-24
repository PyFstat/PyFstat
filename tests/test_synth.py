import logging
import os
import shutil
from glob import glob
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


@pytest.fixture
def amp_priors(amp_val):
    prior_choices = {
        "allfixed": {
            "h0": amp_val,
            "cosi": 0,
            "psi": 0,
            "phi": 0,
        },
        "unifangles": {
            "cosi": {"stats.uniform": {"loc": -1.0, "scale": 2.0}},
            "psi": {"stats.uniform": {"loc": -0.25 * np.pi, "scale": 0.5 * np.pi}},
            "phi": {"stats.uniform": {"loc": 0.0, "scale": 2.0 * np.pi}},
        },
        "fixedsnr": {
            "snr": amp_val,
        },
        "logunifsnr": {
            "snr": {"stats.loguniform": {"a": 1, "b": 10}},
        },
        "unifh0": {
            "h0": {"stats.uniform": {"loc": 0.0, "scale": amp_val}},
        },
    }
    return prior_choices


@pytest.fixture
def sky_priors(amp_val):
    prior_choices = {
        "targeted": {
            "Alpha": 0,
            "Delta": 0,
        },
        "allsky": {
            "Alpha": {"stats.uniform": {"loc": 0.0, "scale": 2 * np.pi}},
            "Delta": {"uniform_sky_declination": {}},
        },
    }
    return prior_choices


@pytest.mark.parametrize(
    "amp_prior_choice", ["allfixed", "unifh0", "fixedsnr", "logunifsnr"]
)
@pytest.mark.parametrize("sky_prior_choice", ["targeted", "allsky"])
@pytest.mark.parametrize("amp_val", [0, 1])
@pytest.mark.parametrize("detectors", ["H1", "H1,L1"])
def test_synth_CW(
    timestamps,
    amp_priors,
    sky_priors,
    amp_prior_choice,
    sky_prior_choice,
    amp_val,
    detectors,
    numDraws=1000,
):
    if amp_prior_choice == "allfixed":
        priors = amp_priors["allfixed"]
    else:
        priors = amp_priors["unifangles"]
        priors.update(amp_priors[amp_prior_choice])
        if amp_val == 0 and amp_prior_choice in ["unifh0", "logunifsnr"]:
            pytest.skip()
    priors.update(sky_priors[sky_prior_choice])

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
    returns = ["detstats", "parameters", "atoms"]
    try:
        cands = synth.synth_candidates(
            numDraws=numDraws,
            returns=returns,
            hdf5_outputs=["detstats", "parameters", "atoms"],
        )
    except ImportError:
        logging.warning("hdf5 not available, skipping output tests.")
        cands = synth.synth_candidates(
            numDraws=numDraws,
            returns=returns,
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

    # only do lalpulsar comparison twice, in the simplest and a more complex prior case
    if (
        amp_val == 1
        and detectors == "H1,L1"
        and (
            (amp_prior_choice == "allfixed" and sky_prior_choice == "targeted")
            or (amp_prior_choice == "fixedsnr" and sky_prior_choice == "allsky")
        )
    ):
        _test_synth_against_lalpulsar_exec(
            timestamps, priors, detectors, numDraws, randSeed, synth.outdir, cands
        )

    if os.path.isdir(synth.outdir):
        shutil.rmtree(synth.outdir)


def _test_synth_against_lalpulsar_exec(
    timestamps, priors, detectors, numDraws, randSeed, outdir, cands
):
    outputParams = os.path.join(outdir, "lalpulsar_synth_params.dat")
    outputStats = os.path.join(outdir, "lalpulsar_synth_stats.dat")
    outputAtoms = os.path.join(outdir, "lalpulsar_synth_atoms.dat")
    Tsft = timestamps[1] - timestamps[0]
    params = {
        "IFOs": detectors,
        "dataStartGPS": timestamps[0],
        "dataDuration": timestamps[-1] - timestamps[0] + Tsft,
        "injectWindow-type": "none",
        "numDraws": numDraws,
        "randSeed": randSeed,
        "outputStats": outputStats,
        "outputInjParams": outputParams,
        "outputAtoms": outputAtoms,
    }
    fixed = np.all([isinstance(prior, Number) for prior in priors.values()])
    if fixed:
        params.update(priors)
        params["fixedh0Nat"] = params.pop("h0")
    else:
        params["fixedSNR"] = priors["snr"]
    cl = "lalpulsar_synthesizeTransientStats --computeFtotal " + " ".join(
        [f"--{key}={val}" for key, val in params.items()]
    )
    pyfstat.utils.run_commandline(cl)
    assert os.path.isfile(outputParams)
    assert os.path.isfile(outputStats)
    atomsFiles = glob(outputAtoms + "*")
    assert len(atomsFiles) == numDraws
    params_lalpulsar = pyfstat.utils.read_txt_file_with_header(
        outputParams, comments="%"
    )
    par_names = {
        "Alpha": "Alpha",
        "Delta": "Delta",
        "h0": "h0",
        "cosi": "cosi",
        "psi": "psi",
        "phi": "phi0",
        "snr": "SNR",
    }
    for k1, k2 in par_names.items():
        if fixed:
            logging.info(
                "Fixed priors, should have unique and identical values for all parameters..."
            )
            assert len(np.unique(cands[k1])) == 1
            assert len(np.unique(params_lalpulsar[k2])) == 1
            assert pytest.approx(cands[k1][0], rel=1e-5) == params_lalpulsar[k2][0]
        else:
            mean_pyfstat = np.mean(cands[k1])
            mean_lalpulsar = np.mean(params_lalpulsar[k2])
            logging.info(
                f"Mean {k1} from PyFstat: {mean_pyfstat}, mean {k2} from lalpulsar: {mean_lalpulsar}"
            )
            if np.abs(mean_lalpulsar) < 0.1:
                assert pytest.approx(mean_pyfstat, abs=0.1) == mean_lalpulsar
            else:
                assert pytest.approx(mean_pyfstat, rel=0.1) == mean_lalpulsar
    stats_lalpulsar = pyfstat.utils.read_txt_file_with_header(outputStats, comments="%")
    par_names = {
        "twoF": "fkdot3",  # hacky solution where total (CW) 2F is stored in this field
    }
    for k1, k2 in par_names.items():
        # stats won't be unique even for fully fixed prior, due to noise variance
        mean_pyfstat = np.mean(cands[k1])
        mean_lalpulsar = np.mean(stats_lalpulsar[k2])
        logging.info(
            f"Mean {k1} from PyFstat: {mean_pyfstat}, mean {k2} from lalpulsar: {mean_lalpulsar}"
        )
        assert pytest.approx(mean_pyfstat, rel=0.1) == mean_lalpulsar
    for n in [0, numDraws - 1]:
        atoms_lalpulsar = pyfstat.utils.read_txt_file_with_header(
            atomsFiles[0], comments="%"
        )
        for X, IFO in enumerate(detectors.split(",")):
            atomsX = pyfstat.utils.reshape_FstatAtomVector_to_array(
                cands["atoms"][n].data[X]
            )
            assert np.shape(atomsX)[0] == np.shape(atoms_lalpulsar)[0] / 2
            assert np.shape(atomsX)[1] == len(atoms_lalpulsar[0])


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
