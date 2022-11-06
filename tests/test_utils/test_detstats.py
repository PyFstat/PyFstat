import lalpulsar
import numpy as np
import pytest

import pyfstat


def test_get_canonical_detstat_name():
    for canonical_name in pyfstat.utils.detstats.detstat_synonyms.keys():
        assert (
            pyfstat.utils.get_canonical_detstat_name(canonical_name) == canonical_name
        )
        for synonym in pyfstat.utils.detstats.detstat_synonyms[canonical_name]:
            assert pyfstat.utils.get_canonical_detstat_name(synonym) == canonical_name
    with pytest.raises(ValueError):
        pyfstat.utils.get_canonical_detstat_name("nostat")


def test_parse_detstats():
    detstats_parsed, params = pyfstat.utils.parse_detstats("2F")
    assert detstats_parsed == [pyfstat.utils.get_canonical_detstat_name("2F")]
    assert params == {}
    detstats_parsed, params = pyfstat.utils.parse_detstats("2F,2FX")
    assert detstats_parsed == [
        pyfstat.utils.get_canonical_detstat_name(stat) for stat in ["2F", "2FX"]
    ]
    assert params == {}
    detstats_parsed, params = pyfstat.utils.parse_detstats(
        ["2F", {"BSGL": {"Fstar0sc": 15}}, "lnBtSG"]
    )
    BSGL = pyfstat.utils.get_canonical_detstat_name("BSGL")
    assert len(np.setdiff1d(detstats_parsed, ["twoF", "twoFX", BSGL, "lnBtSG"])) == 0
    assert len(params) == 1
    assert BSGL in params.keys()
    assert len(params[BSGL]) == 1
    assert "Fstar0sc" in params[BSGL]
    assert params[BSGL]["Fstar0sc"] == 15
    with pytest.raises(ValueError):
        pyfstat.utils.parse_detstats("test")
    with pytest.raises(ValueError):
        pyfstat.utils.parse_detstats("2F,test2")
    with pytest.raises(ValueError):
        pyfstat.utils.parse_detstats("BSGL")
    with pytest.raises(ValueError):
        pyfstat.utils.parse_detstats([{"2F": 42}])
    with pytest.raises(ValueError):
        pyfstat.utils.parse_detstats([{"2F": {"param": 42}}])
    with pytest.raises(ValueError):
        pyfstat.utils.parse_detstats([{"BSGL": {"param": 42}}])


def test_get_BSGL_setup():
    for nSeg in [1, 100]:
        setup = pyfstat.utils.get_BSGL_setup(
            numDetectors=2,
            numSegments=1,
            Fstar0sc=15,
        )
        assert type(setup) == lalpulsar.BSGLSetup
    for oLGX in [None, [0.5, 0.5]]:
        setup = pyfstat.utils.get_BSGL_setup(
            numDetectors=2,
            numSegments=1,
            Fstar0sc=15,
        )
        assert type(setup) == lalpulsar.BSGLSetup
    with pytest.raises(ValueError):
        pyfstat.utils.get_BSGL_setup(
            numDetectors=1,
            numSegments=1,
            Fstar0sc=15,
        )
    with pytest.raises(ValueError):
        pyfstat.utils.get_BSGL_setup(
            numDetectors=2,
            numSegments=1,
            Fstar0sc=15,
            oLGX=[0.5, 0.5, 0.5],
        )


def test_compute_Fstar0sc_from_p_val_threshold():
    Fstar0c = pyfstat.utils.compute_Fstar0sc_from_p_val_threshold()
    assert Fstar0c == pytest.approx(16.7, 1e-2)
    Fstar0sc = pyfstat.utils.compute_Fstar0sc_from_p_val_threshold(numSegments=100)
    assert Fstar0sc > Fstar0c
    with pytest.raises(RuntimeError):
        pyfstat.utils.compute_Fstar0sc_from_p_val_threshold(maxFstar0=10)
    with pytest.raises(ValueError):
        pyfstat.utils.compute_Fstar0sc_from_p_val_threshold(p_val_threshold=-1)
    with pytest.raises(ValueError):
        pyfstat.utils.compute_Fstar0sc_from_p_val_threshold(maxFstar0=-1)
