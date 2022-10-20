import itertools

import numpy as np
import pytest

import pyfstat


def test_gps_to_datestr_utc():

    gps = 1000000000
    # reference from lal_tconvert on en_US.UTF-8 locale
    # but instead from the "GMT" bit it places in the middle,
    # we put the timezone info at the end via datetime.strftime()
    old_str = "Wed Sep 14 01:46:25 GMT 2011"
    new_str = pyfstat.utils.gps_to_datestr_utc(gps)
    assert new_str.rstrip(" UTC") == old_str.replace(" GMT ", " ")


# known correct conversions from literature and lalpulsar
amp_params_test = [
    {
        "aPlus": 0.0,
        "aCross": 0.0,
        "h0": 0.0,
        "cosi": 0.0,
    },
    {
        "aPlus": 1.0,
        "aCross": 0.0,
        "h0": 2.0,
        "cosi": 0.0,
    },
    {
        "aPlus": 1.0,
        "aCross": 1.0,
        "h0": 1.0,
        "cosi": 1.0,
    },
    {
        "aPlus": 1.0,
        "aCross": -1.0,
        "h0": 1.0,
        "cosi": -1.0,
    },
]


def test_convert_h0_cosi_to_aPlus_aCross():
    for params in amp_params_test:
        print(params)
        aPlus, aCross = pyfstat.utils.convert_h0_cosi_to_aPlus_aCross(
            params["h0"], params["cosi"]
        )
        assert aPlus == pytest.approx(params["aPlus"], 1e-9)
        assert aCross == pytest.approx(params["aCross"], 1e-9)


def test_convert_aPlus_aCross_to_h0_cosi():
    for params in amp_params_test:
        h0, cosi = pyfstat.utils.convert_aPlus_aCross_to_h0_cosi(
            params["aPlus"], params["aCross"]
        )
        assert h0 == pytest.approx(params["h0"], 1e-9)
        assert cosi == pytest.approx(params["cosi"], 1e-9)


def test_convert_between_h0_cosi_and_aPlus_aCross():
    h0s = np.logspace(-26, -24, 3)
    cosis = np.linspace(-1, 1, 5)
    h0s_in = np.zeros(len(h0s) * len(cosis))
    cosis_in = np.zeros(len(h0s) * len(cosis))
    h0s_out = np.zeros(len(h0s) * len(cosis))
    cosis_out = np.zeros(len(h0s) * len(cosis))
    for n, pair in enumerate(itertools.product(h0s, cosis)):
        h0s_in[n], cosis_in[n] = pair
        # h0s_out[n], cosis_out[n] = pyfstat.utils.convert_aPlus_aCross_to_h0_cosi(*pyfstat.utils.convert_h0_cosi_to_aPlus_aCross(*pair))
    h0s_out, cosis_out = pyfstat.utils.convert_aPlus_aCross_to_h0_cosi(
        *pyfstat.utils.convert_h0_cosi_to_aPlus_aCross(h0s_in, cosis_in)
    )
    np.testing.assert_allclose(h0s_in, h0s_out, rtol=1e-9)
    np.testing.assert_allclose(cosis_in, cosis_out, rtol=1e-9)
