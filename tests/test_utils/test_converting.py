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
    h0s_in = np.logspace(-26, -24, 3)
    cosis_in = np.linspace(-1, 1, 5)
    h0s_out, cosis_out = pyfstat.utils.convert_aPlus_aCross_to_h0_cosi(
        *pyfstat.utils.convert_h0_cosi_to_aPlus_aCross(
            h0s_in[:, None], cosis_in[None, :]
        )
    )
    for h0col in h0s_out.transpose():
        np.testing.assert_allclose(h0col, h0s_in, rtol=1e-9)
    for cosirow in cosis_out:
        np.testing.assert_allclose(cosirow, cosis_in, rtol=1e-9)
