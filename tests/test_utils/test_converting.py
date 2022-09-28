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


def test_convert_aCross_aPlus_to_h0_cosi():
    h0, cosi = pyfstat.utils.convert_aCross_aPlus_to_h0_cosi(1.0, 0.0)
    assert h0 == pytest.approx(2.0, 1e-9)
    assert cosi == pytest.approx(0.0, 1e-9)
    h0, cosi = pyfstat.utils.convert_aCross_aPlus_to_h0_cosi(1.0, 1.0)
    assert h0 == pytest.approx(1.0, 1e-9)
    assert cosi == pytest.approx(1.0, 1e-9)
    h0, cosi = pyfstat.utils.convert_aCross_aPlus_to_h0_cosi(-1.0, 0.0)
    assert h0 == pytest.approx(0.0, 1e-9)
    assert cosi == pytest.approx(0.0, 1e-9)
