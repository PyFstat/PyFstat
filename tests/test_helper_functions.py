import pyfstat


def test_gps_to_datestr_utc():

    gps = 1000000000
    # reference from lalapps_tconvert on en_US.UTF-8 locale
    # but instead from the "GMT" bit it places in the middle,
    # we put the timezone info at the end via datetime.strftime()
    old_str = "Wed Sep 14 01:46:25 GMT 2011"
    new_str = pyfstat.helper_functions.gps_to_datestr_utc(gps)
    assert new_str.rstrip(" UTC") == old_str.replace(" GMT ", " ")
