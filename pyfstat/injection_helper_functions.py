""" Code used with permissision from Sylvia Zhu to calculate the range in
    frequency space that a signal occupies due to spindown and Doppler
    modulations
"""

import numpy as np
import logging
try:
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
except ImportError:
    logging.warning('Python module astropy not installed')
import lal

# Assume Earth goes around Sun in a non-wobbling circle at constant speed;
# Still take the zero longitude to be the Earth's position during the March
# equinox, or March 20.
# Each day the Earth moves 2*pi/365 radians around its orbit.


def _eqToEcl(alpha, delta):
    source = SkyCoord(alpha*u.radian, delta*u.radian, frame='gcrs')
    out = source.transform_to('geocentrictrueecliptic')
    return np.array([out.lon.radian, out.lat.radian])


def _eclToEq(lon, lat):
    source = SkyCoord(lon*u.radian, lat*u.radian,
                      frame='geocentrictrueecliptic')
    out = source.transform_to('gcrs')
    return np.array([out.ra.radian, out.dec.radian])


def _calcDopplerWings(
        s_freq, s_alpha, s_delta, lonStart, lonStop, numTimes=100):
    e_longitudes = np.linspace(lonStart, lonStop, numTimes)
    v_over_c = 2*np.pi*lal.AU_SI/lal.YRSID_SI/lal.C_SI
    s_lon, s_lat = _eqToEcl(s_alpha, s_delta)

    vertical = s_lat
    horizontals = s_lon - e_longitudes

    dopplerShifts = s_freq * np.sin(horizontals) * np.cos(vertical) * v_over_c
    F0min, F0max = np.amin(dopplerShifts), np.amax(dopplerShifts)

    # Add twice the spin-modulation
    SpinModulationMax = 2*np.pi*lal.REARTH_SI/lal.DAYSID_SI/lal.C_SI * s_freq
    return F0min - 2*SpinModulationMax, F0max + 2*SpinModulationMax


def _calcSpindownWings(freq, fdot, minStartTime, maxStartTime):
    timespan = maxStartTime - minStartTime
    return 0.5 * timespan * np.abs(fdot) * np.array([-1, 1])


def get_frequency_range_of_signal(F0, F1, Alpha, Delta, minStartTime,
                                  maxStartTime):
    """ Calculate the frequency range that a signal will occupy

    Parameters
    ----------
    F0, F1, Alpha, Delta: float
        Frequency, derivative, and sky position for the signal (all angles in
        radians)
    minStartTime, maxStartTime: float
        GPS time of the start and end of the data span

    Note: assumes tref is in the middle of the data span

    Returns
    -------
    [Fmin, Fmax]: array
        The minimum and maximum frequency span
    """
    YEAR_IN_DAYS = lal.YRSID_SI / lal.DAYSID_SI
    tEquinox = 79

    minStartTime_t = Time(minStartTime, format='gps').to_datetime().timetuple()
    maxStartTime_t = Time(maxStartTime, format='gps').to_datetime().timetuple()
    tStart_days = minStartTime_t.tm_yday - tEquinox
    tStop_days = maxStartTime_t.tm_yday - tEquinox
    tStop_days += (maxStartTime_t.tm_year-minStartTime_t.tm_year)*YEAR_IN_DAYS

    lonStart = 2*np.pi*tStart_days/YEAR_IN_DAYS - np.pi
    lonStop = 2*np.pi*tStop_days/YEAR_IN_DAYS - np.pi

    dopplerWings = _calcDopplerWings(F0, Alpha, Delta, lonStart, lonStop)
    spindownWings = _calcSpindownWings(F0, F1, minStartTime, maxStartTime)
    return np.array([F0, F0]) + dopplerWings + spindownWings
