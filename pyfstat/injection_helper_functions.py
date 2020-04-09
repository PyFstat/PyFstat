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
    logging.warning("Python module astropy not installed")
import lal

# Assume Earth goes around Sun in a non-wobbling circle at constant speed;
# Still take the zero longitude to be the Earth's position during the March
# equinox, or March 20.
# Each day the Earth moves 2*pi/365 radians around its orbit.


def _eqToEcl(alpha, delta):
    source = SkyCoord(alpha * u.radian, delta * u.radian, frame="gcrs")
    out = source.transform_to("geocentrictrueecliptic")
    return np.array([out.lon.radian, out.lat.radian])


def _eclToEq(lon, lat):
    source = SkyCoord(lon * u.radian, lat * u.radian, frame="geocentrictrueecliptic")
    out = source.transform_to("gcrs")
    return np.array([out.ra.radian, out.dec.radian])
