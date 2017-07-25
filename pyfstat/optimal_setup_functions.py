"""

Provides functions to aid in calculating the optimal setup based on the metric
volume estimates.

"""

import logging
import numpy as np
import scipy.optimize
import lal
import lalpulsar


def get_optimal_setup(
        R, Nsegs0, tref, minStartTime, maxStartTime, DeltaOmega,
        DeltaFs, fiducial_freq, detector_names, earth_ephem, sun_ephem):
    logging.info('Calculating optimal setup for R={}, Nsegs0={}'.format(
        R, Nsegs0))

    V_0 = get_V_estimate(
        Nsegs0, tref, minStartTime, maxStartTime, DeltaOmega, DeltaFs,
        fiducial_freq, detector_names, earth_ephem, sun_ephem)
    logging.info('Stage {}, nsegs={}, V={}'.format(0, Nsegs0, V_0))

    nsegs_vals = [Nsegs0]
    V_vals = [V_0]

    i = 0
    nsegs_i = Nsegs0
    while nsegs_i > 1:
        nsegs_i, V_i = get_nsegs_ip1(
            nsegs_i, R, tref, minStartTime, maxStartTime, DeltaOmega,
            DeltaFs, fiducial_freq, detector_names, earth_ephem, sun_ephem)
        nsegs_vals.append(nsegs_i)
        V_vals.append(V_i)
        i += 1
        logging.info(
            'Stage {}, nsegs={}, V={}'.format(i, nsegs_i, V_i))

    return nsegs_vals, V_vals


def get_nsegs_ip1(
        nsegs_i, R, tref, minStartTime, maxStartTime, DeltaOmega,
        DeltaFs, fiducial_freq, detector_names, earth_ephem, sun_ephem):

    log10R = np.log10(R)
    log10Vi = np.log10(get_V_estimate(
        nsegs_i, tref, minStartTime, maxStartTime, DeltaOmega, DeltaFs,
        fiducial_freq, detector_names, earth_ephem, sun_ephem))

    def f(nsegs_ip1):
        if nsegs_ip1[0] > nsegs_i:
            return 1e6
        if nsegs_ip1[0] < 0:
            return 1e6
        nsegs_ip1 = int(nsegs_ip1[0])
        if nsegs_ip1 == 0:
            nsegs_ip1 = 1
        Vip1 = get_V_estimate(
            nsegs_ip1, tref, minStartTime, maxStartTime, DeltaOmega,
            DeltaFs, fiducial_freq, detector_names, earth_ephem, sun_ephem)
        if Vip1 is None:
            return 1e6
        else:
            log10Vip1 = np.log10(Vip1)
            return np.abs(log10Vi + log10R - log10Vip1)
    res = scipy.optimize.minimize(f, .5*nsegs_i, method='Powell', tol=0.1,
                                  options={'maxiter': 10})
    nsegs_ip1 = int(res.x)
    if nsegs_ip1 == 0:
        nsegs_ip1 = 1
    if res.success:
        return nsegs_ip1, get_V_estimate(
            nsegs_ip1, tref, minStartTime, maxStartTime, DeltaOmega, DeltaFs,
            fiducial_freq, detector_names, earth_ephem, sun_ephem)
    else:
        raise ValueError('Optimisation unsuccesful')


def get_V_estimate(
        nsegs, tref, minStartTime, maxStartTime, DeltaOmega, DeltaFs,
        fiducial_freq, detector_names, earth_ephem, sun_ephem):
    """ Returns V estimated from the super-sky metric

    Parameters
    ----------
    nsegs: int
        Number of semi-coherent segments
    tref: int
        Reference time in GPS seconds
    minStartTime, maxStartTime: int
        Minimum and maximum SFT timestamps
    DeltaOmega: float
        Solid angle of the sky-patch
    DeltaFs: array
        Array of [DeltaF0, DeltaF1, ...], length determines the number of
        spin-down terms.
    fiducial_freq: float
        Fidicual frequency
    detector_names: array
        Array of detectors to average over
    earth_ephem, sun_ephem: st
        Paths to the ephemeris files

    """
    spindowns = len(DeltaFs) - 1
    tboundaries = np.linspace(minStartTime, maxStartTime, nsegs+1)

    ref_time = lal.LIGOTimeGPS(tref)
    segments = lal.SegListCreate()
    for j in range(len(tboundaries)-1):
        seg = lal.SegCreate(lal.LIGOTimeGPS(tboundaries[j]),
                            lal.LIGOTimeGPS(tboundaries[j+1]),
                            j)
        lal.SegListAppend(segments, seg)
    detNames = lal.CreateStringVector(*detector_names)
    detectors = lalpulsar.MultiLALDetector()
    lalpulsar.ParseMultiLALDetector(detectors, detNames)
    detector_weights = None
    detector_motion = (lalpulsar.DETMOTION_SPIN
                       + lalpulsar.DETMOTION_ORBIT)
    ephemeris = lalpulsar.InitBarycenter(earth_ephem, sun_ephem)
    try:
        SSkyMetric = lalpulsar.ComputeSuperskyMetrics(
            spindowns, ref_time, segments, fiducial_freq, detectors,
            detector_weights, detector_motion, ephemeris)
    except RuntimeError as e:
        logging.debug('Encountered run-time error {}'.format(e))
        return None, None, None

    sqrtdetG_SKY = np.sqrt(np.linalg.det(
        SSkyMetric.semi_rssky_metric.data[:2, :2]))
    sqrtdetG_PE = np.sqrt(np.linalg.det(
        SSkyMetric.semi_rssky_metric.data[2:, 2:]))

    Vsky = .5*sqrtdetG_SKY*DeltaOmega
    Vpe = sqrtdetG_PE * np.prod(DeltaFs)
    if Vsky == 0:
        Vsky = 1
    if Vpe == 0:
        Vpe = 1
    return Vsky * Vpe
