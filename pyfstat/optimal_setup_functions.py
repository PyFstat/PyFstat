"""

Provides functions to aid in calculating the optimal setup based on the metric
volume estimates.

"""
from __future__ import division, absolute_import, print_function

import logging
import numpy as np
import scipy.optimize
import lal
import lalpulsar
import pyfstat.helper_functions as helper_functions


def get_optimal_setup(
        R, Nsegs0, tref, minStartTime, maxStartTime, prior, fiducial_freq,
        detector_names, earth_ephem, sun_ephem):
    logging.info('Calculating optimal setup for R={}, Nsegs0={}'.format(
        R, Nsegs0))

    V_0 = get_V_estimate(
        Nsegs0, tref, minStartTime, maxStartTime, prior, fiducial_freq,
        detector_names, earth_ephem, sun_ephem)
    logging.info('Stage {}, nsegs={}, V={}'.format(0, Nsegs0, V_0))

    nsegs_vals = [Nsegs0]
    V_vals = [V_0]

    i = 0
    nsegs_i = Nsegs0
    while nsegs_i > 1:
        nsegs_i, V_i = get_nsegs_ip1(
            nsegs_i, R, tref, minStartTime, maxStartTime, prior, fiducial_freq,
            detector_names, earth_ephem, sun_ephem)
        nsegs_vals.append(nsegs_i)
        V_vals.append(V_i)
        i += 1
        logging.info(
            'Stage {}, nsegs={}, V={}'.format(i, nsegs_i, V_i))

    return nsegs_vals, V_vals


def get_nsegs_ip1(
        nsegs_i, R, tref, minStartTime, maxStartTime, prior, fiducial_freq,
        detector_names, earth_ephem, sun_ephem):

    log10R = np.log10(R)
    log10Vi = np.log10(get_V_estimate(
        nsegs_i, tref, minStartTime, maxStartTime, prior, fiducial_freq,
        detector_names, earth_ephem, sun_ephem))

    def f(nsegs_ip1):
        if nsegs_ip1[0] > nsegs_i:
            return 1e6
        if nsegs_ip1[0] < 0:
            return 1e6
        nsegs_ip1 = int(nsegs_ip1[0])
        if nsegs_ip1 == 0:
            nsegs_ip1 = 1
        Vip1 = get_V_estimate(
            nsegs_ip1, tref, minStartTime, maxStartTime, prior, fiducial_freq,
            detector_names, earth_ephem, sun_ephem)
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
            nsegs_ip1, tref, minStartTime, maxStartTime, prior, fiducial_freq,
            detector_names, earth_ephem, sun_ephem)
    else:
        raise ValueError('Optimisation unsuccesful')


def get_parallelepiped(prior):
    keys = ['Alpha', 'Delta', 'F0', 'F1', 'F2']
    spindown_keys = keys[3:]
    sky_keys = keys[:2]
    lims = []
    lims_keys = []
    lims_idxs = []
    for i, key in enumerate(keys):
        if type(prior[key]) == dict:
            if prior[key]['type'] == 'unif':
                lims.append([prior[key]['lower'], prior[key]['upper']])
                lims_keys.append(key)
                lims_idxs.append(i)
            else:
                raise ValueError(
                    "Prior type {} not yet supported".format(
                        prior[key]['type']))
        elif key not in spindown_keys:
            lims.append([prior[key], 0])
    lims = np.array(lims)
    lims_keys = np.array(lims_keys)
    base = lims[:, 0]
    p = [base]
    for i in lims_idxs:
        basex = base.copy()
        basex[i] = lims[i, 1]
        p.append(basex)
    spindowns = np.sum([np.sum(lims_keys == k) for k in spindown_keys])
    sky = any([key in lims_keys for key in sky_keys])
    return np.array(p).T, spindowns , sky


def get_V_estimate(
        nsegs, tref, minStartTime, maxStartTime, prior, fiducial_freq,
        detector_names, earth_ephem, sun_ephem):
    """ Returns V estimated from the super-sky metric

    Parameters
    ----------
    nsegs: int
        Number of semi-coherent segments
    tref: int
        Reference time in GPS seconds
    minStartTime, maxStartTime: int
        Minimum and maximum SFT timestamps
    prior: dict
        The prior dictionary
    fiducial_freq: float
        Fidicual frequency
    detector_names: array
        Array of detectors to average over
    earth_ephem, sun_ephem: st
        Paths to the ephemeris files

    """
    in_phys, spindowns, sky = get_parallelepiped(prior)
    out_rssky = np.zeros(in_phys.shape)

    in_phys = helper_functions.convert_array_to_gsl_matrix(in_phys)
    out_rssky = helper_functions.convert_array_to_gsl_matrix(out_rssky)

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

    if sky:
        i = 0
    else:
        i = 2

    lalpulsar.ConvertPhysicalToSuperskyPoints(
        out_rssky, in_phys, SSkyMetric.semi_rssky_transf)

    parallelepiped = (out_rssky.data[i:, 1:].T - out_rssky.data[i:, 0]).T

    sqrtdetG = np.sqrt(np.linalg.det(
        SSkyMetric.semi_rssky_metric.data[i:, i:]))

    dV = np.abs(np.linalg.det(parallelepiped))

    V = sqrtdetG * dV

    return V
