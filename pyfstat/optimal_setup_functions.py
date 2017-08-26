"""

Provides functions to aid in calculating the optimal setup for zoom follow up

"""
from __future__ import division, absolute_import, print_function

import logging
import numpy as np
import scipy.optimize
import lal
import lalpulsar
import pyfstat.helper_functions as helper_functions


def get_optimal_setup(
        NstarMax, Nsegs0, tref, minStartTime, maxStartTime, prior,
        detector_names, earth_ephem, sun_ephem):
    """ Using an optimisation step, calculate the optimal setup ladder

    Parameters
    ----------
    NstarMax : float
    Nsegs0 : int
        The number of segments for the initial step of the ladder
    minStartTime, maxStartTime : int
        GPS times of the start and end time of the search
    prior : dict
        Prior dictionary, each item must either be a fixed scalar value, or
        a uniform prior.
    detector_names : list of str
    earth_ephem, sun_ephem : str

    Returns
    -------
    nsegs, Nstar : list
        Ladder of segment numbers and the corresponding Nstar

    """

    logging.info('Calculating optimal setup for NstarMax={}, Nsegs0={}'.format(
        NstarMax, Nsegs0))

    Nstar_0 = get_Nstar_estimate(
        Nsegs0, tref, minStartTime, maxStartTime, prior,
        detector_names, earth_ephem, sun_ephem)
    logging.info(
        'Stage {}, nsegs={}, Nstar={}'.format(0, Nsegs0, int(Nstar_0)))

    nsegs_vals = [Nsegs0]
    Nstar_vals = [Nstar_0]

    i = 0
    nsegs_i = Nsegs0
    while nsegs_i > 1:
        nsegs_i, Nstar_i = _get_nsegs_ip1(
            nsegs_i, NstarMax, tref, minStartTime, maxStartTime, prior,
            detector_names, earth_ephem, sun_ephem)
        nsegs_vals.append(nsegs_i)
        Nstar_vals.append(Nstar_i)
        i += 1
        logging.info(
            'Stage {}, nsegs={}, Nstar={}'.format(i, nsegs_i, int(Nstar_i)))

    return nsegs_vals, Nstar_vals


def _get_nsegs_ip1(nsegs_i, NstarMax, tref, minStartTime, maxStartTime, prior,
                   detector_names, earth_ephem, sun_ephem):
    """ Calculate Nsegs_{i+1} given Nsegs_{i} """

    log10NstarMax = np.log10(NstarMax)
    log10Nstari = np.log10(get_Nstar_estimate(
        nsegs_i, tref, minStartTime, maxStartTime, prior,
        detector_names, earth_ephem, sun_ephem))

    def f(nsegs_ip1):
        if nsegs_ip1[0] > nsegs_i:
            return 1e6
        if nsegs_ip1[0] < 0:
            return 1e6
        nsegs_ip1 = int(nsegs_ip1[0])
        if nsegs_ip1 == 0:
            nsegs_ip1 = 1
        Nstarip1 = get_Nstar_estimate(
            nsegs_ip1, tref, minStartTime, maxStartTime, prior,
            detector_names, earth_ephem, sun_ephem)
        if Nstarip1 is None:
            return 1e6
        else:
            log10Nstarip1 = np.log10(Nstarip1)
            return np.abs(log10Nstari + log10NstarMax - log10Nstarip1)
    res = scipy.optimize.minimize(f, .4*nsegs_i, method='Powell', tol=1,
                                  options={'maxiter': 10})
    logging.info('{} with {} evaluations'.format(res['message'], res['nfev']))
    nsegs_ip1 = int(res.x)
    if nsegs_ip1 == 0:
        nsegs_ip1 = 1
    if res.success:
        return nsegs_ip1, get_Nstar_estimate(
            nsegs_ip1, tref, minStartTime, maxStartTime, prior,
            detector_names, earth_ephem, sun_ephem)
    else:
        raise ValueError('Optimisation unsuccesful')


def _extract_data_from_prior(prior):
    """ Calculate the input data from the prior

    Parameters
    ----------
    prior: dict

    Returns
    -------
    p : ndarray
        Matrix with columns being the edges of the uniform bounding box
    spindowns : int
        The number of spindowns
    sky : bool
        If true, search includes the sky position
    fiducial_freq : float
        Fidicual frequency

    """
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
    if type(prior['F0']) == dict:
        fiducial_freq = prior['F0']['upper']
    else:
        fiducial_freq = prior['F0']

    return np.array(p).T, spindowns, sky, fiducial_freq


def get_Nstar_estimate(
        nsegs, tref, minStartTime, maxStartTime, prior,
        detector_names, earth_ephem, sun_ephem):
    """ Returns N* estimated from the super-sky metric

    Parameters
    ----------
    nsegs : int
        Number of semi-coherent segments
    tref : int
        Reference time in GPS seconds
    minStartTime, maxStartTime : int
        Minimum and maximum SFT timestamps
    prior : dict
        The prior dictionary
    detector_names : array
        Array of detectors to average over
    earth_ephem, sun_ephem : str
        Paths to the ephemeris files

    Returns
    -------
    Nstar: int
        The estimated approximate number of templates to cover the prior
        parameter space at a mismatch of unity, assuming the normalised
        thickness is unity.

    """
    in_phys, spindowns, sky, fiducial_freq = _extract_data_from_prior(prior)
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
            lalpulsar.SUPERSKY_METRIC_TYPE, spindowns, ref_time, segments,
            fiducial_freq, detectors, detector_weights, detector_motion,
            ephemeris)
    except RuntimeError as e:
        logging.warning('Encountered run-time error {}'.format(e))
        raise RuntimeError("Calculation of the SSkyMetric failed")

    if sky:
        i = 0
    else:
        i = 2

    lalpulsar.ConvertPhysicalToSuperskyPoints(
        out_rssky, in_phys, SSkyMetric.semi_rssky_transf)

    d = out_rssky.data

    g = SSkyMetric.semi_rssky_metric.data

    d[2:] = d[2:][::-1]  # Convert to Alpha, Delta, F0, F1.. ordering
    g[2:] = g[2:][::-1]  # Convert to Alpha, Delta, F0, F1.. ordering

    g = g[i:, i:]  # Remove sky if required
    parallelepiped = (out_rssky.data[i:, 1:].T - out_rssky.data[i:, 0]).T

    Nstars = []
    for j in range(1, len(g)+1):
        dV = np.abs(np.linalg.det(parallelepiped[:j, :j]))
        sqrtdetG = np.sqrt(np.abs(np.linalg.det(g[:j, :j])))
        Nstars.append(sqrtdetG * dV)
    logging.debug('Nstar for each dimension = {}'.format(
        ', '.join(["{:1.1e}".format(n) for n in Nstars])))
    return np.max(Nstars)
