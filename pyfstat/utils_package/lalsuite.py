def get_lal_exec(cmd):
    """Get a lalpulsar/lalapps executable name with the right prefix.

    This is purely to allow for backwards compatibility
    if, for whatever reason,
    someone needs to run with old releases
    (lalapps<9.0.0 and lalpulsar<5.0.0)
    from before the executables were moved.

    Parameters
    -------
    cmd: str
        Base executable name without lalapps/lalpulsar prefix.

    Returns
    -------
    full_cmd: str
        Full executable name with the right prefix.
    """
    full_cmd = shutil.which("lalpulsar_" + cmd) or shutil.which("lalapps_" + cmd)
    if full_cmd is None:
        raise RuntimeError(
            f"Could not find either lalpulsar or lalapps version of command {cmd}."
        )
    return os.path.basename(full_cmd)


def get_covering_band(
    tref,
    tstart,
    tend,
    F0,
    F1,
    F2,
    F0band=0.0,
    F1band=0.0,
    F2band=0.0,
    maxOrbitAsini=0.0,
    minOrbitPeriod=0.0,
    maxOrbitEcc=0.0,
):
    """Get the covering band for CW signals for given time and parameter ranges.

    This uses the lalpulsar function `XLALCWSignalCoveringBand()`,
    accounting for
    the spin evolution of the signals within the given [F0,F1,F2] ranges,
    the maximum possible Dopper modulation due to detector motion
    (i.e. for the worst-case sky locations),
    and for worst-case binary orbital motion.

    Parameters
    ----------
    tref: int
        Reference time (in GPS seconds) for the signal parameters.
    tstart: int
        Start time (in GPS seconds) for the signal evolution to consider.
    tend: int
        End time (in GPS seconds) for the signal evolution to consider.
    F0, F1, F1: float
        Minimum frequency and spin-down of signals to be covered.
    F0band, F1band, F1band: float
        Ranges of frequency and spin-down of signals to be covered.
    maxOrbitAsini: float
        Largest orbital projected semi-major axis to be covered.
    minOrbitPeriod: float
        Shortest orbital period to be covered.
    maxOrbitEcc: float
        Highest orbital eccentricity to be covered.

    Returns
    -------
    minCoverFreq, maxCoverFreq: float
        Estimates of the minimum and maximum frequencies of the signals
        from the given parameter ranges over the `[tstart,tend]` duration.
    """
    tref = lal.LIGOTimeGPS(tref)
    tstart = lal.LIGOTimeGPS(tstart)
    tend = lal.LIGOTimeGPS(tend)
    psr = lalpulsar.PulsarSpinRange()
    psr.fkdot[0] = F0
    psr.fkdot[1] = F1
    psr.fkdot[2] = F2
    psr.fkdotBand[0] = F0band
    psr.fkdotBand[1] = F1band
    psr.fkdotBand[2] = F2band
    psr.refTime = tref
    minCoverFreq, maxCoverFreq = lalpulsar.CWSignalCoveringBand(
        tstart, tend, psr, maxOrbitAsini, minOrbitPeriod, maxOrbitEcc
    )
    if (
        np.isnan(minCoverFreq)
        or np.isnan(maxCoverFreq)
        or minCoverFreq <= 0.0
        or maxCoverFreq <= 0.0
        or maxCoverFreq < minCoverFreq
    ):
        raise RuntimeError(
            "Got invalid pair minCoverFreq={}, maxCoverFreq={} from"
            " lalpulsar.CWSignalCoveringBand.".format(minCoverFreq, maxCoverFreq)
        )
    return minCoverFreq, maxCoverFreq


def predict_fstat(
    h0=None,
    cosi=None,
    psi=None,
    Alpha=None,
    Delta=None,
    F0=None,
    sftfilepattern=None,
    timestampsFiles=None,
    minStartTime=None,
    duration=None,
    IFOs=None,
    assumeSqrtSX=None,
    tempory_filename="fs.tmp",
    earth_ephem=None,
    sun_ephem=None,
    transientWindowType="none",
    transientStartTime=None,
    transientTau=None,
):
    """Wrapper to PredictFstat executable for predicting expected F-stat values.

    Parameters
    ----------
    h0, cosi, psi, Alpha, Delta : float
        Signal parameters, see `lalpulsar_PredictFstat --help` for more info.
    F0: float or None
        Signal frequency.
        Only needed for noise floor estimation when given `sftfilepattern`
        but `assumeSqrtSX=None`.
        The actual F-stat prediction is frequency-independent.
    sftfilepattern : str or None
        Pattern matching the SFT files to use for inferring
        detectors, timestamps and/or estimating the noise floor.
    timestampsFiles : str or None
        Comma-separated list of per-detector files containing timestamps to use.
        Only used if `sftfilepattern=None`.
    minStartTime, duration : int or None
        If `sftfilepattern` given: used as optional constraints.
        If `timestampsFiles` given: ignored.
        If neither given: used as the interval for prediction.
    IFOs : str or None
        Comma-separated list of detectors.
        Required if `sftfilepattern=None`,
        ignored otherwise.
    assumeSqrtSX : float or str
        Assume stationary per-detector noise-floor instead of estimating from SFTs.
        Single float or str value: use same for all IFOs.
        Comma-separated string: must match `len(IFOs)`
        and/or the data in `sftfilepattern`.
        Detectors will be paired to list elements following alphabetical order.
        Required if `sftfilepattern=None`,
        optional otherwise..
    tempory_filename : str
        Temporary file used for `PredictFstat` output,
        will be deleted at the end.
    earth_ephem, sun_ephem : str or None
        Ephemerides files, defaults will be used if `None`.
    transientWindowType: str
        Optional parameter for transient signals,
        see `lalpulsar_PredictFstat --help`.
        Default of `none` means a classical Continuous Wave signal.
    transientStartTime, transientTau: int or None
        Optional parameters for transient signals,
        see `lalpulsar_PredictFstat --help`.

    Returns
    -------
    twoF_expected, twoF_sigma : float
        The expectation and standard deviation of 2F.
    """

    cl_pfs = []
    cl_pfs.append(get_lal_exec("PredictFstat"))

    pars = {"h0": h0, "cosi": cosi, "psi": psi, "Alpha": Alpha, "Delta": Delta}
    cl_pfs.extend([f"--{key}={val}" for key, val in pars.items() if val is not None])

    if sftfilepattern is None:
        if IFOs is None or assumeSqrtSX is None:
            raise ValueError("Without sftfilepattern, need IFOs and assumeSqrtSX!")
        cl_pfs.append("--IFOs={}".format(IFOs))
        if timestampsFiles is None:
            if minStartTime is None or duration is None:
                raise ValueError(
                    "Without sftfilepattern, need timestampsFiles or [minStartTime,duration]!"
                )
            else:
                cl_pfs.append("--minStartTime={}".format(minStartTime))
                cl_pfs.append("--duration={}".format(duration))
        else:
            cl_pfs.append("--timestampsFiles={}".format(timestampsFiles))
    else:
        cl_pfs.append("--DataFiles='{}'".format(sftfilepattern))
        if minStartTime is not None:
            cl_pfs.append("--minStartTime={}".format(minStartTime))
            if duration is not None:
                cl_pfs.append("--maxStartTime={}".format(minStartTime + duration))
        if assumeSqrtSX is None:
            if F0 is None:
                raise ValueError(
                    "With sftfilepattern but without assumeSqrtSX,"
                    " we need F0 to estimate noise floor."
                )
            cl_pfs.append("--Freq={}".format(F0))
    if assumeSqrtSX is not None:
        if np.any([s <= 0 for s in parse_list_of_numbers(assumeSqrtSX)]):
            raise ValueError("assumeSqrtSX must be >0!")
        cl_pfs.append("--assumeSqrtSX={}".format(assumeSqrtSX))

    cl_pfs.append("--outputFstat={}".format(tempory_filename))

    earth_ephem_default, sun_ephem_default = get_ephemeris_files()
    if earth_ephem is None:
        earth_ephem = earth_ephem_default
    if sun_ephem is None:
        sun_ephem = sun_ephem_default
    cl_pfs.append("--ephemEarth='{}'".format(earth_ephem))
    cl_pfs.append("--ephemSun='{}'".format(sun_ephem))

    if transientWindowType != "none":
        cl_pfs.append("--transientWindowType='{}'".format(transientWindowType))
        cl_pfs.append("--transientStartTime='{}'".format(transientStartTime))
        cl_pfs.append("--transientTau='{}'".format(transientTau))

    cl_pfs = " ".join(cl_pfs)
    run_commandline(cl_pfs)
    d = read_par(filename=tempory_filename)
    twoF_expected = float(d["twoF_expected"])
    twoF_sigma = float(d["twoF_sigma"])
    os.remove(tempory_filename)
    return twoF_expected, twoF_sigma


def generate_loudest_file(
    max_params,
    tref,
    outdir,
    label,
    sftfilepattern,
    minStartTime=None,
    maxStartTime=None,
    transientWindowType=None,
    earth_ephem=None,
    sun_ephem=None,
):
    """Use ComputeFstatistic_v2 executable to produce a .loudest file.

    Parameters
    -------
    max_params: dict
        Dictionary of a single parameter-space point.
        This needs to already have been translated to lal conventions
        and must NOT include detection statistic entries!
    tref: int
        Reference time of the max_params.
    outdir: str
        Directory to place the .loudest file in.
    label: str
        Search name bit to be used in the output filename.
    sftfilepattern: str
        Pattern to match SFTs using wildcards (`*?`) and ranges [0-9];
        mutiple patterns can be given separated by colons.
    minStartTime, maxStartTime: int or None
        GPS seconds of the start time and end time;
        default: use al available data.
    transientWindowType: str or None
        optional: transient window type,
        needs to go with t0 and tau parameters inside max_params.
    earth_ephem: str or None
        optional: user-set Earth ephemeris file
    sun_ephem: str or None
        optional: user-set Sun ephemeris file

    Returns
    -------
    loudest_file: str
        The filename of the CFSv2 output file.
    """
    logger.info("Running CFSv2 to get .loudest file")
    if np.any([key in max_params for key in ["delta_F0", "delta_F1", "tglitch"]]):
        raise RuntimeError("CFSv2 --outputLoudest cannot deal with glitch parameters.")
    if transientWindowType:
        logger.warning(
            "CFSv2 --outputLoudest always reports the maximum of the"
            " standard CW 2F-statistic, not the transient max2F."
        )

    loudest_file = os.path.join(outdir, label + ".loudest")
    cmd = get_lal_exec("ComputeFstatistic_v2")
    CFSv2_params = {
        "DataFiles": f'"{sftfilepattern}"',
        "outputLoudest": loudest_file,
        "refTime": tref,
    }
    CFSv2_params.update(max_params)
    opt_params = {
        "minStartTime": minStartTime,
        "maxStartTime": maxStartTime,
        "transient-WindowType": transientWindowType,
        "ephemEarth": earth_ephem,
        "ephemSun": sun_ephem,
    }
    CFSv2_params.update({key: val for key, val in opt_params.items() if val})
    cmd += " " + " ".join([f"--{key}={val}" for key, val in CFSv2_params.items()])

    run_commandline(cmd, return_output=False)
    return loudest_file
