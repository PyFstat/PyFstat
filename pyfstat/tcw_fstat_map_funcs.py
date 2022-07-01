"""Additional helper functions dealing with transient-CW F(t0,tau) maps.

See Prix, Giampanis & Messenger (PRD 84, 023007, 2011):
https://arxiv.org/abs/1104.1704
for the algorithm in general
and Keitel & Ashton (CQG 35, 205003, 2018):
https://arxiv.org/abs/1805.05652
for a detailed discussion of the GPU implementation.
"""

# optional imports
import importlib as imp
import logging
import os
from time import time

import numpy as np


def _optional_import(modulename, shorthand=None):
    """
    Import a module/submodule only if it's available.

    using importlib instead of __import__
    because the latter doesn't handle sub.modules

    Also including a special check to fail more gracefully
    when CUDA_DEVICE is set to too high a number.
    """

    if shorthand is None:
        shorthand = modulename
        shorthandbit = ""
    else:
        shorthandbit = " as " + shorthand

    try:
        globals()[shorthand] = imp.import_module(modulename)
        logging.debug("Successfully imported module %s%s." % (modulename, shorthandbit))
        success = True
    except ImportError:
        logging.debug("Failed to import module {:s}.".format(modulename))
        success = False

    return success


class pyTransientFstatMap:
    """
    Simplified object class for a F(t0,tau) F-stat map.

    This is based on LALSuite's transientFstatMap_t type,
    replacing the gsl matrix with a numpy array.

    Here, `t0` is a transient start time,
    `tau` is a transient duration parameter,
    and `F(t0,tau)` is the F-statistic (not 2F)! evaluated
    for a signal with those parameters
    (and an implicit window function, which is not stored inside this object).

    The 'map' covers a range of different `(t0,tau)` pairs.

    Attributes
    ----------
    F_mn: np.ndarray
        2D array of F values (not 2F!),
        with `m` an index over start-times `t0`,
        and  `n` an index over duration parameters `tau`,
        in steps of `dt0`  in `[t0,  t0+t0Band]`,
        and `dtau` in `[tau, tau+tauBand]`.
    maxF: float
        Maximum of F (not 2F!) over the array.
    t0_ML: float
        Maximum likelihood estimate of the transient start time `t0`.
    tau_ML: float
        Maximum likelihood estimate of the transient duration `tau`.
    """

    def __init__(self, N_t0Range=None, N_tauRange=None, transientFstatMap_t=None):
        """
        The class can be initialized with the following:

        Parameters
        ----------
        N_t0Range: int
            Number of `t0` values covered.
        N_tauRange: int
            Number of `tau` values covered.
        transientFstatMap_t: lalpulsar.transientFstatMap_t
            pre-allocated matrix from lalpulsar.
        """
        if transientFstatMap_t:
            self._init_from_lalpulsar_type(transientFstatMap_t)
        else:
            if not N_t0Range or not N_tauRange:
                raise ValueError(
                    "Need either a transientFstatMap_t or a pair of (N_t0Range, N_tauRange)!."
                )
            self.F_mn = np.zeros((N_t0Range, N_tauRange), dtype=np.float32)
            # Initializing maxF to a negative value ensures
            # that we always update at least once and hence return
            # sane t0_d_ML, tau_d_ML
            # even if there is only a single bin where F=0 happens.
            self.maxF = float(-1.0)
            self.t0_ML = float(0.0)
            self.tau_ML = float(0.0)
            self.logBtSG = np.nan

    def _init_from_lalpulsar_type(self, transientFstatMap_t):
        """This essentially just strips out a redundant member level from the lalpulsar structure."""
        self.F_mn = transientFstatMap_t.F_mn.data
        self.maxF = transientFstatMap_t.maxF
        self.t0_ML = transientFstatMap_t.t0_ML
        self.tau_ML = transientFstatMap_t.tau_ML
        self.logBtSG = np.nan

    def get_maxF_idx(self):
        """Gets the 2D-unravelled index pair of the maximum of the F_mn map

        Returns
        -------
        idx: tuple
            The m,n indices of the map entry with maximal F value.
        """
        return np.unravel_index(np.argmax(self.F_mn), self.F_mn.shape)

    def get_logBtSG(self, windowRange):
        """Compute (natural log of the) transient-CW Bayes-factor B_tSG = P(x|HyptS)/P(x|HypG).

        Here HypG = Gaussian noise hypothesis, HyptS = transient-CW signal hypothesis.

        B_tSG is marginalized over start-time and timescale of transient CW signal,
        using given type and parameters of transient window range.

        NOTE: naive python port of the `lalpulsar.ComputeTransientBstat` implementation.
        Should be optimised by vectorizing the for loops!
        """
        # step through F_mn array subtract maxF and sum e^{F_mn - maxF}
        # The maximum-likelihood Fmax is globally subtracted from F_mn, and stored separatedly in the struct, because in most
        # expressions it is numerically more robust to compute e^(F_mn - Fmax), which at worst can underflow, while
        # e^F_mn can overflow (for F>~700). The constant offset e^Fmax is irrelevant for posteriors (normalization constant), or
        # can be handled separately, eg by computing log(B) = Fmax + log(sum e^(Fmn-Fmax)) for the Bayes-factor.
        N_t0Range = self.F_mn.shape[0]
        N_tauRange = self.F_mn.shape[1]
        sum_eB = 0
        for m in range(N_t0Range):
            for n in range(N_tauRange):
                DeltaF = (
                    self.maxF - self.F_mn[m, n]
                )  # always >= 0, exactly ==0 at {m,n}_max
                sum_eB += np.exp(-DeltaF)
        # combine this to final log(Bstat) result with proper normalization (assuming rhohMax=1):
        logBhat = self.maxF + np.log(sum_eB)  # unnormalized Bhat
        normBh = 70.0 / (
            N_t0Range * N_tauRange
        )  # normalization factor assuming rhohMax=1
        # final normalized Bayes factor, assuming rhohMax=1 */
        # NOTE: correct this for different rhohMax by adding "- 4 * log(rhohMax)" to logBtSG
        return np.log(normBh) + logBhat  # - 4.0 * log ( rhohMax )

    def write_F_mn_to_file(self, tCWfile, windowRange, header=[]):
        """Format a 2D transient-F-stat matrix over `(t0,tau)` and write as a text file.

        Apart from the optional extra header lines,
        the format is consistent with lalpulsar.write_transientFstatMap_to_fp().

        Parameters
        ----------
        tCWfile: str
            Name of the file to write to.
        windowRange: lalpulsar.transientWindowRange_t
            A lalpulsar structure containing the transient parameters.
        header: list
            A list of additional header lines
            to print at the start of the file.
        """
        with open(tCWfile, "w") as tfp:
            for hline in header:
                tfp.write("# {:s}\n".format(hline))
            tfp.write("# t0[s]     tau[s]     2F\n")
            for m, F_m in enumerate(self.F_mn):
                this_t0 = windowRange.t0 + m * windowRange.dt0
                for n, this_F in enumerate(F_m):
                    this_tau = windowRange.tau + n * windowRange.dtau
                    tfp.write(
                        "  %10d %10d %- 11.8g\n" % (this_t0, this_tau, 2.0 * this_F)
                    )


fstatmap_versions = {
    "lal": lambda multiFstatAtoms, windowRange, BtSG: lalpulsar_compute_transient_fstat_map(
        multiFstatAtoms, windowRange, BtSG
    ),
    "pycuda": lambda multiFstatAtoms, windowRange, BtSG: pycuda_compute_transient_fstat_map(
        multiFstatAtoms, windowRange, BtSG
    ),
}
"""Dictionary of the actual callable transient F-stat map functions this module supports.

Actual runtime availability depends on the corresponding external modules
being available.
"""


def init_transient_fstat_map_features(feature="lal", cudaDeviceName=None):
    """Initialization of available modules (or 'features') for computing transient F-stat maps.

    Currently, two implementations are supported and checked for
    through the `_optional_import()` method:

    1. `lal`: requires both `lal` and `lalpulsar` packages to be importable.

    2. `pycuda`: requires the `pycuda` package to be importable
    along with its modules
    `driver`, `gpuarray`, `tools` and `compiler`.

    Parameters
    ----------
    feature: str
        Set the transient F-stat map implementation.
    cudaDeviceName: str or None
        Request a CUDA device with this name.
        Partial matches are allowed.

    Returns
    -------
    features: dict
        A dictionary of available method names, to match `fstatmap_versions`.
        Each key's value is set to `True` only if
        all required modules are importable on this system.
    gpu_context: pycuda.driver.Context or None
        A CUDA device context object, if assigned.
    """

    features = {}

    have_lal = _optional_import("lal")
    have_lalpulsar = _optional_import("lalpulsar")
    features["lal"] = have_lal and have_lalpulsar

    # import GPU features
    have_pycuda = _optional_import("pycuda")
    have_pycuda_drv = _optional_import("pycuda.driver", "drv")
    have_pycuda_gpuarray = _optional_import("pycuda.gpuarray", "gpuarray")
    have_pycuda_tools = _optional_import("pycuda.tools", "cudatools")
    have_pycuda_compiler = _optional_import("pycuda.compiler", "cudacomp")
    features["pycuda"] = (
        have_pycuda
        and have_pycuda_drv
        and have_pycuda_gpuarray
        and have_pycuda_tools
        and have_pycuda_compiler
    )

    logging.debug("Got the following features for transient F-stat maps:")
    logging.debug(features)

    if feature == "pycuda":
        if not features["pycuda"]:
            raise RuntimeError("pycuda use was requested, but imports failed.")
        logging.debug("CUDA version: " + ".".join(map(str, drv.get_version())))

        drv.init()
        logging.debug(
            "Starting with default pyCUDA context,"
            " then checking all available devices..."
        )
        try:
            context0 = pycuda.tools.make_default_context()
        except pycuda._driver.LogicError as e:
            if e.message == "cuDeviceGet failed: invalid device ordinal":
                devn = int(os.environ["CUDA_DEVICE"])
                raise RuntimeError(
                    "Requested CUDA device number {} exceeds"
                    " number of available devices!"
                    " Please change through environment"
                    " variable $CUDA_DEVICE.".format(devn)
                )
            else:
                raise pycuda._driver.LogicError(e.message)

        num_gpus = drv.Device.count()
        logging.debug("Found {} CUDA device(s).".format(num_gpus))

        devices = []
        devnames = np.empty(num_gpus, dtype="S32")
        for n in range(num_gpus):
            devn = drv.Device(n)
            devices.append(devn)
            devnames[n] = devn.name().replace(" ", "-").replace("_", "-")
            logging.debug(
                "device {}: model: {}, RAM: {}MB".format(
                    n, devnames[n], devn.total_memory() / (2.0**20)
                )
            )

        if "CUDA_DEVICE" in os.environ:
            devnum0 = int(os.environ["CUDA_DEVICE"])
        else:
            devnum0 = 0

        matchbit = ""
        if cudaDeviceName:
            # allow partial matches in device names
            devmatches = [
                devidx
                for devidx, devname in enumerate(devnames)
                if cudaDeviceName in devname
            ]
            if len(devmatches) == 0:
                context0.detach()
                raise RuntimeError(
                    'Requested CUDA device "{}" not found.'
                    " Available devices: [{}]".format(
                        cudaDeviceName, ",".join(devnames)
                    )
                )
            else:
                devnum = devmatches[0]
                if len(devmatches) > 1:
                    logging.warning(
                        'Found {} CUDA devices matching name "{}".'
                        " Choosing first one with index {}.".format(
                            len(devmatches), cudaDeviceName, devnum
                        )
                    )
            os.environ["CUDA_DEVICE"] = str(devnum)
            matchbit = '(matched to user request "{}")'.format(cudaDeviceName)
        elif "CUDA_DEVICE" in os.environ:
            devnum = int(os.environ["CUDA_DEVICE"])
        else:
            devnum = 0
        devn = devices[devnum]
        logging.info(
            "Choosing CUDA device {},"
            " of {} devices present: {}{}...".format(
                devnum, num_gpus, devn.name(), matchbit
            )
        )
        if devnum == devnum0:
            gpu_context = context0
        else:
            context0.pop()
            gpu_context = pycuda.tools.make_default_context()
            gpu_context.push()

        _print_GPU_memory_MB("Available")
    elif feature == "lal":
        gpu_context = None
    else:
        raise ValueError(
            "Unknown transient F-stat map computation feature" f"'{feature}' requested."
        )

    return features, gpu_context


def call_compute_transient_fstat_map(
    version, features, multiFstatAtoms=None, windowRange=None, BtSG=False
):
    """Call a version of the ComputeTransientFstatMap function.

    This checks if the requested `version` is available,
    and if so, executes the computation of a transient F-statistic map
    over the `windowRange`.

    Parameters
    ----------
    version: str
        Name of the method to call
        (currently supported: 'lal' or 'pycuda').
    features: dict
        Dictionary of available features,
        as obtained from `init_transient_fstat_map_features()`.
    multiFstatAtoms: lalpulsar.MultiFstatAtomVector or None
        The time-dependent F-stat atoms previously computed by `ComputeFstat`.
    windowRange: lalpulsar.transientWindowRange_t or None
        The structure defining the transient parameters.
    BtSG: boolean
        If true, also compute the logBtSG transient Bayes factor statistic,
        using the appropriate implementation for each feature,
        and store it in `FstatMap.logBtSG`.

    Returns
    -------
    FstatMap: pyTransientFstatMap or lalpulsar.transientFstatMap_t
        The output of the called function,
        including the evaluated transient F-statistic map
        over the windowRange.
    timingFstatMap: float
        Execution time of the called function.
    """
    if version in fstatmap_versions:
        if features[version]:
            time0 = time()
            FstatMap = fstatmap_versions[version](multiFstatAtoms, windowRange, BtSG)
            timingFstatMap = time() - time0
        else:
            raise Exception(
                "Required module(s) for transient F-stat map"
                ' method "{}" not available!'.format(version)
            )
    else:
        raise Exception(
            'Transient F-stat map method "{}"' " not implemented!".format(version)
        )
    return FstatMap, timingFstatMap


def lalpulsar_compute_transient_fstat_map(multiFstatAtoms, windowRange, BtSG=False):
    """Wrapper for the standard lalpulsar function for computing a transient F-statistic map.

    See https://lscsoft.docs.ligo.org/lalsuite/lalpulsar/_transient_c_w__utils_8h.html
    for the wrapped function.

    Parameters
    ----------
    multiFstatAtoms: lalpulsar.MultiFstatAtomVector
        The time-dependent F-stat atoms previously computed by `ComputeFstat`.
    windowRange: lalpulsar.transientWindowRange_t
        The structure defining the transient parameters.
    BtSG: boolean
        If true, also compute the logBtSG transient Bayes factor statistic,
        using the corresponding lalpulsar function,
        and store it in `FstatMap.logBtSG`.

    Returns
    -------
    FstatMap: pyTransientFstatMap
        The computed results, see the class definition for details.
    """
    FstatMap_lalpulsar = lalpulsar.ComputeTransientFstatMap(
        multiFstatAtoms=multiFstatAtoms,
        windowRange=windowRange,
        useFReg=False,
    )
    FstatMap = pyTransientFstatMap(transientFstatMap_t=FstatMap_lalpulsar)
    if BtSG:
        FstatMap.logBtSG = lalpulsar.ComputeTransientBstat(
            windowRange, FstatMap_lalpulsar
        )
    return FstatMap


def reshape_FstatAtomsVector(atomsVector):
    """Make a dictionary of ndarrays out of an F-stat atoms 'vector' structure.

    Parameters
    ----------
    atomsVector: lalpulsar.FstatAtomVector
        The atoms in a 'vector'-like structure:
        iterating over timestamps as the higher hierarchical level,
        with a set of 'atoms' quantities defined at each timestamp.

    Returns
    -------
    atomsDict: dict
        A dictionary with an entry for each quantity,
        which then is a 1D ndarray over timestamps for that one quantity.
    """

    numAtoms = atomsVector.length
    atomsDict = {}
    atom_fieldnames = [
        "timestamp",
        "Fa_alpha",
        "Fb_alpha",
        "a2_alpha",
        "ab_alpha",
        "b2_alpha",
    ]
    atom_dtypes = [np.uint32, complex, complex, np.float32, np.float32, np.float32]
    for f, field in enumerate(atom_fieldnames):
        atomsDict[field] = np.ndarray(numAtoms, dtype=atom_dtypes[f])

    for n, atom in enumerate(atomsVector.data):
        for field in atom_fieldnames:
            atomsDict[field][n] = atom.__getattribute__(field)

    atomsDict["Fa_alpha_re"] = np.float32(atomsDict["Fa_alpha"].real)
    atomsDict["Fa_alpha_im"] = np.float32(atomsDict["Fa_alpha"].imag)
    atomsDict["Fb_alpha_re"] = np.float32(atomsDict["Fb_alpha"].real)
    atomsDict["Fb_alpha_im"] = np.float32(atomsDict["Fb_alpha"].imag)

    return atomsDict


def _get_absolute_kernel_path(kernel):
    pyfstatdir = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
    kernelfile = kernel + ".cu"
    return os.path.join(pyfstatdir, "pyCUDAkernels", kernelfile)


def _print_GPU_memory_MB(key):
    mem_used_MB = drv.mem_get_info()[0] / (2.0**20)
    mem_total_MB = drv.mem_get_info()[1] / (2.0**20)
    logging.debug(
        "{} GPU memory: {:.4f} / {:.4f} MB free".format(key, mem_used_MB, mem_total_MB)
    )


def pycuda_compute_transient_fstat_map(multiFstatAtoms, windowRange, BtSG=False):
    """GPU version of computing a transient F-statistic map.

    This is based on XLALComputeTransientFstatMap from LALSuite,
    (C) 2009 Reinhard Prix, licensed under GPL.

    The 'map' consists of F-statistics evaluated over
    a range of different `(t0,tau)` pairs
    (transient start-times and duration parameters).

    This is a high-level wrapper function;
    the actual CUDA computations are performed in one of the functions
    `pycuda_compute_transient_fstat_map_rect()`
    or `pycuda_compute_transient_fstat_map_exp()`,
    depending on the window functon defined in `windowRange`.

    Parameters
    ----------
    multiFstatAtoms: lalpulsar.MultiFstatAtomVector
        The time-dependent F-stat atoms previously computed by `ComputeFstat`.
    windowRange: lalpulsar.transientWindowRange_t
        The structure defining the transient parameters.
    BtSG: boolean
        If true, also compute the logBtSG transient Bayes factor statistic,
        currently using a naive unoptimised port of the corresponding lalpulsar function,
        and store it in `FstatMap.logBtSG`.

    Returns
    -------
    FstatMap: pyTransientFstatMap
        The computed results, see the class definition for details.
    """

    if windowRange.type >= lalpulsar.TRANSIENT_LAST:
        raise ValueError(
            "Unknown window-type ({}) passed as input."
            " Allowed are [0,{}].".format(
                windowRange.type, lalpulsar.TRANSIENT_LAST - 1
            )
        )

    # internal dict for search/setup parameters
    tCWparams = {}

    # first combine all multi-atoms
    # into a single atoms-vector with *unique* timestamps
    tCWparams["TAtom"] = multiFstatAtoms.data[0].TAtom
    TAtomHalf = int(tCWparams["TAtom"] / 2)  # integer division
    atoms = lalpulsar.mergeMultiFstatAtomsBinned(multiFstatAtoms, tCWparams["TAtom"])

    # make a combined input matrix of all atoms vectors, for transfer to GPU
    tCWparams["numAtoms"] = atoms.length
    atomsDict = reshape_FstatAtomsVector(atoms)
    atomsInputMatrix = np.column_stack(
        (
            atomsDict["a2_alpha"],
            atomsDict["b2_alpha"],
            atomsDict["ab_alpha"],
            atomsDict["Fa_alpha_re"],
            atomsDict["Fa_alpha_im"],
            atomsDict["Fb_alpha_re"],
            atomsDict["Fb_alpha_im"],
        )
    )

    # actual data spans [t0_data, t0_data + tCWparams['numAtoms'] * TAtom]
    # in steps of TAtom
    tCWparams["t0_data"] = int(atoms.data[0].timestamp)
    tCWparams["t1_data"] = int(
        atoms.data[tCWparams["numAtoms"] - 1].timestamp + tCWparams["TAtom"]
    )

    logging.debug(
        "Transient F-stat map:"
        " t0_data={:d}, t1_data={:d}".format(tCWparams["t0_data"], tCWparams["t1_data"])
    )
    logging.debug(
        "Transient F-stat map:"
        " numAtoms={:d}, TAtom={:d},"
        " TAtomHalf={:d}".format(tCWparams["numAtoms"], tCWparams["TAtom"], TAtomHalf)
    )

    # special treatment of window_type = none
    # ==> replace by rectangular window spanning all the data
    if windowRange.type == lalpulsar.TRANSIENT_NONE:
        windowRange.type = lalpulsar.TRANSIENT_RECTANGULAR
        windowRange.t0 = tCWparams["t0_data"]
        windowRange.t0Band = 0
        windowRange.dt0 = tCWparams["TAtom"]  # irrelevant
        windowRange.tau = tCWparams["numAtoms"] * tCWparams["TAtom"]
        windowRange.tauBand = 0
        windowRange.dtau = tCWparams["TAtom"]  # irrelevant

    """ NOTE: indices {i,j} enumerate *actual* atoms and their timestamps t_i,
    * while the indices {m,n} enumerate the full grid of values
    * in [t0_min, t0_max]x[Tcoh_min, Tcoh_max] in steps of deltaT.
    * This allows us to deal with gaps in the data in a transparent way.
    *
    * NOTE2: we operate on the 'binned' atoms returned
    * from XLALmergeMultiFstatAtomsBinned(),
    * which means we can safely assume all atoms to be lined up
    * perfectly on a 'deltaT' binned grid.
    *
    * The mapping used will therefore be {i,j} -> {m,n}:
    *   m = offs_i  / deltaT
    *   start-time offset from t0_min measured in deltaT
    *   n = Tcoh_ij / deltaT
    *   duration Tcoh_ij measured in deltaT,
    *
    * where
    *   offs_i  = t_i - t0_min
    *   Tcoh_ij = t_j - t_i + deltaT
    *
    """

    # We allocate a matrix  {m x n} = t0Range * TcohRange elements
    # covering the full transient window-range [t0,t0+t0Band]x[tau,tau+tauBand]
    tCWparams["N_t0Range"] = int(
        np.floor(1.0 * windowRange.t0Band / windowRange.dt0) + 1
    )
    tCWparams["N_tauRange"] = int(
        np.floor(1.0 * windowRange.tauBand / windowRange.dtau) + 1
    )
    FstatMap = pyTransientFstatMap(tCWparams["N_t0Range"], tCWparams["N_tauRange"])

    logging.debug(
        "Transient F-stat map:"
        " N_t0Range={:d}, N_tauRange={:d},"
        " total grid points: {:d}".format(
            tCWparams["N_t0Range"],
            tCWparams["N_tauRange"],
            tCWparams["N_t0Range"] * tCWparams["N_tauRange"],
        )
    )

    if windowRange.type == lalpulsar.TRANSIENT_RECTANGULAR:
        FstatMap.F_mn = pycuda_compute_transient_fstat_map_rect(
            atomsInputMatrix, windowRange, tCWparams
        )
    elif windowRange.type == lalpulsar.TRANSIENT_EXPONENTIAL:
        FstatMap.F_mn = pycuda_compute_transient_fstat_map_exp(
            atomsInputMatrix, windowRange, tCWparams
        )
    else:
        raise ValueError(
            "Invalid transient window type {}"
            " not in [{}, {}].".format(
                windowRange.type, lalpulsar.TRANSIENT_NONE, lalpulsar.TRANSIENT_LAST - 1
            )
        )

    # out of loop: get max2F and ML estimates over the m x n matrix
    FstatMap.maxF = FstatMap.F_mn.max()
    maxidx = np.unravel_index(
        FstatMap.F_mn.argmax(), (tCWparams["N_t0Range"], tCWparams["N_tauRange"])
    )
    FstatMap.t0_ML = windowRange.t0 + maxidx[0] * windowRange.dt0
    FstatMap.tau_ML = windowRange.tau + maxidx[1] * windowRange.dtau

    logging.debug(
        "Done computing transient F-stat map."
        " maxF={:.4f}, t0_ML={}, tau_ML={}".format(
            FstatMap.maxF, FstatMap.t0_ML, FstatMap.tau_ML
        )
    )

    if BtSG:
        # FIXME this should be moved into cuda kernel!
        FstatMap.logBtSG = FstatMap.get_logBtSG(windowRange)
        logging.debug(f"Also computed: logBtSG={FstatMap.logBtSG:.4f}")

    return FstatMap


def pycuda_compute_transient_fstat_map_rect(atomsInputMatrix, windowRange, tCWparams):
    """GPU computation of the transient F-stat map for rectangular windows.

    As discussed in Keitel & Ashton (CQG 35, 205003, 2018):
    https://arxiv.org/abs/1805.05652
    this version only does GPU parallization for the outer loop,
    keeping the partial sums of the inner loop local to each individual kernel
    using the 'memory trick'.

    Parameters
    ----------
    atomsInputMatrix: np.ndarray
        A 2D array of stacked named columns containing the F-stat atoms.
    windowRange: lalpulsar.transientWindowRange_t
        The structure defining the transient parameters.
    tCWparams: dict
        A dictionary of miscellaneous parameters.

    Returns
    -------
    F_mn: np.ndarray
        A 2D array of the computed transient F-stat map over the
        `[t0,tau]` range.
    """

    # gpu data setup and transfer
    _print_GPU_memory_MB("Initial")
    input_gpu = gpuarray.to_gpu(atomsInputMatrix)
    Fmn_gpu = gpuarray.GPUArray(
        (tCWparams["N_t0Range"], tCWparams["N_tauRange"]), dtype=np.float32
    )
    _print_GPU_memory_MB("After input+output allocation:")

    # GPU kernel
    kernel = "cudaTransientFstatRectWindow"
    kernelfile = _get_absolute_kernel_path(kernel)
    partial_Fstat_cuda_code = cudacomp.SourceModule(open(kernelfile, "r").read())
    partial_Fstat_cuda = partial_Fstat_cuda_code.get_function(kernel)
    partial_Fstat_cuda.prepare("PIIIIIIIIP")

    # GPU grid setup
    blockRows = min(1024, tCWparams["N_t0Range"])
    blockCols = 1
    gridRows = int(np.ceil(1.0 * tCWparams["N_t0Range"] / blockRows))
    gridCols = 1

    # running the kernel
    logging.debug(
        "Calling pyCUDA kernel with a grid of {}*{}={} blocks"
        " of {}*{}={} threads each: {} total threads...".format(
            gridRows,
            gridCols,
            gridRows * gridCols,
            blockRows,
            blockCols,
            blockRows * blockCols,
            gridRows * gridCols * blockRows * blockCols,
        )
    )
    partial_Fstat_cuda.prepared_call(
        (gridRows, gridCols),
        (blockRows, blockCols, 1),
        input_gpu.gpudata,
        tCWparams["numAtoms"],
        tCWparams["TAtom"],
        tCWparams["t0_data"],
        windowRange.t0,
        windowRange.dt0,
        windowRange.tau,
        windowRange.dtau,
        tCWparams["N_tauRange"],
        Fmn_gpu.gpudata,
    )

    # return results to host
    F_mn = Fmn_gpu.get()

    _print_GPU_memory_MB("Final")

    return F_mn


def pycuda_compute_transient_fstat_map_exp(atomsInputMatrix, windowRange, tCWparams):
    """GPU computation of the transient F-stat map for exponential windows.

    As discussed in Keitel & Ashton (CQG 35, 205003, 2018):
    https://arxiv.org/abs/1805.05652
    this version does full GPU parallization
    of both the inner and outer loop.

    Parameters
    ----------
    atomsInputMatrix: np.ndarray
        A 2D array of stacked named columns containing the F-stat atoms.
    windowRange: lalpulsar.transientWindowRange_t
        The structure defining the transient parameters.
    tCWparams: dict
        A dictionary of miscellaneous parameters.

    Returns
    -------
    F_mn: np.ndarray
        A 2D array of the computed transient F-stat map over the
        `[t0,tau]` range.
    """

    # gpu data setup and transfer
    _print_GPU_memory_MB("Initial")
    input_gpu = gpuarray.to_gpu(atomsInputMatrix)
    Fmn_gpu = gpuarray.GPUArray(
        (tCWparams["N_t0Range"], tCWparams["N_tauRange"]), dtype=np.float32
    )
    _print_GPU_memory_MB("After input+output allocation:")

    # GPU kernel
    kernel = "cudaTransientFstatExpWindow"
    kernelfile = _get_absolute_kernel_path(kernel)
    partial_Fstat_cuda_code = cudacomp.SourceModule(open(kernelfile, "r").read())
    partial_Fstat_cuda = partial_Fstat_cuda_code.get_function(kernel)
    partial_Fstat_cuda.prepare("PIIIIIIIIIP")

    # GPU grid setup
    blockRows = min(32, tCWparams["N_t0Range"])
    blockCols = min(32, tCWparams["N_tauRange"])
    gridRows = int(np.ceil(1.0 * tCWparams["N_t0Range"] / blockRows))
    gridCols = int(np.ceil(1.0 * tCWparams["N_tauRange"] / blockCols))

    # running the kernel
    logging.debug(
        "Calling kernel with a grid of {}*{}={} blocks"
        " of {}*{}={} threads each: {} total threads...".format(
            gridRows,
            gridCols,
            gridRows * gridCols,
            blockRows,
            blockCols,
            blockRows * blockCols,
            gridRows * gridCols * blockRows * blockCols,
        )
    )
    partial_Fstat_cuda.prepared_call(
        (gridRows, gridCols),
        (blockRows, blockCols, 1),
        input_gpu.gpudata,
        tCWparams["numAtoms"],
        tCWparams["TAtom"],
        tCWparams["t0_data"],
        windowRange.t0,
        windowRange.dt0,
        windowRange.tau,
        windowRange.dtau,
        tCWparams["N_t0Range"],
        tCWparams["N_tauRange"],
        Fmn_gpu.gpudata,
    )

    # return results to host
    F_mn = Fmn_gpu.get()

    _print_GPU_memory_MB("Final")

    return F_mn
