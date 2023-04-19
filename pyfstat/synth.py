import contextlib
import logging
import os
from numbers import Number
from typing import Sequence, Union

import lal
import lalpulsar
import numpy as np

import pyfstat.utils as utils
from pyfstat import BaseSearchClass, DetectorStates, InjectionParametersGenerator
from pyfstat.tcw_fstat_map_funcs import reshape_FstatAtomsVector

logger = logging.getLogger(__name__)


class Synthesizer(BaseSearchClass):
    """Efficiently generate lots of detection statistics samples under Gaussian noise.

    * Generate samples of detection statistics, e.g. the `F`-statistic,
      from random draws of the underlying distributions, assuming Gaussian noise,
      and with signal parameters drawn from their (given) priors.
    * Can also return signal parameters, `F`-stat atoms and transient `F`-stat maps.
    * Python port of
      `lalpulsar_synthesizeTransientStats <https://lscsoft.docs.ligo.org/lalsuite/lalpulsar/synthesize_transient_stats_8c.html>`_
      and its siblings.
    * See appendix of PGM2011 [ https://arxiv.org/abs/1104.1704 ] for
      the underlying algorithm.
    * ``lalpulsar.SynthesizeTransientAtoms`` has its own internal random draws
      capabilities for sky positions and amplitude parameters,
      but instead we use our own ``InjectionParametersGenerator` class
      and then loop the lalpulsar function over single sets of parameters.
      This allows for more flexible distributions
      (basically anything supported by scipy).
    """

    @utils.initializer
    def __init__(
        self,
        label: str,
        outdir: str,
        priors: dict,
        tstart: int = None,
        duration: int = None,
        Tsft: int = 1800,
        detectors: str = None,
        earth_ephem: str = None,
        sun_ephem: str = None,
        transientWindowType: str = None,
        transientStartTime: int = None,
        transientTau: int = None,
        randSeed: int = 0,
        timestamps: Union[str, dict] = None,
        signalOnly: bool = False,
        detstats: Sequence[Union[str, dict]] = None,
    ):
        """
        Parameters
        ----------
        label:
            A human-readable label to be used in naming the output files.
        outdir:
            The directory where files are written to.
            Default: current working directory.
        tstart:
            Starting GPS epoch of the data set.
            Not yet implemented! Please use `timestamps` instead.
            NOTE: will be mutually exclusive with `timestamps`.
        duration:
            Duration (in GPS seconds) of the total data set.
            Not yet implemented! Please use `timestamps` instead.
            NOTE: will be mutually exclusive with `timestamps`.
        priors:
            Dictionary of priors for parameters [Alpha,Delta,h0,cosi,psi,phi0],
            to be parsed by
            :func:`~pyfstat.injection_parameters.InjectionParametersGenerator`.
            In the simplest case, for fixed values, a minimal example would be
            ::

            priors = {"Alpha": 0, "Delta": 0, "h0": 0, "cosi": 0, "psi": 0, "phi": 0}

        Tsft:
            The SFT duration in seconds.
        detectors:
            Comma-separated list of detectors to generate data for.
            Can be omitted if `timestamps` is a dictionary.
        earth_ephem, sun_ephem:
            Paths of the two files containing positions of Earth and Sun.
            If None, will check standard sources as per
            utils.get_ephemeris_files().
        transientWindowType:
            If `None`, a fully persistent CW signal is simulated.
            If `"rect"` or `"exp"`, a transient signal with the corresponding
            amplitude evolution is simulated.
        transientStartTime:
            Start time for a transient signal.
        transientTau:
            Duration (`rect` case) or decay time (`exp` case) of a transient signal.
        randSeed:
            Optionally fix the random seed of Gaussian noise generation
            for reproducibility. Default of `0` means no fixed seed.
            The same seed will be passed both to the ``InjectionParametersGenerator``
            and to lalpulsar for the F-stat atoms generation.
        timestamps:
            Dictionary of timestamps (each key must refer to a detector),
            list of timestamps (`detectors` should be set),
            or comma-separated list of per-detector timestamps files
            (simple text files, lines with <seconds> <nanoseconds>,
            comments should use `%`;
            order of files must match that of `detectors`)
            Each time stamp gives the start time of one SFT.
            NOTE: mutually exclusive with [`tstart`,`duration`].
        signalOnly:
            Generate pure signal without noise?
        detstats:
            Detection statistics to compute.
            See :func:`pyfstat.utils.detstats.parse_detstats`
            for the supported format.
            For details of supported `BSGL` parameters,
            see :func:`pyfstat.utils.detstats.get_BSGL_setup`.
        """
        self._set_init_params_dict(locals())
        if self.duration is not None or self.tstart is not None:
            raise NotImplementedError(
                "Options 'duration' and 'tstart' are not implemented yet."
                " Please use timestamps instead."
            )
        self.rng = lal.gsl_rng("mt19937", self.randSeed)
        dets = DetectorStates()
        self.multiDetStates = dets.get_multi_detector_states(
            self.timestamps,
            self.Tsft,
            self.detectors,
        )
        self.numDetectors = self.multiDetStates.length
        # in case of fixed sky position, use buffering for efficiency
        self.multiAMBuffer = lalpulsar.multiAMBuffer_t()
        self.skypos = lal.SkyPosition()
        self.skypos.system = lal.COORDINATESYSTEM_EQUATORIAL
        if not isinstance(priors, dict):
            raise ValueError("priors argument must be a dictionary.")
        if isinstance(priors["Alpha"], Number) and isinstance(priors["Delta"], Number):
            self.skypos.longitude = priors["Alpha"]
            self.skypos.latitude = priors["Delta"]
        self.paramsGen = InjectionParametersGenerator(
            priors=self.priors, seed=self.randSeed
        )
        self.transientInjectRange = lalpulsar.transientWindowRange_t()
        if transientWindowType is None:
            self.transientInjectRange.type = lalpulsar.ParseTransientWindowName("none")
        else:
            raise NotImplementedError("Transients are not yet implemented.")
            # FIXME: handle transient injection and search ranges separately
            # self.injectWindow_type = self.transientWindowType
            # self.searchWindow_type = self.transientWindowType
            # self.injectWindow_t0 = self.transientStartTime
            # self.injectWindow_tau = self.transientTau
            # self.injectWindow_t0Band = 0
            # self.injectWindow_tauBand = 0
            # self.transientInjectRange.type = lalpulsar.ParseTransientWindowName(
            # self.injectWindow_type
            # )
            # self.transientInjectRange.t0 = self.tstart + self.injectWindow_t0
            # tauMax = self.injectWindow_tau + self.injectWindow_tau
            # self.transientInjectRange.t0Band = self.injectWindow_t0Band
            # self.transientInjectRange.tau = self.injectWindow_tau
            # self.transientInjectRange.tauBand = self.injectWindow_tauBand
        self.detstats, detstat_params = utils.parse_detstats(self.detstats)
        BSGL = utils.get_canonical_detstat_name("BSGL")
        if BSGL in self.detstats:
            self.BSGLSetup = utils.get_BSGL_setup(
                numDetectors=self.numDetectors,
                numSegments=1,
                **detstat_params[BSGL],
            )
        self.output_file_header = self.get_output_file_header()
        logger.debug(
            f"Creating output directory {self.outdir} if it does not yet exist..."
        )
        os.makedirs(self.outdir, exist_ok=True)

    def _set_amplitude_prior(self, injParams):
        ampPrior = lalpulsar.AmplitudePrior_t()
        if "snr" in injParams.keys():
            ampPrior.fixedSNR = injParams["snr"]
            ampPrior.pdf_h0Nat = lalpulsar.CreateSingularPDF1D(1.0)
        elif "h0" in injParams.keys():
            ampPrior.fixedSNR = -1
            ampPrior.pdf_h0Nat = lalpulsar.CreateSingularPDF1D(injParams["h0"])
        else:
            raise ValueError("Need either 'snr' or 'h0' in injParams!")
        ampPrior.pdf_cosi = lalpulsar.CreateSingularPDF1D(injParams["cosi"])
        ampPrior.pdf_psi = lalpulsar.CreateSingularPDF1D(injParams["psi"])
        ampPrior.pdf_phi0 = lalpulsar.CreateSingularPDF1D(injParams["phi"])
        ampPrior.fixRhohMax = False  # we don't support this
        return ampPrior

    def synth_candidates(
        self,
        numDraws: int = 1,
        returns: list = ["detstats"],
        hdf5_outputs: list = [""],
        txt_outputs: list = [""],
    ):
        """Draw a batch of signal parameters and corresponding statistics.

        Parameters
        ----------
        numDraws:
            How many candidates to draw.
        returns:
            Results to put into the return dictionary:
            "detstats", "parameters", "FstatMaps", "atoms".
        hdf5_outputs:
            Results to put into an .h5 output file
            (one file for all draws together):
            "detstats", "parameters", "FstatMaps", "atoms".
        txt_outputs:
            Results to put into plain-text .dat file(s):
            "detstats" (one file for all draws together),
            "parameters" (one file for all draws together),
            "FstatMaps" (one file per draw),
            "atoms" (one file per draw).

        Returns
        -------
        candidates: dict
            A dictionary with, at a minimum, one entry for each detection statistic
            requested via the instance's `detstats` argument,
            and optional entries for [params, FstatMaps, atoms]
            depending on the `returns` argument.
            Each entry is an array/list over draws.
        """
        for q in ["detstats", "parameters", "FstatMaps"]:
            if q in "txt_outputs":
                raise NotImplementedError(
                    f"text file output of {q}  not yet implemented."
                )
        if "FstatMaps" in hdf5_outputs:
            raise NotImplementedError("hdf5 output of FstatMaps not yet implemented.")
        if len(hdf5_outputs) > 0:
            try:
                import h5py
            except ImportError:
                raise ImportError(
                    "Could not import 'h5py', please install it to use this method. "
                    "For example: `pip install pyfstat[hdf5]`"
                )
            h5file = os.path.join(self.outdir, self.label + "_draws.h5")
            logger.info(f"Will save output from all draws to hdf5 file: {h5file}")
        candidates = {}
        if "detstats" in returns:
            for stat in self.detstats:
                candidates[stat] = (
                    np.tile(np.nan, (numDraws, self.numDetectors))
                    if stat == "twoFX"
                    else np.repeat(np.nan, numDraws)
                )
        if "FstatMaps" in returns:
            candidates["FstatMaps"] = []
        if "atoms" in returns:
            candidates["atoms"] = []
        logger.info(f"Drawing {numDraws} results.")
        with (
            h5py.File(h5file, "w", locking=False)
            if len(hdf5_outputs) > 0
            else contextlib.nullcontext()
        ) as h5:
            for n in range(numDraws):
                (
                    detStats,
                    paramsDrawn,
                    FstatMap,
                    multiFatoms,
                ) = self.synth_one_candidate()
                for key, val in detStats.items():
                    if key == "twoFX":
                        candidates[key][n, :] = val[: self.numDetectors]
                    else:
                        candidates[key][n] = val
                if "parameters" in returns:
                    if n == 0:
                        for key in paramsDrawn.keys():
                            candidates[key] = np.repeat(np.nan, numDraws)
                    for key, val in paramsDrawn.items():
                        candidates[key][n] = val
                if "FstatMaps" in returns:
                    candidates["FstatMaps"].append(FstatMap)
                if "atoms" in returns:
                    candidates["atoms"].append(multiFatoms)
                if "atoms" in txt_outputs:
                    utils.write_atoms_to_txt_file(
                        fname=os.path.join(
                            self.outdir,
                            f"{self.label}_Fstatatoms_draw{n}_of_{numDraws}.dat",
                        ),
                        atoms=multiFatoms,
                        header=self.output_file_header,
                    )
                if "params" in hdf5_outputs:
                    if n == 0:
                        h5_group_params = h5.create_group("parameters")
                        for par in paramsDrawn.keys():
                            h5_group_params.create_dataset(
                                par,
                                shape=(numDraws),
                            )
                    for par in paramsDrawn.keys():
                        h5_group_params[par][n] = paramsDrawn[par]
                # if "FstatMaps" in hdf5_outputs:
                # if n == 0:
                # h5_dset_FstatMaps = h5.create_dataset(
                # "FstatMaps",
                # shape=(numDraws),
                # )
                # FIXME: need to convert to a plain array
                # h5_dset_FstatMaps[n] = FstatMap
                if "atoms" in hdf5_outputs:
                    if n == 0:
                        h5_dset_atoms = h5.create_dataset(
                            "atoms",
                            shape=(numDraws, multiFatoms.data[0].length, 8),
                        )
                    mergedAtoms = lalpulsar.mergeMultiFstatAtomsBinned(
                        multiFatoms, self.Tsft
                    )
                    h5_dset_atoms[n] = self._reshape_FstatAtomVector_to_array(
                        mergedAtoms
                    )
        return candidates

    def synth_one_candidate(self):
        injParams = self.paramsGen.draw()
        ampPrior = self._set_amplitude_prior(injParams)
        injParamsLalpulsar = (
            lalpulsar.InjParams_t()
        )  # output struct, will be filled by lalpulsar call
        if injParams["Alpha"] != self.skypos.longitude:
            self.skypos.longitude = injParams["Alpha"]
        if injParams["Delta"] != self.skypos.latitude:
            self.skypos.latitude = injParams["Delta"]
        multiAtoms = lalpulsar.SynthesizeTransientAtoms(
            injParamsOut=injParamsLalpulsar,
            skypos=self.skypos,
            AmpPrior=ampPrior,
            transientInjectRange=self.transientInjectRange,
            multiDetStates=self.multiDetStates,
            SignalOnly=self.signalOnly,
            multiAMBuffer=self.multiAMBuffer,
            rng=self.rng,
            lineX=-1,
            multiNoiseWeights=None,
        )
        injParamsDict = self._InjParams_t_to_dict(injParamsLalpulsar)
        # FIXME support pycuda version of ComputeTransientFstatMap
        detStats = {}
        BSGL = utils.get_canonical_detstat_name("BSGL")
        BtSG = utils.get_canonical_detstat_name("BtSG")
        if "twoF" in self.detstats or BSGL in self.detstats:
            detStats["twoF"] = lalpulsar.ComputeFstatFromAtoms(multiAtoms, -1)
        if BSGL in self.detstats:
            detStats["twoFX"] = np.zeros(lalpulsar.PULSAR_MAX_DETECTORS)
            detStats["twoFX"][: self.numDetectors] = [
                lalpulsar.ComputeFstatFromAtoms(multiAtoms, X)
                for X in range(self.numDetectors)
            ]
            detStats[BSGL] = lalpulsar.ComputeBSGL(
                detStats["twoF"], detStats["twoFX"], self.BSGLSetup
            )
        if "maxTwoF" in self.detstats or BtSG in self.detstats:
            transientWindowRange = self.transientInjectRange  # FIXME
            FstatMap = lalpulsar.ComputeTransientFstatMap(
                multiAtoms, transientWindowRange, useFReg=False
            )
            if self.signalOnly:
                FstatMap.maxF += 2
        else:
            FstatMap = None
        if "maxTwoF" in self.detstats:
            detStats["maxTwoF"] = 2 * FstatMap.maxF
        if BtSG in self.detstats:
            detStats[BtSG] = lalpulsar.ComputeTransientBstat(
                transientWindowRange, FstatMap
            )
            if self.signalOnly:
                detStats[BtSG] += 2
        for stat in self.detstats:
            if stat not in detStats.keys():
                raise RuntimeError(f"Sorry, we somehow forgot to compute `{stat}`!")
        return detStats, injParamsDict, FstatMap, multiAtoms

    def _InjParams_t_to_dict(self, paramsStruct):
        """
        Convert a lalpulsar InjParams_t object into a flat dictionary

        Currently not converted:
          * ampVect
          * multiAM
          * transientWindow
          * detM1o8
        """
        paramsDict = {
            "Alpha": paramsStruct.skypos.longitude,
            "Delta": paramsStruct.skypos.latitude,
            "aPlus": paramsStruct.ampParams.aPlus,
            "aCross": paramsStruct.ampParams.aCross,
            "phi": paramsStruct.ampParams.phi0,
            "psi": paramsStruct.ampParams.psi,
            "snr": paramsStruct.SNR,
        }
        paramsDict["h0"], paramsDict["cosi"] = utils.convert_aPlus_aCross_to_h0_cosi(
            aPlus=paramsDict["aPlus"], aCross=paramsDict["aCross"]
        )
        return paramsDict

    def _reshape_FstatAtomVector_to_array(self, atomsVector):
        """Reshape a FstatAtomVector into an (Ntimestamps,8) np.ndarray.

        Parameters
        ----------
        atomsVector: lalpulsar.FstatAtomVector
            The atoms in a 'vector'-like structure:
            iterating over timestamps as the higher hierarchical level,
            with a set of 'atoms' quantities defined at each timestamp.

        Returns
        -------
        atoms_for_h5: np.ndarray
            Array of the atoms, with shape (Ntimestamps,8).
        """
        atomsDict = reshape_FstatAtomsVector(atomsVector)
        atomsArray = np.array(list(atomsDict.values())).transpose()
        return atomsArray
