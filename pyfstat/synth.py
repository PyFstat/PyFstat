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
        if "twoFX" in self.detstats or BSGL in self.detstats:
            for IFO in self.detectors.split(","):
                self.detstats.append("twoF" + IFO)
        if "twoFX" in self.detstats:
            self.detstats.remove("twoFX")
        self.output_file_header = self.get_output_file_header()
        self.param_keys = [
            "Alpha",
            "Delta",
            "aPlus",
            "aCross",
            "h0",
            "cosi",
            "psi",
            "phi",
            "snr",
        ]
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
        returns: Sequence[str] = ("detstats", "parameters"),
        hdf5_outputs: Sequence[str] = (),
        txt_outputs: Sequence[str] = (),
        **hdf5_kwargs,
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
        hdf5_kwargs:
            Dictionary of extra arguments for hdf5 output.
            "chunk_size" will be used locally for efficient writing
            (default: 1000).
            All other kwargs (e.g. compression settings)
            will be passed on to h5py,
            see https://docs.h5py.org/en/stable/high/dataset.html

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
        if "parameters" in returns:
            for key in self.param_keys:
                # FIXME: add transient parameters
                candidates[key] = np.repeat(np.nan, numDraws)
        if "detstats" in returns:
            for stat in self.detstats:
                candidates[stat] = np.repeat(np.nan, numDraws)
        if "FstatMaps" in returns:
            candidates["FstatMaps"] = [None for n in range(numDraws)]
        if "atoms" in returns:
            candidates["atoms"] = [None for n in range(numDraws)]
        chunk_size_hdf5 = (
            hdf5_kwargs.pop("chunk_size") if "chunk_size" in hdf5_kwargs else 1000
        )
        params_for_hdf5 = (
            np.zeros((chunk_size_hdf5, len(self.param_keys)))
            if "parameters" in hdf5_outputs
            else None
        )
        detstats_for_hdf5 = (
            np.zeros((chunk_size_hdf5, len(self.detstats)))
            if "detstats" in hdf5_outputs
            else None
        )
        # FstatMaps_for_hdf5 = (
        # [None for n in range(chunk_size_hdf5)]
        # if "FstatMaps" in hdf5_outputs
        # else None
        # )
        atoms_for_hdf5 = (
            [None for n in range(chunk_size_hdf5)] if "atoms" in hdf5_outputs else None
        )
        logger.info(f"Drawing {numDraws} sets of signal parameters.")
        injParams = self.paramsGen.draw_many(numDraws)
        logger.info(f"Synthesizing {numDraws} results.")
        with (
            h5py.File(h5file, "w", locking=False)
            if len(hdf5_outputs) > 0
            else contextlib.nullcontext()
        ) as h5f:
            for n in range(numDraws):
                injParams_n = {key: val[n] for key, val in injParams.items()}
                (
                    paramsDrawn,
                    detStats,
                    FstatMap,
                    multiFatoms,
                ) = self.synth_one_candidate(injParams_n)
                if "parameters" in returns:
                    for key, val in paramsDrawn.items():
                        candidates[key][n] = val
                for key, val in detStats.items():
                    candidates[key][n] = val
                if "FstatMaps" in returns:
                    candidates["FstatMaps"][n] = FstatMap
                if "atoms" in returns:
                    candidates["atoms"][n] = multiFatoms
                if "atoms" in txt_outputs:
                    utils.write_atoms_to_txt_file(
                        fname=os.path.join(
                            self.outdir,
                            f"{self.label}_Fstatatoms_draw{n}_of_{numDraws}.dat",
                        ),
                        atoms=multiFatoms,
                        header=self.output_file_header,
                    )
                if len(hdf5_outputs) > 0:
                    if n == 0:
                        # this can't be done before the loop
                        # because for the multiFatoms we can't easily guess the length
                        self._prepare_hdf5_datasets(
                            h5f,
                            hdf5_outputs,
                            numDraws,
                            multiFatoms,
                            chunk_size_hdf5,
                            **hdf5_kwargs,
                        )
                    if "parameters" in hdf5_outputs:
                        params_for_hdf5[np.mod(n, chunk_size_hdf5), :] = list(
                            paramsDrawn.values()
                        )
                    if "detstats" in hdf5_outputs:
                        detstats_for_hdf5[np.mod(n, chunk_size_hdf5), :] = list(
                            detStats.values()
                        )
                    if "atoms" in hdf5_outputs:
                        mergedAtoms = lalpulsar.mergeMultiFstatAtomsBinned(
                            multiFatoms, self.Tsft
                        )
                        atoms_for_hdf5[
                            np.mod(n, chunk_size_hdf5)
                        ] = utils.reshape_FstatAtomVector_to_array(mergedAtoms)
                    if np.mod(n + 1, chunk_size_hdf5) == 0 or n == numDraws - 1:
                        nLast = int(chunk_size_hdf5 * np.floor(n / chunk_size_hdf5))
                        logger.info(
                            f"Writing chunk {nLast} to {n} of {numDraws} draws"
                            f" to {h5file} ..."
                        )
                        self._write_to_hdf5_chunked(
                            h5f,
                            hdf5_outputs,
                            nLast,
                            chunk_size_hdf5,
                            numDraws,
                            params_for_hdf5,
                            detstats_for_hdf5,
                            FstatMap,
                            atoms_for_hdf5,
                        )
        return candidates

    def synth_one_candidate(self, injParams):
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
        if (
            "twoF" + self.detectors.split(",")[0] in self.detstats
            or BSGL in self.detstats
        ):
            twoFX = np.zeros(lalpulsar.PULSAR_MAX_DETECTORS)
            twoFX[: self.numDetectors] = [
                lalpulsar.ComputeFstatFromAtoms(multiAtoms, X)
                for X in range(self.numDetectors)
            ]
            for X, IFO in enumerate(self.detectors.split(",")):
                detStats["twoF" + IFO] = twoFX[X]
        if BSGL in self.detstats:
            detStats[BSGL] = lalpulsar.ComputeBSGL(
                detStats["twoF"], twoFX, self.BSGLSetup
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
        return injParamsDict, detStats, FstatMap, multiAtoms

    def _prepare_hdf5_datasets(
        self, h5f, hdf5_outputs, numDraws, multiFatoms, chunk_size_hdf5, **kwargs
    ):
        # FIXME: for metadata would be better to store the individual params
        h5f.attrs["header"] = self.output_file_header
        h5f.attrs["numDraws"] = numDraws
        if "parameters" in hdf5_outputs:
            h5f.create_dataset(
                "parameters",
                shape=(numDraws),
                chunks=(chunk_size_hdf5),
                dtype=np.dtype(
                    {
                        "names": self.param_keys,
                        "formats": [(float)] * len(self.param_keys),
                    }
                ),
                **kwargs,
            )
        if "detstats" in hdf5_outputs:
            h5f.create_dataset(
                "detstats",
                shape=(numDraws),
                chunks=(chunk_size_hdf5),
                dtype=np.dtype(
                    {"names": self.detstats, "formats": [(float)] * len(self.detstats)}
                ),
                **kwargs,
            )
        if "FstatMaps" in hdf5_outputs:
            h5f.create_dataset(
                "FstatMaps", shape=(numDraws), chunks=(chunk_size_hdf5), **kwargs
            )
        if "atoms" in hdf5_outputs:
            h5f.create_dataset(
                "atoms",
                shape=(numDraws, multiFatoms.data[0].length, 8),
                chunks=(chunk_size_hdf5, multiFatoms.data[0].length, 8),
                **kwargs,
            )

    def _write_to_hdf5_chunked(
        self,
        h5f,
        hdf5_outputs,
        nLast,
        chunk_size_hdf5,
        numDraws,
        paramsDrawn,
        detStats,
        FstatMaps,
        atoms,
    ):
        if "parameters" in hdf5_outputs:
            for k, key in enumerate(h5f["parameters"].dtype.names):
                h5f["parameters"][nLast : nLast + chunk_size_hdf5, key] = paramsDrawn[
                    : min(chunk_size_hdf5, numDraws - nLast), k
                ]
        if "detstats" in hdf5_outputs:
            for k, key in enumerate(h5f["detstats"].dtype.names):
                h5f["detstats"][nLast : nLast + chunk_size_hdf5, key] = detStats[
                    : min(chunk_size_hdf5, numDraws - nLast), k
                ]
        # if "FstatMaps" in hdf5_outputs:
        # FIXME: need to convert to a plain array
        # h5f["FstatMaps"][nLast:nLast+chunk_size_hdf5] = FstatMaps[:min(chunk_size_hdf5,numDraws-nLast)]
        if "atoms" in hdf5_outputs:
            h5f["atoms"][nLast : nLast + chunk_size_hdf5] = atoms[
                : min(chunk_size_hdf5, numDraws - nLast)
            ]

    def _InjParams_t_to_dict(self, paramsStruct):
        """
        Convert a lalpulsar InjParams_t object into a flat dictionary

        Currently not converted:
          * ampVect
          * multiAM
          * transientWindow
          * detM1o8
        """
        # ensure we always store in the same order
        paramsDict = dict.fromkeys(self.param_keys)
        paramsDict["Alpha"] = paramsStruct.skypos.longitude
        paramsDict["Delta"] = paramsStruct.skypos.latitude
        paramsDict["aPlus"] = paramsStruct.ampParams.aPlus
        paramsDict["aCross"] = paramsStruct.ampParams.aCross
        paramsDict["h0"], paramsDict["cosi"] = utils.convert_aPlus_aCross_to_h0_cosi(
            aPlus=paramsDict["aPlus"], aCross=paramsDict["aCross"]
        )
        paramsDict["psi"] = paramsStruct.ampParams.psi
        paramsDict["phi"] = paramsStruct.ampParams.phi0
        paramsDict["snr"] = paramsStruct.SNR
        return paramsDict
