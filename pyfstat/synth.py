import logging
import os
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
      drawn from their respective distributions,
      assuming Gaussian noise, and drawing signal parameters from their (given) priors.
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
        transientWindowType: str = "none",
        transientStartTime: int = None,
        transientTau: int = None,
        randSeed: int = 0,
        timestamps: Union[str, dict] = None,
        signalOnly: bool = False,
        detstats: Sequence[Union[str, dict]] = [],
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
            NOTE: mutually exclusive with `timestamps`.
        duration:
            Duration (in GPS seconds) of the total data set.
            NOTE: mutually exclusive with `timestamps`.
        priors:
            List of priors for parameters [Alpha,Delta,h0,cosi,psi,phi0],
            to be parsed by
            :func:`~pyfstat.injection_parameters.InjectionParametersGenerator`.
            In the simplest case, for fixed values, a minimal example would be
            ::

            priors = {"Alpha": 0, "Delta": 0, "h0": 0, "cosi": 0, "psi": 0, "phi": 0}

        Tsft:
            The SFT duration in seconds.
            Will be ignored if `noiseSFTs` are given.
        detectors:
            Comma-separated list of detectors to generate data for.
        earth_ephem, sun_ephem:
            Paths of the two files containing positions of Earth and Sun.
            If None, will check standard sources as per
            utils.get_ephemeris_files().
        transientWindowType:
            If `none`, a fully persistent CW signal is simulated.
            If `rect` or `exp`, a transient signal with the corresponding
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
            comments should use `%`, each time stamp gives the
            start time of one SFT).
            WARNING: In that last case, order must match that of `detectors`!
            NOTE: mutually exclusive with [`tstart`,`duration`]
            and with `noiseSFTs`.
        signalOnly:
            Generate pure signal without noise?
        detstats:
            Detection statistics to compute.
            See :func:`pyfstat.utils.detstats.parse_detstats`
            for the supported format.
            For details of supported `BSGL` parameters,
            see :func:`pyfstat.utils.detstats.get_BSGL_setup`.
        """
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
        if isinstance(priors["Alpha"], float) and isinstance(priors["Delta"]):
            self.skypos.longitude = priors["Alpha"]
            self.skypos.latitude = priors["Delta"]
        self.paramsGen = InjectionParametersGenerator(
            priors=self.priors, seed=self.randSeed
        )
        # FIXME: handle these separately
        self.injectWindow_type = self.transientWindowType
        self.searchWindow_type = self.transientWindowType
        self.injectWindow_t0 = self.transientStartTime
        self.injectWindow_tau = self.transientTau
        self.injectWindow_t0Band = 0
        self.injectWindow_tauBand = 0
        self.transientInjectRange = lalpulsar.transientWindowRange_t()
        self.transientInjectRange.type = lalpulsar.ParseTransientWindowName(
            self.injectWindow_type
        )
        self.transientInjectRange.t0 = self.tstart + self.injectWindow_t0
        # tauMax = self.injectWindow_tau + self.injectWindow_tau
        self.transientInjectRange.t0Band = self.injectWindow_t0Band
        self.transientInjectRange.tau = self.injectWindow_tau
        self.transientInjectRange.tauBand = self.injectWindow_tauBand
        self.detstats, detstat_params = utils.parse_detstats(self.detstats)
        BSGL = utils.get_canonical_detstat_name("BSGL")
        if BSGL in self.detstats:
            self.BSGLSetup = utils.get_BSGL_setup(
                numDetectors=self.numDetectors,
                numSegments=1,
                **detstat_params[BSGL],
            )
        logger.debug(
            f"Creating output directory {self.outdir} if it does not yet exist..."
        )
        os.makedirs(self.outdir, exist_ok=True)

    def _set_amplitude_prior(self, injParams):
        for key, val in injParams.items():
            logging.info(f"{key}={val}")
        ampPrior = lalpulsar.AmplitudePrior_t()
        ampPrior.pdf_h0Nat = lalpulsar.CreateSingularPDF1D(injParams["h0"])
        ampPrior.pdf_cosi = lalpulsar.CreateSingularPDF1D(injParams["cosi"])
        ampPrior.pdf_psi = lalpulsar.CreateSingularPDF1D(injParams["psi"])
        ampPrior.pdf_phi0 = lalpulsar.CreateSingularPDF1D(injParams["phi"])
        ampPrior.fixedSNR = -1
        # FIXME support this (-1 means don't use, >=0 legal)
        ampPrior.fixRhohMax = False
        # FIXME support this
        return ampPrior

    def synth_candidates(
        self, numDraws=1, keep_params=False, keep_FstatMaps=False, keep_atoms=False
    ):
        candidates = {
            stat: np.tile(np.nan, (numDraws, self.numDetectors))
            if stat == "twoFX"
            else np.repeat(np.nan, numDraws)
            for stat in self.detstats
        }
        if keep_FstatMaps:
            candidates["FstatMaps"] = []
        if keep_atoms:
            candidates["atoms"] = []
        logger.info(f"Drawing {numDraws} results.")
        for n in range(numDraws):
            detStats, params, FstatMap, atoms = self.synth_one_candidate()
            for key, val in detStats.items():
                if key == "twoFX":
                    candidates[key][n, :] = val[: self.numDetectors]
                else:
                    candidates[key][n] = val
            if keep_params:
                if n == 0:
                    for key in params.keys():
                        candidates[key] = np.repeat(np.nan, numDraws)
                for key, val in params.items():
                    candidates[key][n] = val
            if keep_FstatMaps:
                candidates["FstatMaps"].append(FstatMap)
            if keep_atoms:
                candidates["atoms"].append(atoms)
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
