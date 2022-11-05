import logging
import os

import lal
import lalpulsar
import numpy as np

import pyfstat.utils as utils
from pyfstat.core import BaseSearchClass
from pyfstat.snr import DetectorStates

logger = logging.getLogger(__name__)


class Synthesizer(BaseSearchClass):
    """Efficiently generate lots of F-statistic values and derived statistics.

    * Generate N samples of detection statistics drawn from their respective distributions,
      assuming Gaussian noise, and drawing signal parameters from their (given) priors.
    * Python port of lalpulsar_synthesizeTransientStats and its siblings.
    * See appendix of PGM2011 [ https://arxiv.org/abs/1104.1704 ] for details.
    """

    @utils.initializer
    def __init__(
        self,
        label,
        outdir,
        tstart=None,
        duration=None,
        Alpha=None,
        Delta=None,
        h0=None,
        cosi=None,
        psi=0.0,
        phi=0,
        Tsft=1800,
        detectors=None,
        earth_ephem=None,
        sun_ephem=None,
        transientWindowType="none",
        transientStartTime=None,
        transientTau=None,
        randSeed=0,
        timestamps=None,
        signalOnly=False,
        detstats=[],
    ):
        """
        Parameters
        ----------
        label: string
            A human-readable label to be used in naming the output files.
        outdir: str
            The directory where files are written to.
            Default: current working directory.
        tstart: int
            Starting GPS epoch of the data set.
            NOTE: mutually exclusive with `timestamps`.
        duration: int
            Duration (in GPS seconds) of the total data set.
            NOTE: mutually exclusive with `timestamps`.
        Alpha, Delta, h0, cosi, psi, phi: float or None
            Additional frequency evolution and amplitude parameters for a signal.
            If `h0=None` or `h0=0`, these are all ignored.
            If `h0>0`, then at least `[Alpha,Delta,cosi]` need to be set explicitly.
        Tsft: int
            The SFT duration in seconds.
            Will be ignored if `noiseSFTs` are given.
        detectors: str or None
            Comma-separated list of detectors to generate data for.
        earth_ephem, sun_ephem: str or None
            Paths of the two files containing positions of Earth and Sun.
            If None, will check standard sources as per
            utils.get_ephemeris_files().
        transientWindowType: str
            If `none`, a fully persistent CW signal is simulated.
            If `rect` or `exp`, a transient signal with the corresponding
            amplitude evolution is simulated.
        transientStartTime: int or None
            Start time for a transient signal.
        transientTau: int or None
            Duration (`rect` case) or decay time (`exp` case) of a transient signal.
        randSeed: int
            Optionally fix the random seed of Gaussian noise generation
            for reproducibility. Default of `0` means no fixed seed.
        timestamps: str or dict
            Dictionary of timestamps (each key must refer to a detector),
            list of timestamps (`detectors` should be set),
            or comma-separated list of per-detector timestamps files
            (simple text files, lines with <seconds> <nanoseconds>,
            comments should use `%`, each time stamp gives the
            start time of one SFT).
            WARNING: In that last case, order must match that of `detectors`!
            NOTE: mutually exclusive with [`tstart`,`duration`]
            and with `noiseSFTs`.
        signalOnly: bool
            Generate pure signal without noise?
        detstats: list
            Detection statistics to compute.
            See :fuc:`~pyfstat.utils.parse_detstats`
            for the supported format.
            For details of supported `BSGL` parameters,
            see :func:`~pyfstat.utils.get_BSGL_setup`.
        """
        self.rng = lal.gsl_rng("mt19937", self.randSeed)
        dets = DetectorStates()
        self.multiDetStates = dets.get_multi_detector_states(
            self.timestamps,
            self.Tsft,
            self.detectors,
        )
        self.numDetectors = self.multiDetStates.length
        # FIXME: should this really be done at init time, or at synth time?
        self.ampPrior = self._init_amplitude_prior()
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
        BSGL = utils.translate_detstats("BSGL")
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

    def _init_amplitude_prior(self):
        ampPrior = lalpulsar.AmplitudePrior_t()
        # FIXME handle the various different cases from XLALInitAmplitudePrior()
        logger.debug(f"self.h0={self.h0} for amp prior")
        ampPrior.pdf_h0Nat = lalpulsar.CreateSingularPDF1D(self.h0)
        ampPrior.pdf_cosi = lalpulsar.CreateSingularPDF1D(self.cosi)
        ampPrior.pdf_psi = lalpulsar.CreateSingularPDF1D(self.psi)
        ampPrior.pdf_phi0 = lalpulsar.CreateSingularPDF1D(self.phi)
        ampPrior.fixedSNR = -1
        # FIXME support this (-1 means don't use, >=0 legal)
        ampPrior.fixRhohMax = False
        # FIXME support thi
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
        logger.info(f"Drawing {numDraws} F-stats with h0={self.h0}.")
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
        injParamsDrawn = (
            lalpulsar.InjParams_t()
        )  # output struct, will be filled by lalpulsar call
        skypos = lal.SkyPosition()
        skypos.longitude = self.Alpha
        skypos.latitude = self.Delta
        skypos.system = lal.COORDINATESYSTEM_EQUATORIAL
        multiAMBuffer = lalpulsar.multiAMBuffer_t()
        multiAtoms = lalpulsar.SynthesizeTransientAtoms(
            injParamsOut=injParamsDrawn,
            skypos=skypos,
            AmpPrior=self.ampPrior,
            transientInjectRange=self.transientInjectRange,
            multiDetStates=self.multiDetStates,
            SignalOnly=self.signalOnly,
            multiAMBuffer=multiAMBuffer,
            rng=self.rng,
            lineX=-1,
            multiNoiseWeights=None,
        )
        injParamsDict = self._InjParams_t_to_dict(injParamsDrawn)
        # FIXME support pycuda version of ComputeTransientFstatMap
        detStats = {}
        BSGL = utils.translate_detstats("BSGL")
        BtSG = utils.translate_detstats("BtSG")
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
