import os

import lal
import lalpulsar
import numpy as np

import pyfstat.helper_functions as helper_functions
from pyfstat.core import BaseSearchClass
from pyfstat.snr import DetectorStates


class Synthesizer(BaseSearchClass):
    """Efficiently generate lots of F-stats and derived statistics.

    * Generate N samples of detection statistics drawn from their respective distributions,
      assuming Gaussian noise, and drawing signal params from their (given) priors.
    * Python port of lalapps_synthesizeTransientStats and its siblings.
    * See appendix of PGM2011 for details.
    """

    @helper_functions.initializer
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
        randSeed=None,
        timestamps=None,
        signalOnly=False,
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
            helper_functions.get_ephemeris_files().
        transientWindowType: str
            If `none`, a fully persistent CW signal is simulated.
            If `rect` or `exp`, a transient signal with the corresponding
            amplitude evolution is simulated.
        transientStartTime: int or None
            Start time for a transient signal.
        transientTau: int or None
            Duration (`rect` case) or decay time (`exp` case) of a transient signal.
        randSeed: int or None
            Optionally fix the random seed of Gaussian noise generation
            for reproducibility.
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
        """
        self.rng = lal.gsl_rng("mt19937", self.randSeed)
        dets = DetectorStates()
        self.multiDetStates = dets.get_multi_detector_states(
            self.timestamps,
            self.Tsft,
            self.detectors,
        )
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
        print(f"Creating output directory {self.outdir} if it does not yet exist...")
        os.makedirs(self.outdir, exist_ok=True)

    def _init_amplitude_prior(self):
        ampPrior = lalpulsar.AmplitudePrior_t()
        # FIXME handle the various different cases from XLALInitAmplitudePrior()
        print(f"self.h0={self.h0} for amp prior")
        ampPrior.pdf_h0Nat = lalpulsar.CreateSingularPDF1D(self.h0)
        ampPrior.pdf_cosi = lalpulsar.CreateSingularPDF1D(self.cosi)
        ampPrior.pdf_psi = lalpulsar.CreateSingularPDF1D(self.psi)
        ampPrior.pdf_phi0 = lalpulsar.CreateSingularPDF1D(self.phi)
        ampPrior.fixedSNR = -1
        # FIXME support this (-1 means don't use, >=0 legal)
        ampPrior.fixRhohMax = False
        # FIXME support thi
        return ampPrior

    def synth_Fstats(self, numDraws=1):
        print(f"Drawing {numDraws} F-stats with h0={self.h0}.")
        twoF = np.zeros(numDraws)
        for n in range(numDraws):
            twoF[n] = self.synth_one_stat()
        print(twoF)
        return twoF

    def synth_one_stat(self):
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
        # print(f"drawn amplitude parameters: psi,phi0,aPlus,aCross = {injParamsDrawn.ampParams.psi}, {injParamsDrawn.ampParams.phi0}, {injParamsDrawn.ampParams.aPlus}, {injParamsDrawn.ampParams.aCross}")
        windowRange = self.transientInjectRange  # FIXME
        FstatMap = lalpulsar.ComputeTransientFstatMap(
            multiAtoms, windowRange, useFReg=False
        )
        if self.signalOnly:
            FstatMap.maxF += 2
        return 2 * FstatMap.maxF
