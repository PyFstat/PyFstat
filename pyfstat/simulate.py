import logging
from typing import Optional, Union

import lal
import lalpulsar
import numpy as np
from attrs import converters, define, field

from pyfstat.snr import DetectorStates, SignalToNoiseRatio
from pyfstat.utils import get_ephemeris_files

logger = logging.getLogger(__name__)


@define(kw_only=True, slots=False)
class MakeFakeData:

    ephemeris: lalpulsar.EphemerisData = field(
        factory=lambda: lalpulsar.InitBarycenter(*get_ephemeris_files())
    )

    def set_data_params(
        self,
        fMin: float,
        Band: float,
        Tsft: int,
        timestamps: dict,
        sqrtSX: dict,
        randSeed: int = 0,
    ):

        self.data_params = lalpulsar.CWMFDataParams()
        self.data_params.randSeed = randSeed
        self.data_params.fMin = fMin
        self.data_params.Band = Band

        (
            multi_timestamps,
            multi_detector,
        ) = DetectorStates._parse_timestamps_and_detectors(
            timestamps,
            Tsft,
            detectors=None,
        )

        # FIXME: Waiting for a SWIG fix
        self._swig_multitimestamps = multi_timestamps

        self.data_params.multiTimestamps = multi_timestamps
        self.data_params.multiIFO = multi_detector

        self.data_params.multiNoiseFloor = lalpulsar.MultiNoiseFloor()
        lalpulsar.ParseMultiNoiseFloor(
            self.data_params.multiNoiseFloor,
            [
                str(sqrtSX[prefix])
                for ifo_site in self.data_params.multiIFO.sites
                if (prefix := ifo_site.frDetector.prefix)
            ],
            self.data_params.multiIFO.length,
        )
        return self

    # def set_from_noise_sfts(cls, sftpattern, window_type, window_beta):
    #    pass

    # def parse_signal_parameters(
    #    self,
    #    signal_parameters,
    # ):
    #    # FIXME: Discuss this input format
    #    amplitude_parameters =
    #    {key: signal_parameters.get(key) for key in ["h0", "cosi", "aPlus", "aCross"]}
    #    aPlus, aCross = SignalToNoiseRatio._convert_amplitude_parameters(
    #        h0=h0, cosi=cosi, aPlus=aPlus, aCross=aCross
    #    )

    #    ppv = lalpulsar.CreatePulsarParamsVector(1)

    #    ppv.data[0].Amp.aPlus = aPlus
    #    ppv.data[0].Amp.aCross = aCross
    #    ppv.data[0].Amp.psi = psi
    #    ppv.data[0].Amp.phi0 = phi0

    #    ppv.data[0].Doppler.refTime = lal.LIGOTimeGPS(float(refTime))
    #    ppv.data[0].Doppler.fkdot[0] = F0
    #    ppv.data[0].Doppler.fkdot[1] = F1
    #    ppv.data[0].Doppler.Alpha = Alpha
    #    ppv.data[0].Doppler.Delta = Delta

    #    return ppv

    # def simulate_data_SFT(self, **params):
    #    return lalpulsar.CWMakeFakeMultiData(
    #        0,
    #        None,
    #        self.parse_signal_parameters(**params),
    #        self.data_parameters,
    #        self.ephemeris,
    #    )[0]

    # def simulate_data(self, **params):
    #    sfts = self.simulate_data_SFT(**params)
    #    return helper_functions.unpack_multi_sft_into_dicts(sfts)
