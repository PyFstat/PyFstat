from typing import Union

import lal
import lalpulsar
import numpy as np
from attrs import define, field

import pyfstat.helper_functions as helper_functions
from pyfstat.snr import DetectorStates, SignalToNoiseRatio


@define(kw_only=True, slots=False)
class MakeFakeData:

    ephemeris: lalpulsar.EphemerisData = field(
        factory=lambda: lalpulsar.InitBarycenter(
            *helper_functions.get_ephemeris_files()
        )
    )
    fMin: Union[float, None] = field(default=None)
    Band: Union[float, None] = field(default=None)

    timestamps: Union[np.array, dict, None] = field()
    Tsft: Union[int, None] = field(converter=int, default=None)
    detectors: Union[list, None] = field(
        default=None, converter=helper_functions.convert_css_to_list
    )

    sqrtSX: Union[list, None] = field(
        default=None,
        converter=helper_functions.convert_css_to_list,
    )

    randSeed: Union[int, None] = field(default=0)

    def __attrs_post_init__(self):
        self.set_data_parameters()

    def set_data_parameters(self):
        (
            self.multiTimestamps,
            self.multiIFO,
        ) = DetectorStates.parse_timestamps_and_detectors(
            self.timestamps, self.Tsft, self.detectors
        )
        self.multiNoiseFloor = lalpulsar.MultiNoiseFloor()
        lalpulsar.ParseMultiNoiseFloor(
            self.multiNoiseFloor, self.sqrtSX, len(self.detectors)
        )

        self.data_parameters = lalpulsar.CWMFDataParams()
        for param in [
            "multiIFO",
            "multiNoiseFloor",
            "multiTimestamps",
            "randSeed",
            "fMin",
            "Band",
        ]:
            setattr(self.data_parameters, param, getattr(self, param))

    def set_fMin_Band(self, new_fMin=None, new_Band=None):
        self.fMin = new_fMin or self.fMin
        self.Band = new_Band or self.Band
        if hasattr(self, "data_parameters"):
            self.data_parameters.fMin = self.fMin
            self.data_parameters.Band = self.Band

    def parse_signal_parameters(
        self,
        refTime,
        F0,
        F1,
        Alpha,
        Delta,
        psi,
        phi0,
        aPlus=None,
        aCross=None,
        h0=None,
        cosi=None,
    ):
        # FIXME: Discuss this input format

        aPlus, aCross = SignalToNoiseRatio._convert_amplitude_parameters(
            h0=h0, cosi=cosi, aPlus=aPlus, aCross=aCross
        )

        ppv = lalpulsar.CreatePulsarParamsVector(1)

        ppv.data[0].Amp.aPlus = aPlus
        ppv.data[0].Amp.aCross = aCross
        ppv.data[0].Amp.psi = psi
        ppv.data[0].Amp.phi0 = phi0

        ppv.data[0].Doppler.refTime = lal.LIGOTimeGPS(float(refTime))
        ppv.data[0].Doppler.fkdot[0] = F0
        ppv.data[0].Doppler.fkdot[1] = F1
        ppv.data[0].Doppler.Alpha = Alpha
        ppv.data[0].Doppler.Delta = Delta

        return ppv

    @sqrtSX.validator
    def _check_sqrtSX_size(self, attribute, value):

        num_noise_floors = len(value)
        num_detectors = len(self.detectors)
        if (num_noise_floors > 1) and (num_noise_floors != num_detectors):
            raise ValueError(
                "`sqrtSX` should contain one value for each of the detectors "
                "or a single value for all of them. "
                f"Found {num_noise_floors} `sqrtSX` values and {num_detectors} detectors"
            )

    @classmethod
    def from_noiseSFTs(cls, sftpattern, window_type, window_beta):
        pass

    def simulate_data_SFT(self, **params):
        return lalpulsar.CWMakeFakeMultiData(
            0,
            None,
            self.parse_signal_parameters(**params),
            self.data_parameters,
            self.ephemeris,
        )[0]

    def simulate_data(self, **params):
        sfts = self.simulate_data_SFT(**params)
        return helper_functions.unpack_multi_sft_into_dicts(sfts)
