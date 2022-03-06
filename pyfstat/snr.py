from attrs import define, field
import lal
import lalpulsar
import logging
import numpy as np
from typing import Union

from pyfstat.helper_functions import get_ephemeris_files


@define(kw_only=True, slots=False)
class SignalToNoiseRatio:

    detector_states: lalpulsar.MultiDetectorStateSeries = field()
    noise_weights: Union[lalpulsar.MultiNoiseWeights, None] = field(default=None)
    assumeSqrtSX: float = field(default=None)
    Tsft: float = field(default=None)

    @classmethod
    def from_sfts(cls):
        """
        Allow to include noise weights from SFTs.
        This will mean detector states will be retrieved from SFTs directly
        """
        pass

    def __attrs_post_init__(self):
        if self.noise_weights is None:
            if self.assumeSqrtSX is not None and self.Tsft is not None:
                self.Sinv_Tsft = self.Tsft / self.assumeSqrtSX**2
            else:
                raise ValueError(
                    "Need either (`assumeSqrtSX`, `Tsft`) or `noise_weights` to account for background noise"
                )

    def compute_snr2(
        self, Alpha, Delta, psi, phi0, h0=None, cosi=None, aPlus=None, aCross=None
    ):
        aPlus, aCross = self._convert_amplitude_parameters(
            h0=h0, cosi=cosi, aPlus=aPlus, aCross=aCross
        )

        Aphys = lalpulsar.PulsarAmplitudeParams()
        Aphys.psi = psi
        Aphys.phi0 = phi0
        Aphys.aPlus = aPlus
        Aphys.aCross = aCross

        # FIXME Here as well!!!!!
        M = self.compute_Mmunu(Alpha=Alpha, Delta=Delta)
        M.Sinv_Tsft = self.Sinv_Tsft

        return lalpulsar.ComputeOptimalSNR2FromMmunu(Aphys, M)

    def compute_twoF(self, *args, **kwargs):
        snr2 = self.compute_snr2(*args, **kwargs)
        expected_2F = snr2 + 4.0
        stdev_2F = np.sqrt(8.0 + 4.0 * snr2)
        return expected_2F, stdev_2F

    def compute_Mmunu(self, Alpha, Delta):

        sky = lal.SkyPosition()
        sky.longitude = Alpha
        sky.latitude = Delta
        sky.system = lal.COORDINATESYSTEM_EQUATORIAL
        lal.NormalizeSkyPosition(sky.longitude, sky.latitude)

        # FIXME Missing Mmunu.Sinv_Tsft = t_sft / psd !!!!!!
        return lalpulsar.ComputeMultiAMCoeffs(
            multiDetStates=self.detector_states,
            multiWeights=self.noise_weights,
            skypos=sky,
        ).Mmunu

    def _convert_amplitude_parameters(self, h0, cosi, aPlus, aCross):
        h0_cosi = h0 is not None and cosi is not None
        aPlusCross = aPlus is not None and aCross is not None

        if h0_cosi == aPlusCross:
            raise ValueError("Need either (h0, cosi) or (aPlus, aCross), but not both")

        if h0_cosi:
            aPlus = 0.5 * h0 * (1 + cosi**2)
            aCross = h0 * cosi
            return aPlus, aCross

        return aPlus, aCross


class DetectorStates:
    def __init__(self):
        self.ephems = lalpulsar.InitBarycenter(*get_ephemeris_files())

    def multi_detector_states(self, timestamps, detectors=None, time_offset=0):
        """
        Parameters
        ----------
        time_stamps: array-like or dict
            GPS timestamps at which detector states will be retrieved.
            If array, use the same set of timestamps for all detectors,
            which must be explicitly given by the user via`detectors`.
            If dictionary, each key should correspond to a valid detector name
            to be parsed by XLALParseMultiLALDetector and the associated value
            should be an array-like set of GPS timestamps for each individual detector.

        detectors: list[str] or comma-separated string
            List of detectors to be parsed using XLALParseMultiLALDetector.

        time_offset: float
            Timestamp offset to retrieve detector states.


        Returns
        -------
        multi_detector_states: lalpulsar.MultiDetectorStateSeries
            Resulting multi-detector states produced by XLALGetMultiDetectorStates
        """

        self._parse_timestamps_and_detectors(timestamps, detectors)
        return lalpulsar.GetMultiDetectorStates(
            self.multi_timestamps,
            self.multi_detector,
            self.ephems,
            time_offset,
        )

    def multi_detector_states_from_sfts(self, sftpath, time_offset):
        pass

    def _parse_timestamps_and_detectors(self, timestamps, detectors):

        if isinstance(timestamps, dict):

            if detectors is not None:
                raise ValueError("`timestamps`' keys are redundant with `detectors`. ")

            logging.debug("Retrieving detectors from tiemstamps dictionary.")
            detectors = list(timestamps.key())
            timestamps = timestamps.values()

        elif detectors is not None:
            if isinstance(detectors, str):
                logging.debug("Converting `detectors` string to list")
                detectors = detectors.replace(" ", "").split(",")

            logging.debug("Checking integrity of `timestamps`")
            ts = np.array(timestamps)
            if ts.dtype == np.dtype("O") or ts.ndim > 1:
                raise ValueError("`timestamps` is not a 1D list of numerical values")
            timestamps = (ts for ifo in detectors)

        self.multi_detector = lalpulsar.MultiLALDetector()
        lalpulsar.ParseMultiLALDetector(self.multi_detector, detectors)

        self.multi_timestamps = lalpulsar.CreateMultiLIGOTimeGPSVector(
            self.multi_detector.length
        )
        for ind, ts in enumerate(timestamps):
            self.multi_timestamps.data[ind] = self._numpy_array_to_LIGOTimeGPSVector(ts)

    @staticmethod
    def _numpy_array_to_LIGOTimeGPSVector(numpy_array):
        """Pure brute force conversion"""

        if numpy_array.ndim != 1:
            raise ValueError(
                f"Time stamps array must be 1D: Current one has {numpy_array.ndim}."
            )

        seconds_array = np.floor(numpy_array)
        nanoseconds_array = np.floor(1e9 * (numpy_array - seconds_array))

        time_gps_vector = lalpulsar.CreateTimestampVector(numpy_array.shape[0])
        for ind in range(time_gps_vector.length):
            time_gps_vector.data[ind] = lal.LIGOTimeGPS(
                int(seconds_array[ind]), int(nanoseconds_array[ind])
            )

        return time_gps_vector
