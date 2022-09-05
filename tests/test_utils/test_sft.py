import numpy as np

from pyfstat import Writer
from pyfstat.utils import get_sft_as_arrays


def test_get_sft_as_arrays(tmp_path):

    writer_kwargs = {
        "sqrtSX": 1,
        "Tsft": 1800,
        "timestamps": {
            "H1": np.array([1, 2, 3, 6, 7, 10, 12]),
            "L1": np.array([14, 25, 36, 47]),
        },
        "detectors": "H1,L1",
        "SFTWindowType": "tukey",
        "SFTWindowBeta": 0.001,
        "randSeed": 42,
        "F0": 10.0,
        "Band": 0.1,
        "label": "testing_sft_to_array",
        "outdir": tmp_path,
    }
    writer = Writer(**writer_kwargs)
    writer.make_data()

    frequency_step = 1 / writer_kwargs["Tsft"]
    num_frequencies = np.floor(writer_kwargs["Band"] * writer_kwargs["Tsft"] + 0.5)
    expected_frequencies = (
        writer_kwargs["F0"]
        - writer_kwargs["Band"] / 2
        + frequency_step * np.arange(num_frequencies)
    )

    # Single detector
    single_detector_path = writer.sftfilepath.split(";")[0]
    ifo = "H1" if "H1" in single_detector_path else "L1"
    frequencies, times, amplitudes = get_sft_as_arrays(single_detector_path)
    np.testing.assert_allclose(frequencies, expected_frequencies)
    np.testing.assert_equal(times[ifo], writer_kwargs["timestamps"][ifo])
    assert frequencies.shape + times[ifo].shape == amplitudes[ifo].shape

    # Multi-detector
    frequencies, times, amplitudes = get_sft_as_arrays(single_detector_path)
    np.testing.assert_allclose(frequencies, expected_frequencies)
    for ifo in times:
        np.testing.assert_equal(times[ifo], writer_kwargs["timestamps"][ifo])
        assert frequencies.shape + times[ifo].shape == amplitudes[ifo].shape
