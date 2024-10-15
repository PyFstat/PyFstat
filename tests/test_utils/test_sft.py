import os

import matplotlib
import numpy as np
import pytest

from pyfstat import Writer
from pyfstat.utils import get_sft_as_arrays, plot_spectrogram


@pytest.fixture
def timestamps_for_test():
    # Set up SFTs with arbitrary gaps
    return {
        "H1": 1800 * np.array([1, 2, 4, 6, 8, 10]),
        "L1": 1800 * np.array([1, 3, 5, 7, 9, 12]),
    }


@pytest.fixture
def data_for_test(tmp_path, timestamps_for_test):
    writer_kwargs = {
        "sqrtSX": 1,
        "Tsft": 1800,
        "timestamps": timestamps_for_test,
        "detectors": "H1,L1",
        "SFTWindowType": "tukey",
        "SFTWindowParam": 0.001,
        "randSeed": 42,
        "F0": 10.0,
        "Band": 0.1,
        "label": "TestingSftToArray",
        "outdir": tmp_path,
    }
    writer = Writer(**writer_kwargs)
    writer.make_data()

    return writer


def test_get_sft_as_arrays(data_for_test, timestamps_for_test):

    frequency_step = 1 / data_for_test.Tsft
    num_frequencies = np.floor(data_for_test.Band * data_for_test.Tsft + 0.5)
    expected_frequencies = (
        data_for_test.F0
        - data_for_test.Band / 2
        + frequency_step * np.arange(num_frequencies)
    )

    # Single detector
    single_detector_path = data_for_test.sftfilepath.split(";")[0]
    ifo = data_for_test.detectors.split(",")[0]
    frequencies, times, amplitudes = get_sft_as_arrays(single_detector_path)
    np.testing.assert_allclose(frequencies, expected_frequencies)
    np.testing.assert_equal(times[ifo], timestamps_for_test[ifo])
    assert frequencies.shape + times[ifo].shape == amplitudes[ifo].shape

    # Multi-detector
    frequencies, times, amplitudes = get_sft_as_arrays(single_detector_path)
    np.testing.assert_allclose(frequencies, expected_frequencies)
    for ifo in times:
        np.testing.assert_equal(times[ifo], timestamps_for_test[ifo])
        assert frequencies.shape + times[ifo].shape == amplitudes[ifo].shape


@pytest.mark.parametrize("quantity", ["power", "normpower", "real", "imag"])
def test_spectrogram(data_for_test, timestamps_for_test, quantity):

    ax = plot_spectrogram(
        sftfilepattern=data_for_test.sftfilepath,
        quantity=quantity,
        sqrtSX=data_for_test.sqrtSX,
        detector=data_for_test.detectors.split(",")[0],
        savefig=True,
        outdir=data_for_test.outdir,
        label=data_for_test.label,
    )

    assert isinstance(ax, matplotlib.axes.Axes)

    plotfile = os.path.join(data_for_test.outdir, data_for_test.label + ".png")

    assert os.path.isfile(plotfile)
