import os

import matplotlib
import numpy as np
import pytest

from pyfstat import Writer
from pyfstat.utils import get_sft_as_arrays, plot_spectrogram


@pytest.fixture(scope="module", params=["nogaps", "gaps"])
def data_for_test(tmp_path_factory, request):
    Tsft = 1800
    ts_base = np.arange(start=1, stop=7 * Tsft, step=Tsft, dtype=int)
    timestamps = {
        "H1": ts_base,
        "L1": ts_base,
    }
    gaps = request.param
    if gaps == "gaps":
        for ifo in timestamps.keys():
            timestamps[ifo] = np.delete(timestamps[ifo], [2, 4])
    writer_kwargs = {
        "sqrtSX": 1,
        "Tsft": 1800,
        "timestamps": timestamps,
        "detectors": "H1,L1",
        "SFTWindowType": "tukey",
        "SFTWindowParam": 0.001,
        "randSeed": 42,
        "F0": 10.0,
        "Band": 0.1,
        "label": "TestSFTs" + gaps,
        "outdir": tmp_path_factory.mktemp("SFTdata" + gaps),
    }
    writer = Writer(**writer_kwargs)
    writer.make_data()
    return writer, timestamps


def test_get_sft_as_arrays(data_for_test):

    writer, timestamps = data_for_test

    frequency_step = 1 / writer.Tsft
    num_frequencies = np.floor(writer.Band * writer.Tsft + 0.5)
    expected_frequencies = (
        writer.F0 - writer.Band / 2 + frequency_step * np.arange(num_frequencies)
    )

    # Single detector
    single_detector_path = writer.sftfilepath.split(";")[0]
    ifo = writer.detectors.split(",")[0]
    frequencies, times, amplitudes = get_sft_as_arrays(single_detector_path)
    np.testing.assert_allclose(frequencies, expected_frequencies)
    np.testing.assert_equal(times[ifo], timestamps[ifo])
    assert frequencies.shape + times[ifo].shape == amplitudes[ifo].shape

    # Multi-detector
    frequencies, times, amplitudes = get_sft_as_arrays(single_detector_path)
    np.testing.assert_allclose(frequencies, expected_frequencies)
    for ifo in times:
        np.testing.assert_equal(times[ifo], timestamps[ifo])
        assert frequencies.shape + times[ifo].shape == amplitudes[ifo].shape


@pytest.mark.parametrize("quantity", ["power", "normpower", "real", "imag"])
def test_spectrogram(data_for_test, quantity):

    writer, timestamps = data_for_test

    plotlabel = f"{writer.label}_{quantity}"
    ax = plot_spectrogram(
        sftfilepattern=writer.sftfilepath,
        quantity=quantity,
        sqrtSX=writer.sqrtSX,
        detector=writer.detectors.split(",")[0],
        savefig=True,
        outdir=writer.outdir,
        label=plotlabel,
    )
    assert isinstance(ax, matplotlib.axes.Axes)

    plotfile = os.path.join(writer.outdir, plotlabel + ".png")
    assert os.path.isfile(plotfile)
