import os

import numpy as np

from pyfstat import Writer
from pyfstat.utils import get_sft_as_arrays, plot_real_imag_spectrograms


def test_get_sft_as_arrays(tmp_path):
    writer_kwargs = {
        "sqrtSX": 1,
        "Tsft": 1800,
        "timestamps": {  # Set up SFTs with arbitrary gaps
            "H1": 1800 * np.array([1, 2, 4, 6, 8, 10]),
            "L1": 1800 * np.array([1, 3, 5, 7, 9, 12]),
        },
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

    frequency_step = 1 / writer_kwargs["Tsft"]
    num_frequencies = np.floor(writer_kwargs["Band"] * writer_kwargs["Tsft"] + 0.5)
    expected_frequencies = (
        writer_kwargs["F0"]
        - writer_kwargs["Band"] / 2
        + frequency_step * np.arange(num_frequencies)
    )

    # Single detector
    single_detector_path = writer.sftfilepath.split(";")[0]
    ifo = writer.detectors.split(",")[0]
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


def test_spectrogram():

    data_parameters = {
        "label": "SpectogramTest",
        "outdir": "tests/test_utils",
        "sqrtSX": 1e-23,
        "tstart": 1238166018,
        "duration": 15 * 1800,
        "detectors": "H1",
        "Tsft": 1800,
    }

    signal_parameters = {
        "F0": 100.0,
        "F1": -1e-9,
        "F2": 0.0,
        "Alpha": 0.0,
        "Delta": 0.5,
        "cosi": 0.0,
        "psi": 0.0,
        "phi": 0.0,
        "tref": 1238166018,
    }

    # making data
    data = Writer(**data_parameters, **signal_parameters)
    data.make_data()

    outdir = data_parameters["outdir"]
    label = data_parameters["label"]

    ax = plot_real_imag_spectrograms(
        sftfilepattern=data.sftfilepath, savefig=True, outdir=outdir, label=label
    )

    plotfile = os.path.join(outdir, label + ".png")

    ax

    assert os.path.isfile(plotfile)
