import glob
import logging
import os
import time

import lalpulsar
import numpy as np
import pytest
from commons_for_tests import (
    default_binary_params,
    default_signal_params,
    default_signal_params_no_sky,
    default_transient_params,
    default_Writer_params,
)

import pyfstat


def test_timestamp_files(tmp_path, caplog):
    Tsft = 1800
    single_column = 10000000 + Tsft * np.arange(10)
    np.savetxt(tmp_path / "single_column.txt", single_column[:, None])
    label = "TimestampsFileTesting"
    outdir = tmp_path / label

    writer = pyfstat.Writer(
        outdir=outdir,
        label=label,
        F0=10,
        Band=0.1,
        detectors="H1",
        sqrtSX=1e-23,
        Tsft=Tsft,
        timestamps=str(tmp_path / "single_column.txt"),
    )

    assert single_column[0] == writer.tstart
    assert single_column[-1] - single_column[0] == writer.duration - Tsft

    np.savetxt(tmp_path / "dual_column.txt", np.hstack(2 * [single_column[:, None]]))
    with caplog.at_level("WARNING"):
        writer = pyfstat.Writer(
            outdir=outdir,
            label=label,
            F0=10,
            Band=0.1,
            detectors="H1",
            sqrtSX=1e-23,
            Tsft=Tsft,
            timestamps=str(tmp_path / "dual_column.txt"),
        )
        log_warning_messages = "\n".join(
            [log_tuple[-1] for log_tuple in caplog.record_tuples]
        )
        assert "has more than 1 column" in log_warning_messages
        assert "we will ignore the rest" in log_warning_messages

    np.savetxt(tmp_path / "float_column.txt", 0.01 + single_column[:, None])
    with caplog.at_level("WARNING"):
        writer = pyfstat.Writer(
            outdir=outdir,
            label=label,
            F0=10,
            Band=0.1,
            detectors="H1",
            sqrtSX=1e-23,
            Tsft=Tsft,
            timestamps=str(tmp_path / "float_column.txt"),
        )
        log_warning_messages = "\n".join(
            [log_tuple[-1] for log_tuple in caplog.record_tuples]
        )
        assert "non-integer timestamps" in log_warning_messages
        assert "floor" in log_warning_messages

    # Test wrong number of detectors
    with pytest.raises(ValueError):
        writer = pyfstat.Writer(
            outdir=outdir,
            label=label,
            F0=10,
            Band=0.1,
            detectors="H1,L1",
            sqrtSX=1e-23,
            timestamps=str(tmp_path / "single_column.txt"),
        )

    # Test unspecified detectors
    with pytest.raises(ValueError):
        writer = pyfstat.Writer(
            outdir=outdir,
            label=label,
            F0=10,
            Band=0.1,
            sqrtSX=1e-23,
            timestamps=str(tmp_path / "single_column.txt"),
        )


@pytest.mark.usefixtures("data_fixture")
class TestWriter:
    label = "TestWriter"
    writer_class_to_test = pyfstat.Writer
    signal_parameters = default_signal_params
    # this needs to be overwritable by child test classes that don't support multi-IFOs
    multi_detectors = "H1,L1"

    def test_make_cff(self):
        self.Writer.make_cff(verbose=True)
        assert os.path.isfile(os.path.join(".", self.outdir, self.label + ".cff"))

    def test_make_cff_new_style_separate_F0(self):
        if self.style == "old":
            pytest.skip()
        signal_params = default_signal_params.copy()
        signal_params.pop("F0")
        test_Writer = pyfstat.Writer(
            label=self.label + "NoF0",
            tstart=self.Writer.tstart,
            duration=self.Writer.duration,
            signal_parameters=signal_params,
            F0=default_signal_params["F0"],
            Tsft=self.Writer.Tsft,
            outdir=self.outdir,
            sqrtSX=self.Writer.sqrtSX,
            Band=self.Writer.Band,
            detectors=self.Writer.detectors,
            SFTWindowType=self.Writer.SFTWindowType,
            SFTWindowParam=self.Writer.SFTWindowParam,
            randSeed=self.Writer.randSeed,
        )
        test_Writer.make_cff(verbose=True)
        assert os.path.isfile(
            os.path.join(".", test_Writer.outdir, test_Writer.label + ".cff")
        )

    def test_make_cff_old_style_without_tref(self):
        if self.style == "new":
            pytest.skip()
        test_Writer = pyfstat.Writer(
            label=self.label + "NoTref",
            tstart=self.Writer.tstart,
            duration=self.Writer.duration,
            tref=None,
            F0=default_signal_params["F0"],
            F1=default_signal_params["F1"],
            F2=default_signal_params["F2"],
            Alpha=default_signal_params["Alpha"],
            Delta=default_signal_params["Delta"],
            h0=default_signal_params["h0"],
            cosi=default_signal_params["cosi"],
            psi=default_signal_params["psi"],
            phi=default_signal_params["phi"],
            signal_parameters=None,
            Tsft=self.Writer.Tsft,
            outdir=self.outdir,
            sqrtSX=self.Writer.sqrtSX,
            Band=self.Writer.Band,
            detectors=self.Writer.detectors,
            SFTWindowType=self.Writer.SFTWindowType,
            SFTWindowParam=self.Writer.SFTWindowParam,
            randSeed=self.Writer.randSeed,
        )
        test_Writer.make_cff(verbose=True)
        assert os.path.isfile(
            os.path.join(".", test_Writer.outdir, test_Writer.label + ".cff")
        )

    def test_run_makefakedata(self):
        self.Writer.make_data(verbose=True)
        numSFTs = int(np.ceil(self.Writer.duration / self.Writer.Tsft))
        expected_outfile = os.path.join(
            self.Writer.outdir,
            "{:1s}-{:d}_{:2s}_{:d}SFT_{:s}-{:d}-{:d}.sft".format(
                self.Writer.detectors[0],
                numSFTs,
                self.Writer.detectors,
                self.Writer.Tsft,
                self.Writer.label,
                self.Writer.tstart,
                numSFTs * self.Writer.Tsft,
            ),
        )
        assert os.path.isfile(expected_outfile)
        assert lalpulsar.ValidateSFTFile(expected_outfile) == 0

    def test_makefakedata_usecached(self):
        if os.path.isfile(self.Writer.config_file_name):
            os.remove(self.Writer.config_file_name)
        if os.path.isfile(self.Writer.sftfilepath):
            os.remove(self.Writer.sftfilepath)

        # first run: make everything from scratch
        self.Writer.make_cff(verbose=True)
        self.Writer.run_makefakedata()
        time_first = os.path.getmtime(self.Writer.sftfilepath)

        # second run: should reuse .cff and .sft
        self.Writer.make_cff(verbose=True)
        self.Writer.run_makefakedata()
        time_second = os.path.getmtime(self.Writer.sftfilepath)
        assert time_first == time_second

        # third run: touch the .cff to force regeneration
        time.sleep(1)  # make sure timestamp is actually different!
        os.system("touch {}".format(self.Writer.config_file_name))
        self.Writer.run_makefakedata()
        time_third = os.path.getmtime(self.Writer.sftfilepath)
        assert not (time_first == time_third)

        # fourth run: delete .cff and expect a RuntimeError
        os.remove(self.Writer.config_file_name)
        with pytest.raises(RuntimeError):
            self.Writer.run_makefakedata()

    def test_noise_sfts(self):
        randSeed = 69420

        # create SFTs with both noise and a signal in them
        noise_and_signal_writer = self.writer_class_to_test(
            label="TestNoiseSFTsNoiseAndSignal",
            outdir=self.outdir,
            duration=self.Writer.duration,
            Tsft=self.Writer.Tsft,
            tstart=self.Writer.tstart,
            detectors=self.multi_detectors,
            randSeed=randSeed,
            SFTWindowType=self.Writer.SFTWindowType,
            SFTWindowParam=self.Writer.SFTWindowParam,
            sqrtSX=self.Writer.sqrtSX,
            Band=0.5,
            **(
                {
                    k: v
                    for k, v in self.signal_parameters.items()
                    if not (k.startswith("F") and int(k[-1]) > 2)
                }
                if self.style == "old"
                else {}
            ),
            signal_parameters=self.signal_parameters if self.style == "new" else None,
        )
        noise_and_signal_writer.make_data(verbose=True)

        # create noise-only SFTs
        noise_writer = self.writer_class_to_test(
            label="TestNoiseSFTsOnlyNoise",
            outdir=self.outdir,
            duration=self.Writer.duration,
            Tsft=self.Writer.Tsft,
            tstart=self.Writer.tstart,
            detectors=self.multi_detectors,
            randSeed=randSeed,
            SFTWindowType=self.Writer.SFTWindowType,
            SFTWindowParam=self.Writer.SFTWindowParam,
            sqrtSX=self.Writer.sqrtSX,
            Band=0.5,
            F0=self.signal_parameters["F0"],
        )
        noise_writer.make_data(verbose=True)

        # inject into noise-only SFTs without additional SFT loading constraints
        add_signal_writer = self.writer_class_to_test(
            label="TestNoiseSFTsAddSignal",
            outdir=self.outdir,
            SFTWindowType=self.Writer.SFTWindowType,
            SFTWindowParam=self.Writer.SFTWindowParam,
            noiseSFTs=noise_writer.sftfilepath,
            **(
                {
                    k: v
                    for k, v in self.signal_parameters.items()
                    if not (k.startswith("F") and int(k[-1]) > 2)
                }
                if self.style == "old"
                else {}
            ),
            signal_parameters=self.signal_parameters if self.style == "new" else None,
        )
        add_signal_writer.make_data(verbose=True)

        # same again but with explicit (tstart,duration) to build constraints
        add_signal_writer_constr = self.writer_class_to_test(
            label="TestNoiseSFTsAddSignalWithConstraints",
            outdir=self.outdir,
            duration=self.Writer.duration / 2,
            Tsft=self.Writer.Tsft,
            tstart=self.Writer.tstart,
            SFTWindowType=self.Writer.SFTWindowType,
            SFTWindowParam=self.Writer.SFTWindowParam,
            sqrtSX=0,
            noiseSFTs=noise_writer.sftfilepath,
            **(
                {
                    k: v
                    for k, v in self.signal_parameters.items()
                    if not (k.startswith("F") and int(k[-1]) > 2)
                }
                if self.style == "old"
                else {}
            ),
            signal_parameters=self.signal_parameters if self.style == "new" else None,
        )
        add_signal_writer_constr.make_data(verbose=True)

        (
            noise_and_signal_freqs,
            times,
            noise_and_signal_data,
        ) = pyfstat.utils.get_sft_as_arrays(noise_and_signal_writer.sftfilepath)

        noise_freqs, _, noise_data = pyfstat.utils.get_sft_as_arrays(
            noise_writer.sftfilepath
        )

        (
            add_signal_freqs,
            _,
            add_signal_data,
        ) = pyfstat.utils.get_sft_as_arrays(add_signal_writer.sftfilepath)

        constr_freqs, _, constr_data = pyfstat.utils.get_sft_as_arrays(
            add_signal_writer_constr.sftfilepath
        )

        for ifo in self.multi_detectors.split(","):
            ns_data = np.abs(noise_and_signal_data[ifo])
            max_values_noise_and_signal = np.max(ns_data, axis=0)
            max_freqs_noise_and_signal = noise_and_signal_freqs[
                np.argmax(ns_data, axis=0)
            ]
            assert len(
                times[ifo] == int(np.ceil(self.Writer.duration / self.Writer.Tsft))
            )
            # FIXME: CW signals don't have to peak at the same frequency, but there
            # are some consistency criteria which may be useful to implement here.
            # assert len(np.unique(max_freqs_noise_and_signal) == 1)

            n_data = np.abs(noise_data[ifo])
            max_values_noise = np.max(n_data, axis=0)
            max_freqs_noise = noise_freqs[np.argmax(n_data, axis=0)]
            assert len(max_freqs_noise) == len(max_freqs_noise_and_signal)
            # pure noise: random peak freq in each SFT, lower max values
            assert not (len(np.unique(max_freqs_noise)) == 1)
            assert np.all(max_values_noise < max_values_noise_and_signal)

            as_data = np.abs(add_signal_data[ifo])
            max_values_added_signal = np.max(as_data, axis=0)
            max_freqs_added_signal = add_signal_freqs[np.argmax(as_data, axis=0)]
            assert len(max_freqs_added_signal) == len(max_freqs_noise_and_signal)
            # peak freqs expected exactly equal to first case,
            # peak values can have a bit of numerical diff
            assert np.all(max_freqs_added_signal == max_freqs_noise_and_signal)
            assert np.allclose(
                max_values_added_signal,
                max_values_noise_and_signal,
                rtol=1e-6,
                atol=0,
            )

            c_data = np.abs(constr_data[ifo])
            max_values_added_signal_constr = np.max(c_data, axis=0)
            max_freqs_added_signal_constr = constr_freqs[np.argmax(c_data, axis=0)]
            assert 2 * len(max_freqs_added_signal_constr) == len(
                max_freqs_noise_and_signal
            )
            # peak freqs and values expected to be exactly equal
            # regardless of read-in constraints
            assert np.all(
                max_freqs_added_signal_constr
                == max_freqs_added_signal[: len(max_freqs_added_signal_constr)]
            )
            assert np.all(
                max_values_added_signal_constr
                == max_values_added_signal[: len(max_values_added_signal_constr)]
            )

    def test_noise_sfts_with_gaps(self, default_signal_parameters):
        duration = 10 * self.Writer.duration
        gap_time = 4 * self.Writer.Tsft
        Band = 0.01

        first_chunk_of_data = self.writer_class_to_test(
            label="FirstChunk",
            outdir=self.outdir,
            duration=duration,
            Tsft=self.Writer.Tsft,
            tstart=self.Writer.tstart,
            detectors=self.Writer.detectors,
            SFTWindowType=self.Writer.SFTWindowType,
            SFTWindowParam=self.Writer.SFTWindowParam,
            F0=default_signal_parameters["F0"],
            Band=Band,
        )
        first_chunk_of_data.make_data(verbose=True)

        second_chunk_of_data = self.writer_class_to_test(
            label="SecondChunk",
            outdir=self.outdir,
            duration=duration,
            Tsft=self.Writer.Tsft,
            tstart=self.Writer.tstart + duration + gap_time,
            tref=self.Writer.tstart,
            detectors=self.Writer.detectors,
            SFTWindowType=self.Writer.SFTWindowType,
            SFTWindowParam=self.Writer.SFTWindowParam,
            F0=default_signal_parameters["F0"],
            Band=Band,
        )
        second_chunk_of_data.make_data(verbose=True)

        both_chunks_of_data = self.writer_class_to_test(
            label="BothChunks",
            outdir=self.outdir,
            noiseSFTs=first_chunk_of_data.sftfilepath
            + ";"
            + second_chunk_of_data.sftfilepath,
            SFTWindowType=self.Writer.SFTWindowType,
            SFTWindowParam=self.Writer.SFTWindowParam,
            F0=default_signal_parameters["F0"],
            Band=Band,
        )
        both_chunks_of_data.make_data(verbose=True)

        Tsft = both_chunks_of_data.Tsft
        total_duration = 2 * duration + gap_time
        Nsft = int((total_duration - gap_time) / Tsft)
        expected_SFT_filepath = os.path.join(
            self.outdir,
            "{:1s}-{:d}_{:2s}_{:d}SFT_{:s}-{:d}-{:d}.sft".format(
                self.Writer.detectors[0],
                Nsft,
                self.Writer.detectors,
                Tsft,
                both_chunks_of_data.label,
                self.Writer.tstart,
                total_duration,
            ),
        )
        assert os.path.isfile(
            expected_SFT_filepath
        ), f"Could not find expected SFT file '{expected_SFT_filepath}'!"

    def test_noise_sfts_narrowbanded(self, default_signal_parameters):
        # create some broad SFTs
        writer = self.writer_class_to_test(
            label="TestNoiseSFTsBroad",
            outdir=self.outdir,
            duration=self.Writer.duration,
            Tsft=self.Writer.Tsft,
            tstart=self.Writer.tstart,
            detectors=self.Writer.detectors,
            SFTWindowType=self.Writer.SFTWindowType,
            SFTWindowParam=self.Writer.SFTWindowParam,
            sqrtSX=self.Writer.sqrtSX,
            Band=3,
            F0=default_signal_parameters["F0"],
        )
        writer.make_data(verbose=True)
        # split them by frequency
        cl_split = "lalpulsar_splitSFTs"
        cl_split += " --frequency-bandwidth 1"
        cl_split += f" --start-frequency {writer.fmin}"
        cl_split += f" --end-frequency {writer.fmin + writer.Band}"
        cl_split += f" --output-directory {self.outdir}"
        cl_split += f" -- {writer.sftfilepath}"
        pyfstat.utils.run_commandline(cl_split)
        splitSFTs = os.path.join(self.outdir, "*NB*")
        # reuse the split SFTs as noiseSFTs
        NB_recycling_writer = self.writer_class_to_test(
            label="TestNoiseSFTsRecycle",
            outdir=self.outdir,
            SFTWindowType=self.Writer.SFTWindowType,
            SFTWindowParam=self.Writer.SFTWindowParam,
            noiseSFTs=splitSFTs,
            F0=(
                default_signal_parameters["F0"]
                if self.writer_class_to_test.mfd.endswith("v4")
                else None
            ),
            # **self.signal_parameters, # FIXME this will fail, need MFDv5 fix
        )
        NB_recycling_writer.make_data(verbose=True)
        # manual cleanup in case test is rerun,
        # because fixture's auto cleanup won't catch the splitSFTs output quickly enough
        try:
            for f in glob.glob(splitSFTs):
                os.remove(f)
        except OSError:
            logging.warning(f"Could not clean up all split SFT files: {splitSFTs}")

    def _test_writer_with_tsfiles(self, gaps=False, nanoseconds=False):
        """helper function for timestamps tests

        Can be used to rerun timestamps tests with/without gaps,
        and with new single-column (nanoseconds=False)
        and old two-column (nanoseconds=True)
        formats.
        """
        IFOs = self.multi_detectors.split(",")
        tsfiles = [
            os.path.join(
                self.outdir,
                (
                    f"timestamps_{IFO}"
                    f"{'_gaps' if gaps else ''}"
                    f"{'_ns' if nanoseconds else ''}.txt"
                ),
            )
            for IFO in IFOs
        ]
        numSFTs = []
        for X, tsfile in enumerate(tsfiles):
            with open(tsfile, "w") as fp:
                k = 0
                while k * self.Writer.Tsft < self.Writer.duration:
                    # add gaps at different points for each IFO
                    if not gaps or not k == X + 1:
                        line = (
                            f"{self.Writer.tstart + k * self.Writer.Tsft}"
                            f"{' 0' if nanoseconds else ''}\n"
                        )
                        fp.write(line)
                    k += 1
            if gaps:
                numSFTs.append(k - 1)
            else:
                numSFTs.append(k)
            total_duration = k * self.Writer.Tsft
        tsWriter = self.writer_class_to_test(
            label="TestWriterWithTSFiles",
            Tsft=self.Writer.Tsft,
            outdir=self.outdir,
            sqrtSX=self.Writer.sqrtSX,
            Band=self.Writer.Band,
            detectors=self.multi_detectors,
            SFTWindowType=self.Writer.SFTWindowType,
            SFTWindowParam=self.Writer.SFTWindowParam,
            randSeed=self.Writer.randSeed,
            timestamps=",".join(tsfiles),
            **(
                {
                    k: v
                    for k, v in self.signal_parameters.items()
                    if not (k.startswith("F") and int(k[-1]) > 2)
                }
                if self.style == "old"
                else {}
            ),
            signal_parameters=self.signal_parameters if self.style == "new" else None,
        )
        tsWriter.make_data(verbose=True)
        for X, IFO in enumerate(IFOs):
            expected_outfile = os.path.join(
                tsWriter.outdir,
                "{:1s}-{:d}_{:2s}_{:d}SFT_{:s}-{:d}-{:d}.sft".format(
                    IFO[0],
                    numSFTs[X],
                    IFO,
                    self.Writer.Tsft,
                    tsWriter.label,
                    self.Writer.tstart,
                    total_duration,
                ),
            )
            assert os.path.isfile(expected_outfile)
            assert lalpulsar.ValidateSFTFile(expected_outfile) == 0
        if not gaps:
            # test only first IFO's SFT against standard (tstart,duration) run
            SFTnamesplit = tsWriter.sftfilepath.split(";")[0].split("Test")
            assert self.Writer.sftfilepath.split("Test")[0] == SFTnamesplit[0]
            assert (
                self.Writer.sftfilepath.split("Test")[1].split("-")[1:]
                == SFTnamesplit[1].split("-")[1:]
            )

    def test_timestamps(self):
        for gaps in [False, True]:
            for nanoseconds in [False, True]:
                self._test_writer_with_tsfiles(gaps, nanoseconds)

    def test_timestamps_file_generation(self):
        # Test dictionary
        ts_gap4 = np.arange(
            self.Writer.tstart,
            self.Writer.tstart + 4 * self.Writer.Tsft,
            self.Writer.Tsft,
        )
        ts_gap8 = np.arange(
            self.Writer.tstart,
            self.Writer.tstart + 8 * self.Writer.Tsft,
            self.Writer.Tsft,
        )
        timestamps = {"H1": ts_gap4}
        if "v4" not in self.writer_class_to_test.mfd:
            timestamps["L1"] = ts_gap8

        tsWriter = self.writer_class_to_test(
            label="TimestampsUsingDict",
            Tsft=self.Writer.Tsft,
            outdir=self.Writer.outdir,
            sqrtSX=self.Writer.sqrtSX,
            Band=self.Writer.Band,
            SFTWindowType=self.Writer.SFTWindowType,
            SFTWindowParam=self.Writer.SFTWindowParam,
            randSeed=self.Writer.randSeed,
            timestamps=timestamps,
            **(
                {
                    k: v
                    for k, v in self.signal_parameters.items()
                    if not (k.startswith("F") and int(k[-1]) > 2)
                }
                if self.style == "old"
                else {}
            ),
            signal_parameters=self.signal_parameters if self.style == "new" else None,
        )

        for ifo in timestamps:
            timestamps_file = os.path.join(
                tsWriter.outdir, f"{tsWriter.label}_timestamps_{ifo}.csv"
            )
            assert os.path.isfile(timestamps_file)
            ts = np.genfromtxt(timestamps_file)
            np.testing.assert_almost_equal(ts, timestamps[ifo])

        # Test dictionary with input detector
        timestamps = {"H1": ts_gap4}
        if "v4" not in self.writer_class_to_test.mfd:
            timestamps["L1"] = ts_gap8
        detectors = ",".join(list(timestamps.keys()))

        tsWriter = self.writer_class_to_test(
            label="TimestampsUsingDict",
            Tsft=self.Writer.Tsft,
            outdir=self.Writer.outdir,
            sqrtSX=self.Writer.sqrtSX,
            Band=self.Writer.Band,
            detectors=detectors,
            SFTWindowType=self.Writer.SFTWindowType,
            SFTWindowParam=self.Writer.SFTWindowParam,
            randSeed=self.Writer.randSeed,
            timestamps=timestamps,
            **(
                {
                    k: v
                    for k, v in self.signal_parameters.items()
                    if not (k.startswith("F") and int(k[-1]) > 2)
                }
                if self.style == "old"
                else {}
            ),
            signal_parameters=self.signal_parameters if self.style == "new" else None,
        )

        for ifo in timestamps:
            timestamps_file = os.path.join(
                tsWriter.outdir, f"{tsWriter.label}_timestamps_{ifo}.csv"
            )
            assert os.path.isfile(timestamps_file)
            ts = np.genfromtxt(timestamps_file)
            np.testing.assert_almost_equal(ts, timestamps[ifo])

        # Test single list
        detectors = "H1"
        if "v4" not in self.writer_class_to_test.mfd:
            detectors += ",L1"

        tsWriter = self.writer_class_to_test(
            label="TimestampsUsingList",
            Tsft=self.Writer.Tsft,
            outdir=self.outdir,
            sqrtSX=self.Writer.sqrtSX,
            Band=self.Writer.Band,
            SFTWindowType=self.Writer.SFTWindowType,
            SFTWindowParam=self.Writer.SFTWindowParam,
            randSeed=self.Writer.randSeed,
            detectors=detectors,
            timestamps=ts_gap4,
            **(
                {
                    k: v
                    for k, v in self.signal_parameters.items()
                    if not (k.startswith("F") and int(k[-1]) > 2)
                }
                if self.style == "old"
                else {}
            ),
            signal_parameters=self.signal_parameters if self.style == "new" else None,
        )

        for ifo in detectors.split(","):
            timestamps_file = os.path.join(
                tsWriter.outdir, f"{tsWriter.label}_timestamps_{ifo}.csv"
            )
            assert os.path.isfile(timestamps_file)
            ts = np.genfromtxt(timestamps_file)
            np.testing.assert_almost_equal(ts, ts_gap4)


class TestLineWriter(TestWriter):
    label = "TestLineWriter"
    writer_class_to_test = pyfstat.make_sfts.LineWriter
    signal_parameters = default_signal_params_no_sky
    transient_signal_parameters = signal_parameters | default_transient_params
    multi_detectors = "H1"

    def test_multi_ifo_fails(self):
        detectors = "H1,L1"
        with pytest.raises(NotImplementedError):
            self.writer_class_to_test(
                label="TestNoiseSFTsNoiseAndSignal",
                outdir=self.outdir,
                duration=self.Writer.duration,
                Tsft=self.Writer.Tsft,
                tstart=self.Writer.tstart,
                detectors=detectors,
                sqrtSX=self.Writer.sqrtSX,
                Band=0.5,
                **(
                    {
                        k: v
                        for k, v in self.signal_parameters.items()
                        if not (k.startswith("F") and int(k[-1]) > 2)
                    }
                    if self.style == "old"
                    else {}
                ),
                signal_parameters=(
                    self.signal_parameters if self.style == "new" else None
                ),
            )

    def test_makefakedata_usecached(self):
        # Make everything from scratch
        writer = self.writer_class_to_test(
            outdir=self.outdir,
            **default_Writer_params,
            **(
                {
                    k: v
                    for k, v in self.transient_signal_parameters.items()
                    if not (k.startswith("F") and int(k[-1]) > 2)
                }
                if self.style == "old"
                else {}
            ),
            signal_parameters=(
                self.transient_signal_parameters if self.style == "new" else None
            ),
        )
        writer.make_data(verbose=True)
        first_time = os.path.getmtime(writer.sftfilepath)

        # Re-run, and should be unchanged
        writer.make_data(verbose=True)
        second_time = os.path.getmtime(writer.sftfilepath)
        assert first_time == second_time

        # third run: touch the .cff to force regeneration
        time.sleep(1)  # make sure timestamp is actually different!
        os.system("touch {}".format(writer.config_file_name))
        writer.run_makefakedata()
        third_time = os.path.getmtime(writer.sftfilepath)
        assert not (first_time == third_time)

    def _check_maximum_power_consistency(self, writer):
        freqs, times, data = pyfstat.utils.get_sft_as_arrays(writer.sftfilepath)
        for ifo in times.keys():
            max_power_freq_index = np.argmax(np.abs(data[ifo]), axis=0)
            line_active_mask = (
                writer.signal_parameters["transientStartTime"] <= times[ifo]
            ) & (
                times[ifo]
                < (
                    writer.signal_parameters["transientStartTime"]
                    + writer.signal_parameters["transientTauDays"] * 86400
                )
            )
            max_power_freq_index_with_line = max_power_freq_index[line_active_mask]

            # Maximum power should be at the transient line whenever that's on
            assert np.all(
                max_power_freq_index_with_line == max_power_freq_index_with_line[0]
            )
            assert np.allclose(
                freqs[max_power_freq_index_with_line], writer.F0
            ), f"max SFT power not found at injected line frequency {writer.F0} but at {freqs[max_power_freq_index_with_line]} across IFOs"

    def test_transient_line_injection(self):
        # Create data with a line
        writer = self.writer_class_to_test(
            outdir=self.outdir,
            **default_Writer_params,
            **(
                {
                    k: v
                    for k, v in self.transient_signal_parameters.items()
                    if not (k.startswith("F") and int(k[-1]) > 2)
                }
                if self.style == "old"
                else {}
            ),
            signal_parameters=(
                self.transient_signal_parameters if self.style == "new" else None
            ),
        )
        writer.make_data(verbose=True)

        self._check_maximum_power_consistency(writer)

    def test_noise_sfts(self):
        # Create data with a line
        writer = self.writer_class_to_test(
            outdir=self.outdir,
            **{k: v for k, v in default_Writer_params.items() if k != "sqrtSX"},
            **(
                {
                    k: v
                    for k, v in self.transient_signal_parameters.items()
                    if not (k.startswith("F") and int(k[-1]) > 2)
                }
                if self.style == "old"
                else {}
            ),
            signal_parameters=(
                self.transient_signal_parameters if self.style == "new" else None
            ),
            noiseSFTs=self.Writer.sftfilepath,
        )
        writer.make_data(verbose=True)

        self._check_maximum_power_consistency(writer)


class TestWriterOtherTsft(TestWriter):
    label = "TestWriterOtherTsft"
    writer_class_to_test = pyfstat.Writer

    @classmethod
    def setUpClass(self):
        self.Tsft = 1024
        super().setUpClass()


class TestBinaryModulatedWriter(TestWriter):
    label = "TestBinaryModulatedWriter"
    writer_class_to_test = pyfstat.BinaryModulatedWriter
    signal_parameters = {**default_signal_params, **default_binary_params}

    def test_tp_parsing(self):
        this_writer = self.writer_class_to_test(
            outdir=self.outdir,
            **default_Writer_params,
            **(
                {
                    k: v
                    for k, v in self.signal_parameters.items()
                    if not (k.startswith("F") and int(k[-1]) > 2)
                }
                if self.style == "old"
                else {}
            ),
            signal_parameters=self.signal_parameters if self.style == "new" else None,
        )
        this_writer.make_data()

        theta_prior = {
            key: value
            for key, value in default_signal_params.items()
            if key
            not in [
                "h0",
                "cosi",
                "psi",
                "phi",
                "tref",
            ]
            + [
                f"F{k}" for k in range(3, this_writer.max_fkdot + 1)
            ]  # FIXME when MCMC supports F3 and higher
        }
        theta_prior.update({key: value for key, value in default_binary_params.items()})
        theta_prior["tp"] = {
            "type": "unif",
            "lower": default_binary_params["tp"]
            - 0.5 * default_binary_params["period"],
            "upper": default_binary_params["tp"]
            + 0.5 * default_binary_params["period"],
        }
        theta_prior.pop

        mcmc_kwargs = {
            "nsteps": [50],
            "nwalkers": 150,
            "ntemps": 3,
        }
        print(f"theta_prior: {theta_prior}")
        mcmc = pyfstat.MCMCSearch(
            binary=True,
            label="TpParsing",
            outdir=self.outdir,
            sftfilepattern=this_writer.sftfilepath,
            theta_prior=theta_prior,
            tref=this_writer.tstart,
            minStartTime=this_writer.tstart,
            maxStartTime=this_writer.tend,
            **mcmc_kwargs,
        )
        mcmc.run(plot_walkers=False)
        max_twoF_sample, _ = mcmc.get_max_twoF()

        relative_difference = np.abs(
            1.0 - max_twoF_sample["tp"] / default_binary_params["tp"]
        )
        assert relative_difference < 1e-5


class TestGlitchWriter(TestWriter):
    label = "TestGlitchWriter"
    writer_class_to_test = pyfstat.GlitchWriter

    def test_glitch_injection(self):
        Band = 1
        vanillaWriter = pyfstat.Writer(
            label=self.label + "Vanilla",
            outdir=self.outdir,
            duration=self.Writer.duration,
            tstart=self.Writer.tstart,
            detectors=self.Writer.detectors,
            Band=Band,
            **(
                {
                    k: v
                    for k, v in self.signal_parameters.items()
                    if not (k.startswith("F") and int(k[-1]) > 2)
                }
                if self.style == "old"
                else {}
            ),
            signal_parameters=self.signal_parameters if self.style == "new" else None,
        )
        vanillaWriter.make_cff(verbose=True)
        vanillaWriter.run_makefakedata()
        noGlitchWriter = self.writer_class_to_test(
            label=self.label + "Noglitch",
            outdir=self.outdir,
            duration=self.Writer.duration,
            tstart=self.Writer.tstart,
            detectors=self.Writer.detectors,
            Band=Band,
            **(
                {
                    k: v
                    for k, v in self.signal_parameters.items()
                    if not (k.startswith("F") and int(k[-1]) > 2)
                }
                if self.style == "old"
                else {}
            ),
            signal_parameters=self.signal_parameters if self.style == "new" else None,
        )
        noGlitchWriter.make_cff(verbose=True)
        noGlitchWriter.run_makefakedata()
        glitchWriter = self.writer_class_to_test(
            label=self.label + "Glitch",
            outdir=self.outdir,
            duration=self.Writer.duration,
            tstart=self.Writer.tstart,
            detectors=self.Writer.detectors,
            Band=Band,
            **(
                {
                    k: v
                    for k, v in self.signal_parameters.items()
                    if not (k.startswith("F") and int(k[-1]) > 2)
                }
                if self.style == "old"
                else {}
            ),
            signal_parameters=self.signal_parameters if self.style == "new" else None,
            dtglitch=2 * 1800,
            delta_F0=0.1,
        )
        glitchWriter.make_cff(verbose=True)
        glitchWriter.run_makefakedata()
        (
            freqs_vanilla,
            times_vanilla,
            data_vanilla,
        ) = pyfstat.utils.get_sft_as_arrays(vanillaWriter.sftfilepath)
        (
            freqs_noglitch,
            times_noglitch,
            data_noglitch,
        ) = pyfstat.utils.get_sft_as_arrays(noGlitchWriter.sftfilepath)
        (
            freqs_glitch,
            times_glitch,
            data_glitch,
        ) = pyfstat.utils.get_sft_as_arrays(glitchWriter.sftfilepath)

        for ifo in self.Writer.detectors.split(","):
            max_freq_vanilla = freqs_vanilla[
                np.argmax(np.abs(data_vanilla[ifo]), axis=0)
            ]
            max_freq_noglitch = freqs_noglitch[
                np.argmax(np.abs(data_noglitch[ifo]), axis=0)
            ]
            max_freq_glitch = freqs_glitch[np.argmax(np.abs(data_glitch[ifo]), axis=0)]
            print([max_freq_vanilla, max_freq_noglitch, max_freq_glitch])
            assert np.all(times_noglitch[ifo] == times_vanilla[ifo])
            assert np.all(times_glitch[ifo] == times_vanilla[ifo])
            assert len(np.unique(max_freq_vanilla)) == 1
            assert len(np.unique(max_freq_noglitch)) == 1
            assert len(np.unique(max_freq_glitch)) == 2
            assert max_freq_noglitch[0] == max_freq_vanilla[0]
            assert max_freq_glitch[0] == max_freq_noglitch[0]
            assert max_freq_glitch[-1] > max_freq_noglitch[-1]


@pytest.mark.usefixtures("outdir")
class TestFrequencyModulatedArtifactWriter:
    label = "TestFrequencyModulatedArtifactWriter"

    def test(self):
        writer = pyfstat.FrequencyModulatedArtifactWriter(
            label=self.label,
            duration=3600,
            detectors="H1",
            tref=700000000,
            outdir=self.outdir,
            Band=0.1,
        )
        writer.make_data()
        assert lalpulsar.ValidateSFTFile(writer.sftfilepath) == 0
