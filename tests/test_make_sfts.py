import os
import time

import lalpulsar
import numpy as np
import pytest

# FIXME this should be made cleaner with fixtures
from commons_for_tests import (
    BaseForTestsWithData,
    BaseForTestsWithOutdir,
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

        _, _, log_message = caplog.record_tuples[-1]
        assert "has more than 1 column" in log_message
        assert "will ignore" in log_message

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

        _, _, log_message = caplog.record_tuples[-1]
        assert "non-integer timestamps" in log_message
        assert "floor" in log_message

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


class TestWriter(BaseForTestsWithData):
    label = "TestWriter"
    writer_class_to_test = pyfstat.Writer
    signal_parameters = default_signal_params
    multi_detectors = "H1,L1"  # this needs to be overwritable by child test classes that don't support multi-IFOs

    def test_make_cff(self):
        self.Writer.make_cff(verbose=True)
        self.assertTrue(
            os.path.isfile(os.path.join(".", self.outdir, self.label + ".cff"))
        )

    def test_run_makefakedata(self):
        self.Writer.make_data(verbose=True)
        numSFTs = int(np.ceil(self.duration / self.Tsft))
        expected_outfile = os.path.join(
            self.Writer.outdir,
            "{:1s}-{:d}_{:2s}_{:d}SFT_{:s}-{:d}-{:d}.sft".format(
                self.detectors[0],
                numSFTs,
                self.detectors,
                self.Tsft,
                self.Writer.label,
                self.Writer.tstart,
                numSFTs * self.Tsft,
            ),
        )
        self.assertTrue(os.path.isfile(expected_outfile))
        self.assertTrue(lalpulsar.ValidateSFTFile(expected_outfile) == 0)

    def test_makefakedata_usecached(self):
        if os.path.isfile(self.Writer.config_file_name):
            os.remove(self.Writer.config_file_name)
        if os.path.isfile(self.Writer.sftfilepath):
            os.remove(self.Writer.sftfilepath)

        # first run: make everything from scratch
        self.Writer.make_cff(verbose=True)
        self.Writer.run_makefakedata()
        time_first = os.path.getmtime(self.Writer.sftfilepath)

        # second run: should re-use .cff and .sft
        self.Writer.make_cff(verbose=True)
        self.Writer.run_makefakedata()
        time_second = os.path.getmtime(self.Writer.sftfilepath)
        self.assertTrue(time_first == time_second)

        # third run: touch the .cff to force regeneration
        time.sleep(1)  # make sure timestamp is actually different!
        os.system("touch {}".format(self.Writer.config_file_name))
        self.Writer.run_makefakedata()
        time_third = os.path.getmtime(self.Writer.sftfilepath)
        self.assertFalse(time_first == time_third)

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
            duration=self.duration,
            Tsft=self.Tsft,
            tstart=self.tstart,
            detectors=self.multi_detectors,
            randSeed=randSeed,
            SFTWindowType=self.SFTWindowType,
            SFTWindowParam=self.SFTWindowParam,
            sqrtSX=self.sqrtSX,
            Band=0.5,
            **self.signal_parameters,
        )
        noise_and_signal_writer.make_data(verbose=True)

        # create noise-only SFTs
        noise_writer = self.writer_class_to_test(
            label="TestNoiseSFTsOnlyNoise",
            outdir=self.outdir,
            duration=self.duration,
            Tsft=self.Tsft,
            tstart=self.tstart,
            detectors=self.multi_detectors,
            randSeed=randSeed,
            SFTWindowType=self.SFTWindowType,
            SFTWindowParam=self.SFTWindowParam,
            sqrtSX=self.sqrtSX,
            Band=0.5,
            F0=self.signal_parameters["F0"],
        )
        noise_writer.make_data(verbose=True)

        # inject into noise-only SFTs without additional SFT loading constraints
        add_signal_writer = self.writer_class_to_test(
            label="TestNoiseSFTsAddSignal",
            outdir=self.outdir,
            SFTWindowType=self.SFTWindowType,
            SFTWindowParam=self.SFTWindowParam,
            noiseSFTs=noise_writer.sftfilepath,
            **self.signal_parameters,
        )
        add_signal_writer.make_data(verbose=True)

        # same again but with explicit (tstart,duration) to build constraints
        add_signal_writer_constr = self.writer_class_to_test(
            label="TestNoiseSFTsAddSignalWithConstraints",
            outdir=self.outdir,
            duration=self.duration / 2,
            Tsft=self.Tsft,
            tstart=self.tstart,
            SFTWindowType=self.SFTWindowType,
            SFTWindowParam=self.SFTWindowParam,
            sqrtSX=0,
            noiseSFTs=noise_writer.sftfilepath,
            **self.signal_parameters,
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
            self.assertTrue(len(times[ifo]) == int(np.ceil(self.duration / self.Tsft)))
            # FIXME: CW signals don't have to peak at the same frequency, but there
            # are some consistency criteria which may be useful to implement here.
            # self.assertTrue(len(np.unique(max_freqs_noise_and_signal)) == 1)

            n_data = np.abs(noise_data[ifo])
            max_values_noise = np.max(n_data, axis=0)
            max_freqs_noise = noise_freqs[np.argmax(n_data, axis=0)]
            self.assertEqual(len(max_freqs_noise), len(max_freqs_noise_and_signal))
            # pure noise: random peak freq in each SFT, lower max values
            self.assertFalse(len(np.unique(max_freqs_noise)) == 1)
            self.assertTrue(np.all(max_values_noise < max_values_noise_and_signal))

            as_data = np.abs(add_signal_data[ifo])
            max_values_added_signal = np.max(as_data, axis=0)
            max_freqs_added_signal = add_signal_freqs[np.argmax(as_data, axis=0)]
            self.assertEqual(
                len(max_freqs_added_signal), len(max_freqs_noise_and_signal)
            )
            # peak freqs expected exactly equal to first case,
            # peak values can have a bit of numerical diff
            self.assertTrue(
                np.all(max_freqs_added_signal == max_freqs_noise_and_signal)
            )
            self.assertTrue(
                np.allclose(
                    max_values_added_signal,
                    max_values_noise_and_signal,
                    rtol=1e-6,
                    atol=0,
                )
            )

            c_data = np.abs(constr_data[ifo])
            max_values_added_signal_constr = np.max(c_data, axis=0)
            max_freqs_added_signal_constr = constr_freqs[np.argmax(c_data, axis=0)]
            self.assertEqual(
                2 * len(max_freqs_added_signal_constr), len(max_freqs_noise_and_signal)
            )
            # peak freqs and values expected to be exactly equal
            # regardless of read-in constraints
            self.assertTrue(
                np.all(
                    max_freqs_added_signal_constr
                    == max_freqs_added_signal[: len(max_freqs_added_signal_constr)]
                )
            )
            self.assertTrue(
                np.all(
                    max_values_added_signal_constr
                    == max_values_added_signal[: len(max_values_added_signal_constr)]
                )
            )

    def test_noise_sfts_with_gaps(self):
        duration = 10 * self.Tsft
        gap_time = 4 * self.Tsft
        window = "tukey"
        window_param = 0.01
        Band = 0.01

        first_chunk_of_data = self.writer_class_to_test(
            label="FirstChunk",
            outdir=self.outdir,
            duration=duration,
            Tsft=self.Tsft,
            tstart=self.tstart,
            detectors=self.detectors,
            SFTWindowType=window,
            SFTWindowParam=window_param,
            F0=self.F0,
            Band=Band,
        )
        first_chunk_of_data.make_data(verbose=True)

        second_chunk_of_data = self.writer_class_to_test(
            label="SecondChunk",
            outdir=self.outdir,
            duration=duration,
            Tsft=self.Tsft,
            tstart=self.tstart + duration + gap_time,
            tref=self.tstart,
            detectors=self.detectors,
            SFTWindowType=window,
            SFTWindowParam=window_param,
            F0=self.F0,
            Band=Band,
        )
        second_chunk_of_data.make_data(verbose=True)

        both_chunks_of_data = self.writer_class_to_test(
            label="BothChunks",
            outdir=self.outdir,
            noiseSFTs=first_chunk_of_data.sftfilepath
            + ";"
            + second_chunk_of_data.sftfilepath,
            SFTWindowType=window,
            SFTWindowParam=window_param,
            F0=self.F0,
            Band=Band,
        )
        both_chunks_of_data.make_data(verbose=True)

        Tsft = both_chunks_of_data.Tsft
        total_duration = 2 * duration + gap_time
        Nsft = int((total_duration - gap_time) / Tsft)
        expected_SFT_filepath = os.path.join(
            self.outdir,
            "{:1s}-{:d}_{:2s}_{:d}SFT_{:s}-{:d}-{:d}.sft".format(
                self.detectors[0],
                Nsft,
                self.detectors,
                Tsft,
                both_chunks_of_data.label,
                self.tstart,
                total_duration,
            ),
        )
        self.assertTrue(
            os.path.isfile(expected_SFT_filepath),
            f"Could not find expected SFT file '{expected_SFT_filepath}'!",
        )

    def test_noise_sfts_narrowbanded(self):
        # create some broad SFTs
        writer = self.writer_class_to_test(
            label="TestNoiseSFTsBroad",
            outdir=self.outdir,
            duration=self.duration,
            Tsft=self.Tsft,
            tstart=self.tstart,
            detectors="H1",
            SFTWindowType=self.SFTWindowType,
            SFTWindowParam=self.SFTWindowParam,
            sqrtSX=self.sqrtSX,
            Band=3,
            F0=self.signal_parameters["F0"],
        )
        writer.make_data(verbose=True)
        # split them by frequency
        cl_split = "lalpulsar_splitSFTs"
        cl_split += " --frequency-bandwidth 1"
        cl_split += f" --start-frequency {writer.fmin}"
        cl_split += f" --end-frequency {writer.fmin+writer.Band}"
        cl_split += f" --output-directory {self.outdir}"
        cl_split += f" -- {writer.sftfilepath}"
        pyfstat.utils.run_commandline(cl_split)
        # reuse the split SFTs as noiseSFTs
        NB_recycling_writer = self.writer_class_to_test(
            label="TestNoiseSFTsRecycle",
            outdir=self.outdir,
            SFTWindowType=self.SFTWindowType,
            SFTWindowParam=self.SFTWindowParam,
            noiseSFTs=os.path.join(self.outdir, "*NB*"),
            F0=(
                self.signal_parameters["F0"]
                if self.writer_class_to_test.mfd.endswith("v4")
                else None
            ),
            # **self.signal_parameters, # FIXME this will fail, need MFDv5 fix
        )
        NB_recycling_writer.make_data(verbose=True)

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
                f"timestamps_{IFO}{'_gaps' if gaps else ''}{'_ns' if nanoseconds else ''}.txt",
            )
            for IFO in IFOs
        ]
        numSFTs = []
        for X, tsfile in enumerate(tsfiles):
            with open(tsfile, "w") as fp:
                k = 0
                while k * self.Tsft < self.duration:
                    if (
                        not gaps or not k == X + 1
                    ):  # add gaps at different points for each IFO
                        line = f"{self.tstart + k*self.Tsft}{' 0' if nanoseconds else ''}\n"
                        fp.write(line)
                    k += 1
            if gaps:
                numSFTs.append(k - 1)
            else:
                numSFTs.append(k)
            total_duration = k * self.Tsft
        tsWriter = self.writer_class_to_test(
            label="TestWriterWithTSFiles",
            tref=self.tref,
            Tsft=self.Tsft,
            outdir=self.outdir,
            sqrtSX=self.sqrtSX,
            Band=self.Band,
            detectors=self.multi_detectors,
            SFTWindowType=self.SFTWindowType,
            SFTWindowParam=self.SFTWindowParam,
            randSeed=self.randSeed,
            timestamps=",".join(tsfiles),
            **self.signal_parameters,
        )
        tsWriter.make_data(verbose=True)
        for X, IFO in enumerate(IFOs):
            expected_outfile = os.path.join(
                tsWriter.outdir,
                "{:1s}-{:d}_{:2s}_{:d}SFT_{:s}-{:d}-{:d}.sft".format(
                    IFO[0],
                    numSFTs[X],
                    IFO,
                    self.Tsft,
                    tsWriter.label,
                    self.tstart,
                    total_duration,
                ),
            )
            self.assertTrue(os.path.isfile(expected_outfile))
            self.assertTrue(lalpulsar.ValidateSFTFile(expected_outfile) == 0)
        if not gaps:
            # test only first IFO's SFT against standard (tstart,duration) run
            SFTnamesplit = tsWriter.sftfilepath.split(";")[0].split("Test")
            self.assertTrue(self.Writer.sftfilepath.split("Test")[0] == SFTnamesplit[0])
            self.assertTrue(
                self.Writer.sftfilepath.split("Test")[1].split("-")[1:]
                == SFTnamesplit[1].split("-")[1:]
            )

    def test_timestamps(self):
        for gaps in [False, True]:
            for nanoseconds in [False, True]:
                self._test_writer_with_tsfiles(gaps, nanoseconds)

    def test_timestamps_file_generation(self):
        # Test dictionary
        timestamps = {"H1": np.arange(self.tref, self.tref + 4 * self.Tsft, self.Tsft)}
        if "v4" not in self.writer_class_to_test.mfd:
            timestamps["L1"] = np.arange(
                self.tref, self.tref + 8 * self.Tsft, self.Tsft
            )

        tsWriter = self.writer_class_to_test(
            label="TimestampsUsingDict",
            tref=self.tref,
            Tsft=self.Tsft,
            outdir=self.outdir,
            sqrtSX=self.sqrtSX,
            Band=self.Band,
            SFTWindowType=self.SFTWindowType,
            SFTWindowParam=self.SFTWindowParam,
            randSeed=self.randSeed,
            timestamps=timestamps,
            **self.signal_parameters,
        )

        for ifo in timestamps:
            timestamps_file = os.path.join(
                tsWriter.outdir, f"{tsWriter.label}_timestamps_{ifo}.csv"
            )
            self.assertTrue(os.path.isfile(timestamps_file))
            ts = np.genfromtxt(timestamps_file)
            np.testing.assert_almost_equal(ts, timestamps[ifo])

        # Test dictionary with input detector
        timestamps = {"H1": np.arange(self.tref, self.tref + 4 * self.Tsft, self.Tsft)}
        if "v4" not in self.writer_class_to_test.mfd:
            timestamps["L1"] = np.arange(
                self.tref, self.tref + 8 * self.Tsft, self.Tsft
            )
        detectors = ",".join(list(timestamps.keys()))

        tsWriter = self.writer_class_to_test(
            label="TimestampsUsingDict",
            tref=self.tref,
            Tsft=self.Tsft,
            outdir=self.outdir,
            sqrtSX=self.sqrtSX,
            Band=self.Band,
            detectors=detectors,
            SFTWindowType=self.SFTWindowType,
            SFTWindowParam=self.SFTWindowParam,
            randSeed=self.randSeed,
            timestamps=timestamps,
            **self.signal_parameters,
        )

        for ifo in timestamps:
            timestamps_file = os.path.join(
                tsWriter.outdir, f"{tsWriter.label}_timestamps_{ifo}.csv"
            )
            self.assertTrue(os.path.isfile(timestamps_file))
            ts = np.genfromtxt(timestamps_file)
            np.testing.assert_almost_equal(ts, timestamps[ifo])

        # Test single list
        detectors = "H1"
        timestamps = np.arange(self.tref, self.tref + 4 * self.Tsft, self.Tsft)
        if "v4" not in self.writer_class_to_test.mfd:
            detectors += ",L1"

        tsWriter = self.writer_class_to_test(
            label="TimestampsUsingList",
            tref=self.tref,
            Tsft=self.Tsft,
            outdir=self.outdir,
            sqrtSX=self.sqrtSX,
            Band=self.Band,
            SFTWindowType=self.SFTWindowType,
            SFTWindowParam=self.SFTWindowParam,
            randSeed=self.randSeed,
            detectors=detectors,
            timestamps=timestamps,
            **self.signal_parameters,
        )

        for ifo in detectors.split(","):
            timestamps_file = os.path.join(
                tsWriter.outdir, f"{tsWriter.label}_timestamps_{ifo}.csv"
            )
            self.assertTrue(os.path.isfile(timestamps_file))
            ts = np.genfromtxt(timestamps_file)
            np.testing.assert_almost_equal(ts, timestamps)


class TestLineWriter(TestWriter):
    label = "TestLineWriter"
    writer_class_to_test = pyfstat.make_sfts.LineWriter
    signal_parameters = default_signal_params_no_sky
    multi_detectors = "H1"

    def test_multi_ifo_fails(self):
        detectors = "H1,L1"
        with pytest.raises(NotImplementedError):
            self.writer_class_to_test(
                label="TestNoiseSFTsNoiseAndSignal",
                outdir=self.outdir,
                duration=self.duration,
                Tsft=self.Tsft,
                tstart=self.tstart,
                detectors=detectors,
                sqrtSX=self.sqrtSX,
                Band=0.5,
                **self.signal_parameters,
            )

    def test_makefakedata_usecached(self):
        # Make everything from scratch
        writer = self.writer_class_to_test(
            outdir=self.outdir,
            **default_Writer_params,
            **default_signal_params,
            **default_transient_params,
        )
        writer.make_data(verbose=True)
        first_time = os.path.getmtime(writer.sftfilepath)

        # Re-run, and should be unchanged
        writer.make_data(verbose=True)
        second_time = os.path.getmtime(writer.sftfilepath)
        self.assertTrue(first_time == second_time)

        # third run: touch the .cff to force regeneration
        time.sleep(1)  # make sure timestamp is actually different!
        os.system("touch {}".format(writer.config_file_name))
        writer.run_makefakedata()
        third_time = os.path.getmtime(writer.sftfilepath)
        self.assertFalse(first_time == third_time)

    def _check_maximum_power_consistency(self, writer):
        freqs, times, data = pyfstat.utils.get_sft_as_arrays(writer.sftfilepath)
        for ifo in times.keys():
            max_power_freq_index = np.argmax(np.abs(data[ifo]), axis=0)
            line_active_mask = (writer.transientStartTime <= times[ifo]) & (
                times[ifo] < (writer.transientStartTime + writer.transientTau)
            )
            max_power_freq_index_with_line = max_power_freq_index[line_active_mask]

            # Maximum power should be a the transient line whenever that's on
            self.assertTrue(
                np.all(
                    max_power_freq_index_with_line == max_power_freq_index_with_line[0]
                )
            )
            self.assertTrue(
                np.allclose(freqs[max_power_freq_index_with_line], writer.F0)
            )

    def test_transient_line_injection(self):
        # Create data with a line
        writer = self.writer_class_to_test(
            outdir=self.outdir,
            **default_Writer_params,
            **default_signal_params,
            **default_transient_params,
        )
        writer.make_data(verbose=True)

        self._check_maximum_power_consistency(writer)

    def test_noise_sfts(self):
        # Create data with a line
        writer = self.writer_class_to_test(
            outdir=self.outdir,
            **default_signal_params,
            **default_transient_params,
            SFTWindowType="tukey",
            SFTWindowParam=0.001,
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
            outdir=self.outdir, **default_Writer_params, **self.signal_parameters
        )
        this_writer.make_data()

        theta_prior = {
            key: value
            for key, value in default_signal_params.items()
            if key not in ["h0", "cosi", "psi", "phi"]
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
        print(theta_prior)
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
        self.assertTrue(relative_difference < 1e-5)


class TestGlitchWriter(TestWriter):
    label = "TestGlitchWriter"
    writer_class_to_test = pyfstat.GlitchWriter

    def test_glitch_injection(self):
        Band = 1
        vanillaWriter = pyfstat.Writer(
            label=self.label + "Vanilla",
            outdir=self.outdir,
            duration=self.duration,
            tstart=self.tstart,
            detectors=self.detectors,
            Band=Band,
            **default_signal_params,
        )
        vanillaWriter.make_cff(verbose=True)
        vanillaWriter.run_makefakedata()
        noGlitchWriter = self.writer_class_to_test(
            label=self.label + "Noglitch",
            outdir=self.outdir,
            duration=self.duration,
            tstart=self.tstart,
            detectors=self.detectors,
            Band=Band,
            **default_signal_params,
        )
        noGlitchWriter.make_cff(verbose=True)
        noGlitchWriter.run_makefakedata()
        glitchWriter = self.writer_class_to_test(
            label=self.label + "Glitch",
            outdir=self.outdir,
            duration=self.duration,
            tstart=self.tstart,
            detectors=self.detectors,
            Band=Band,
            **default_signal_params,
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

        for ifo in self.detectors.split(","):
            max_freq_vanilla = freqs_vanilla[
                np.argmax(np.abs(data_vanilla[ifo]), axis=0)
            ]
            max_freq_noglitch = freqs_noglitch[
                np.argmax(np.abs(data_noglitch[ifo]), axis=0)
            ]
            max_freq_glitch = freqs_glitch[np.argmax(np.abs(data_glitch[ifo]), axis=0)]
            print([max_freq_vanilla, max_freq_noglitch, max_freq_glitch])
            self.assertTrue(np.all(times_noglitch[ifo] == times_vanilla[ifo]))
            self.assertTrue(np.all(times_glitch[ifo] == times_vanilla[ifo]))
            self.assertEqual(len(np.unique(max_freq_vanilla)), 1)
            self.assertEqual(len(np.unique(max_freq_noglitch)), 1)
            self.assertEqual(len(np.unique(max_freq_glitch)), 2)
            self.assertEqual(max_freq_noglitch[0], max_freq_vanilla[0])
            self.assertEqual(max_freq_glitch[0], max_freq_noglitch[0])
            self.assertTrue(max_freq_glitch[-1] > max_freq_noglitch[-1])


class TestFrequencyModulatedArtifactWriter(BaseForTestsWithOutdir):
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
        self.assertTrue(lalpulsar.ValidateSFTFile(writer.sftfilepath) == 0)
