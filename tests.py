import unittest
import numpy as np
import os
import shutil
import pyfstat
import lalpulsar
import logging
import pytest
import time
from scipy.stats import chi2


class BaseForTestsWithOutdir(unittest.TestCase):
    outdir = "TestData"

    @classmethod
    def setUpClass(self):
        # ensure a clean working directory
        if os.path.isdir(self.outdir):
            try:
                shutil.rmtree(self.outdir)
            except OSError:
                logging.warning("{} not removed prior to tests".format(self.outdir))
        os.makedirs(self.outdir, exist_ok=True)

    @classmethod
    def tearDownClass(self):
        if os.path.isdir(self.outdir):
            try:
                shutil.rmtree(self.outdir)
            except OSError:
                logging.warning("{} not removed after tests".format(self.outdir))


default_Writer_params = {
    "label": "test",
    "sqrtSX": 1,
    "Tsft": 1800,
    "tstart": 700000000,
    "duration": 4 * 1800,
    "detectors": "H1",
    "SFTWindowType": "tukey",
    "SFTWindowBeta": 0.001,
    "randSeed": 42,
    "Band": None,
}

default_signal_params_no_sky = {
    "F0": 30.0,
    "F1": -1e-10,
    "F2": 0,
    "h0": 5.0,
    "cosi": 0,
    "psi": 0,
    "phi": 0,
}

default_signal_params = {
    **default_signal_params_no_sky,
    **{"Alpha": 5e-1, "Delta": 1.2},
}


default_binary_params = {
    "period": 45 * 24 * 3600.0,
    "asini": 10.0,
    "tp": default_Writer_params["tstart"] + 0.25 * default_Writer_params["duration"],
    "ecc": 0.5,
    "argp": 0.3,
}

default_transient_params = {
    "transientWindowType": "rect",
    "transientStartTime": default_Writer_params["Tsft"]
    + default_Writer_params["tstart"],
    "transientTau": 2 * default_Writer_params["Tsft"],
}


class BaseForTestsWithData(BaseForTestsWithOutdir):
    outdir = "TestData"

    @classmethod
    def setUpClass(self):
        # ensure a clean working directory
        if os.path.isdir(self.outdir):
            try:
                shutil.rmtree(self.outdir)
            except OSError:
                logging.warning("{} not removed prior to tests".format(self.outdir))
        # skip making outdir, since Writer should do so on first call
        # os.makedirs(self.outdir, exist_ok=True)

        # create fake data SFTs
        # if we directly set any options as self.xy = 1 here,
        # then values set for derived classes may get overwritten,
        # so use a default dict and only insert if no value previous set
        for key, val in {**default_Writer_params, **default_signal_params}.items():
            if not hasattr(self, key):
                setattr(self, key, val)
        self.tref = self.tstart
        self.Writer = pyfstat.Writer(
            label=self.label,
            tstart=self.tstart,
            duration=self.duration,
            tref=self.tref,
            F0=self.F0,
            F1=self.F1,
            F2=self.F2,
            Alpha=self.Alpha,
            Delta=self.Delta,
            h0=self.h0,
            cosi=self.cosi,
            Tsft=self.Tsft,
            outdir=self.outdir,
            sqrtSX=self.sqrtSX,
            Band=self.Band,
            detectors=self.detectors,
            SFTWindowType=self.SFTWindowType,
            SFTWindowBeta=self.SFTWindowBeta,
            randSeed=self.randSeed,
        )
        self.Writer.make_data(verbose=True)
        self.search_keys = ["F0", "F1", "F2", "Alpha", "Delta"]
        self.search_ranges = {key: [getattr(self, key)] for key in self.search_keys}


class TestInjectionParametersGenerator(BaseForTestsWithOutdir):
    label = "TestInjectionParametersGenerator"
    class_to_test = pyfstat.InjectionParametersGenerator

    def test_numpy_priors(self):
        numpy_priors = {
            "ParameterA": {"uniform": {"low": 0.0, "high": 0.0}},
            "ParameterB": {"uniform": {"low": 1.0, "high": 1.0}},
        }
        InjectionGenerator = self.class_to_test(numpy_priors)

        parameters = InjectionGenerator.draw()
        self.assertTrue(parameters["ParameterA"] == 0.0)
        self.assertTrue(parameters["ParameterB"] == 1.0)

    def test_callable_priors(self):
        callable_priors = {"ParameterA": lambda: 0.0, "ParameterB": lambda: 1.0}
        InjectionGenerator = self.class_to_test(callable_priors)

        parameters = InjectionGenerator.draw()
        self.assertTrue(parameters["ParameterA"] == 0.0)
        self.assertTrue(parameters["ParameterB"] == 1.0)

    def test_constant_priors(self):
        constant_priors = {"ParameterA": 0.0, "ParameterB": 1.0}
        InjectionGenerator = self.class_to_test(constant_priors)

        parameters = InjectionGenerator.draw()
        self.assertTrue(parameters["ParameterA"] == 0.0)
        self.assertTrue(parameters["ParameterB"] == 1.0)

    def test_rng_seed(self):
        a_seed = 420

        samples = []
        for i in range(2):
            InjectionGenerator = self.class_to_test(
                priors={"ParameterA": {"normal": {"loc": 0, "scale": 1}}}, seed=a_seed
            )
            samples.append(InjectionGenerator.draw())
        self.assertTrue(samples[0]["ParameterA"] == samples[1]["ParameterA"])

    def test_rng_generation(self):
        InjectionGenerator = self.class_to_test(
            priors={"ParameterA": {"normal": {"loc": 0, "scale": 0.01}}}
        )
        samples = [InjectionGenerator.draw()["ParameterA"] for i in range(100)]
        mean = np.mean(samples)
        self.assertTrue(np.abs(mean) < 0.1)


class TestAllSkyInjectionParametersGenerator(TestInjectionParametersGenerator):
    label = "TestAllInjectionParametersGenerator"

    class_to_test = pyfstat.AllSkyInjectionParametersGenerator

    def test_rng_seed(self):
        a_seed = 420

        samples = []
        for i in range(2):
            InjectionGenerator = self.class_to_test(seed=a_seed)
            samples.append(InjectionGenerator.draw())
        self.assertTrue(samples[0]["Alpha"] == samples[1]["Alpha"])
        self.assertTrue(samples[0]["Delta"] == samples[1]["Delta"])

    def test_rng_generation(self):
        InjectionGenerator = self.class_to_test()
        ra_samples = [
            InjectionGenerator.draw()["Alpha"] / np.pi - 1.0 for i in range(500)
        ]
        dec_samples = [
            InjectionGenerator.draw()["Delta"] * 2.0 / np.pi for i in range(500)
        ]
        self.assertTrue(np.abs(np.mean(ra_samples)) < 0.1)
        self.assertTrue(np.abs(np.mean(dec_samples)) < 0.1)


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
            label="test_noiseSFTs_noise_and_signal",
            outdir=self.outdir,
            duration=self.duration,
            Tsft=self.Tsft,
            tstart=self.tstart,
            detectors=self.multi_detectors,
            randSeed=randSeed,
            SFTWindowType=self.SFTWindowType,
            SFTWindowBeta=self.SFTWindowBeta,
            sqrtSX=self.sqrtSX,
            Band=0.5,
            **self.signal_parameters,
        )
        noise_and_signal_writer.make_data(verbose=True)

        # create noise-only SFTs
        noise_writer = self.writer_class_to_test(
            label="test_noiseSFTs_only_noise",
            outdir=self.outdir,
            duration=self.duration,
            Tsft=self.Tsft,
            tstart=self.tstart,
            detectors=self.multi_detectors,
            randSeed=randSeed,
            SFTWindowType=self.SFTWindowType,
            SFTWindowBeta=self.SFTWindowBeta,
            sqrtSX=self.sqrtSX,
            Band=0.5,
            F0=self.signal_parameters["F0"],
        )
        noise_writer.make_data(verbose=True)

        # inject into noise-only SFTs without additional SFT loading constraints
        add_signal_writer = self.writer_class_to_test(
            label="test_noiseSFTs_add_signal",
            outdir=self.outdir,
            SFTWindowType=self.SFTWindowType,
            SFTWindowBeta=self.SFTWindowBeta,
            noiseSFTs=noise_writer.sftfilepath,
            **self.signal_parameters,
        )
        add_signal_writer.make_data(verbose=True)

        # same again but with explicit (tstart,duration) to build constraints
        add_signal_writer_constr = self.writer_class_to_test(
            label="test_noiseSFTs_add_signal_with_constraints",
            outdir=self.outdir,
            duration=self.duration / 2,
            Tsft=self.Tsft,
            tstart=self.tstart,
            SFTWindowType=self.SFTWindowType,
            SFTWindowBeta=self.SFTWindowBeta,
            sqrtSX=0,
            noiseSFTs=noise_writer.sftfilepath,
            **self.signal_parameters,
        )
        add_signal_writer_constr.make_data(verbose=True)

        (
            noise_and_signal_freqs,
            times,
            noise_and_signal_data,
        ) = pyfstat.helper_functions.get_sft_as_arrays(
            noise_and_signal_writer.sftfilepath
        )

        noise_freqs, _, noise_data = pyfstat.helper_functions.get_sft_as_arrays(
            noise_writer.sftfilepath
        )

        (
            add_signal_freqs,
            _,
            add_signal_data,
        ) = pyfstat.helper_functions.get_sft_as_arrays(add_signal_writer.sftfilepath)

        constr_freqs, _, constr_data = pyfstat.helper_functions.get_sft_as_arrays(
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
        window_beta = 0.01
        Band = 0.01

        first_chunk_of_data = self.writer_class_to_test(
            label="first_chunk_of_data",
            outdir=self.outdir,
            duration=duration,
            Tsft=self.Tsft,
            tstart=self.tstart,
            detectors=self.detectors,
            SFTWindowType=window,
            SFTWindowBeta=window_beta,
            F0=self.F0,
            Band=Band,
        )
        first_chunk_of_data.make_data(verbose=True)

        second_chunk_of_data = self.writer_class_to_test(
            label="second_chunk_of_data",
            outdir=self.outdir,
            duration=duration,
            Tsft=self.Tsft,
            tstart=self.tstart + duration + gap_time,
            tref=self.tstart,
            detectors=self.detectors,
            SFTWindowType=window,
            SFTWindowBeta=window_beta,
            F0=self.F0,
            Band=Band,
        )
        second_chunk_of_data.make_data(verbose=True)

        both_chunks_of_data = self.writer_class_to_test(
            label="both_chunks_of_data",
            outdir=self.outdir,
            noiseSFTs=first_chunk_of_data.sftfilepath
            + ";"
            + second_chunk_of_data.sftfilepath,
            SFTWindowType=window,
            SFTWindowBeta=window_beta,
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
        self.assertTrue(os.path.isfile(expected_SFT_filepath))

    def test_noise_sfts_narrowbanded(self):

        # create some broad SFTs
        writer = self.writer_class_to_test(
            label="test_noiseSFTs_broad",
            outdir=self.outdir,
            duration=self.duration,
            Tsft=self.Tsft,
            tstart=self.tstart,
            detectors="H1",
            SFTWindowType=self.SFTWindowType,
            SFTWindowBeta=self.SFTWindowBeta,
            sqrtSX=self.sqrtSX,
            Band=3,
            F0=self.signal_parameters["F0"],
        )
        writer.make_data(verbose=True)
        # split them by frequency
        cl_split = "lalapps_splitSFTs --frequency-bandwidth 1"
        cl_split += f" --start-frequency {writer.fmin}"
        cl_split += f" --end-frequency {writer.fmin+writer.Band}"
        cl_split += f" --output-directory {self.outdir}"
        cl_split += f" -- {writer.sftfilepath}"
        pyfstat.helper_functions.run_commandline(cl_split)
        # reuse the split SFTs as noiseSFTs
        NB_recycling_writer = self.writer_class_to_test(
            label="test_noiseSFTs_recycle",
            outdir=self.outdir,
            SFTWindowType=self.SFTWindowType,
            SFTWindowBeta=self.SFTWindowBeta,
            noiseSFTs=os.path.join(self.outdir, "*NB*"),
            F0=self.signal_parameters["F0"]
            if self.writer_class_to_test.mfd.endswith("v4")
            else None,
            # **self.signal_parameters, # FIXME this will fail, need MFDv5 fix
        )
        NB_recycling_writer.make_data(verbose=True)

    def _test_writer_with_tsfiles(self, gaps=False):
        """helper function to rerun with/without gaps"""
        IFOs = self.multi_detectors.split(",")
        tsfiles = [os.path.join(self.outdir, f"timestamps_{IFO}.txt") for IFO in IFOs]
        numSFTs = []
        for X, tsfile in enumerate(tsfiles):
            with open(tsfile, "w") as fp:
                k = 0
                while k * self.Tsft < self.duration:
                    if (
                        not gaps or not k == X + 1
                    ):  # add gaps at different points for each IFO
                        fp.write(f"{self.tstart + k*self.Tsft} 0\n")
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
            SFTWindowBeta=self.SFTWindowBeta,
            randSeed=self.randSeed,
            timestampsFiles=",".join(tsfiles),
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

    def test_timestampsFiles(self):
        self._test_writer_with_tsfiles(gaps=False)
        self._test_writer_with_tsfiles(gaps=True)


class TestLineWriter(TestWriter):
    label = "TestLineWriter"
    writer_class_to_test = pyfstat.make_sfts.LineWriter
    signal_parameters = default_signal_params_no_sky
    multi_detectors = "H1"

    def test_multi_ifo_fails(self):
        detectors = "H1,L1"
        with pytest.raises(NotImplementedError):
            self.writer_class_to_test(
                label="test_noiseSFTs_noise_and_signal",
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
        freqs, times, data = pyfstat.helper_functions.get_sft_as_arrays(
            writer.sftfilepath
        )
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
            SFTWindowBeta=0.001,
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
            label="tp_parsing",
            outdir=self.outdir,
            sftfilepattern=this_writer.sftfilepath,
            theta_prior=theta_prior,
            tref=this_writer.tstart,
            minStartTime=this_writer.tstart,
            maxStartTime=this_writer.tend(),
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
            label=self.label + "_vanilla",
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
            label=self.label + "_noglitch",
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
            label=self.label + "_glitch",
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
        ) = pyfstat.helper_functions.get_sft_as_arrays(vanillaWriter.sftfilepath)
        (
            freqs_noglitch,
            times_noglitch,
            data_noglitch,
        ) = pyfstat.helper_functions.get_sft_as_arrays(noGlitchWriter.sftfilepath)
        (
            freqs_glitch,
            times_glitch,
            data_glitch,
        ) = pyfstat.helper_functions.get_sft_as_arrays(glitchWriter.sftfilepath)

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


class TestReadParFile(BaseForTestsWithOutdir):
    label = "TestReadParFile"

    def test(self):
        parfile = os.path.join(self.outdir, self.label + ".par")
        os.system('echo "x=100\ny=10" > ' + parfile)

        par = pyfstat.helper_functions.read_par(filename=parfile)
        self.assertTrue(par["x"] == 100)
        self.assertTrue(par["y"] == 10)

        par = pyfstat.helper_functions.read_par(outdir=self.outdir, label=self.label)
        self.assertTrue(par["x"] == 100)
        self.assertTrue(par["y"] == 10)


class TestPredictFstat(BaseForTestsWithOutdir):
    label = "TestPredictFstat"
    # here we only test the modes WITHOUT sftfilepattern,
    # which itself is tested through the Writer and Search classes

    def test_PFS_noise(self):
        twoF_expected, twoF_sigma = pyfstat.helper_functions.predict_fstat(
            minStartTime=default_Writer_params["tstart"],
            duration=default_Writer_params["duration"],
            IFOs=default_Writer_params["detectors"],
            assumeSqrtSX=1,
        )
        print(
            "predict_fstat() returned: E[2F]={}+-{}".format(twoF_expected, twoF_sigma)
        )
        self.assertTrue(twoF_expected == 4)
        self.assertAlmostEqual(twoF_sigma, chi2.std(df=4), places=5)

    def test_PFS_noise_TSfiles(self):
        IFOs = ["H1", "L1"]
        TSfiles = [
            os.path.join(self.outdir, "{:s}_{:s}.ts".format(self.label, IFO))
            for IFO in IFOs
        ]
        for f in TSfiles:
            with open(f, "w") as fp:
                fp.write(
                    "{:d} 0\n{:d} 0\n".format(
                        default_Writer_params["tstart"],
                        default_Writer_params["tstart"] + default_Writer_params["Tsft"],
                    )
                )
        twoF_expected, twoF_sigma = pyfstat.helper_functions.predict_fstat(
            timestampsFiles=",".join(TSfiles),
            IFOs=",".join(IFOs),
            assumeSqrtSX=1,
        )
        print(
            "predict_fstat() returned: E[2F]={}+-{}".format(twoF_expected, twoF_sigma)
        )
        self.assertTrue(twoF_expected == 4)
        self.assertAlmostEqual(twoF_sigma, chi2.std(df=4), places=5)

    def test_PFS_signal(self):
        duration = 10 * default_Writer_params["duration"]
        twoF_expected, twoF_sigma = pyfstat.helper_functions.predict_fstat(
            h0=1,
            cosi=0,
            psi=0,
            Alpha=0,
            Delta=0,
            minStartTime=default_Writer_params["tstart"],
            duration=duration,
            IFOs=default_Writer_params["detectors"],
            assumeSqrtSX=1,
        )
        print("predict_fstat() returned:" f" E[2F]={twoF_expected}+-{twoF_sigma}")
        self.assertTrue(twoF_expected > 4)
        self.assertTrue(twoF_sigma > 0)
        # call again but this time using a dictionary of parameters
        params = {
            "h0": 1,
            "cosi": 0,
            "psi": 0,
            "Alpha": 0,
            "Delta": 0,
            "F0": 0,
            "F1": 0,
        }
        params = pyfstat.helper_functions.get_predict_fstat_parameters_from_dict(params)
        twoF_expected_dict, twoF_sigma_dict = pyfstat.helper_functions.predict_fstat(
            **params,
            minStartTime=default_Writer_params["tstart"],
            duration=duration,
            IFOs=default_Writer_params["detectors"],
            assumeSqrtSX=1,
        )
        print(
            "predict_fstat() called with a dict returned:"
            f" E[2F]={twoF_expected_dict}+-{twoF_sigma_dict}"
        )
        self.assertEqual(twoF_expected_dict, twoF_expected)
        # add transient parameters
        params["transientWindowType"] = "rect"
        params["transient_tstart"] = default_Writer_params["tstart"]
        params["transient_duration"] = 0.5 * duration
        params = pyfstat.helper_functions.get_predict_fstat_parameters_from_dict(params)
        (
            twoF_expected_transient,
            twoF_sigma_transient,
        ) = pyfstat.helper_functions.predict_fstat(
            **params,
            minStartTime=default_Writer_params["tstart"],
            duration=duration,
            IFOs=default_Writer_params["detectors"],
            assumeSqrtSX=1,
        )
        print(
            "predict_fstat() called with a dict including a transient returned:"
            f" E[2F]={twoF_expected_transient}+-{twoF_sigma_transient}"
        )
        self.assertTrue(twoF_expected_transient < twoF_expected)


class TestBaseSearchClass(unittest.TestCase):
    # TODO test the basic methods
    pass


class TestComputeFstat(BaseForTestsWithData):
    label = "TestComputeFstat"

    def test_run_computefstatistic_single_point_injectSqrtSX(self):
        # not using any SFTs
        search = pyfstat.ComputeFstat(
            tref=self.tref,
            minStartTime=self.tstart,
            maxStartTime=self.tstart + self.duration,
            detectors=self.detectors,
            injectSqrtSX=self.sqrtSX,
            minCoverFreq=self.F0 - 0.1,
            maxCoverFreq=self.F0 + 0.1,
        )
        FS = search.get_fullycoherent_twoF(
            F0=self.F0,
            F1=self.F1,
            F2=self.F2,
            Alpha=self.Alpha,
            Delta=self.Delta,
        )
        self.assertTrue(FS > 0.0)

    def test_run_computefstatistic_single_point_with_SFTs(self):

        twoF_predicted = self.Writer.predict_fstat()

        search = pyfstat.ComputeFstat(
            tref=self.Writer.tref,
            sftfilepattern=self.Writer.sftfilepath,
            search_ranges=self.search_ranges,
        )
        twoF = search.get_fullycoherent_twoF(
            F0=self.Writer.F0,
            F1=self.Writer.F1,
            F2=self.Writer.F2,
            Alpha=self.Writer.Alpha,
            Delta=self.Writer.Delta,
        )
        diff = np.abs(twoF - twoF_predicted) / twoF_predicted
        print(
            (
                "Predicted twoF is {}"
                " while recovered value is {},"
                " relative difference: {}".format(twoF_predicted, twoF, diff)
            )
        )
        self.assertTrue(diff < 0.3)

        # the following seems to be a leftover from when this test case was
        # doing separate H1 vs H1,L1 searches, but now only really tests the
        # SSBprec. But well, it still does add a tiny bit of coverage, can still
        # be replaced by something more systematic later.
        search = pyfstat.ComputeFstat(
            tref=self.Writer.tref,
            detectors=self.Writer.detectors,
            sftfilepattern=self.Writer.sftfilepath,
            SSBprec=lalpulsar.SSBPREC_RELATIVISTIC,
            search_ranges=self.search_ranges,
        )
        twoF2 = search.get_fullycoherent_twoF(
            F0=self.Writer.F0,
            F1=self.Writer.F1,
            F2=self.Writer.F2,
            Alpha=self.Writer.Alpha,
            Delta=self.Writer.Delta,
        )
        diff = np.abs(twoF2 - twoF_predicted) / twoF_predicted
        print(
            (
                "Predicted twoF is {}"
                " while recovered value is {},"
                " relative difference: {}".format(twoF_predicted, twoF2, diff)
            )
        )
        self.assertTrue(diff < 0.3)
        diff = np.abs(twoF2 - twoF) / twoF
        self.assertTrue(diff < 0.001)

    def test_run_computefstatistic_allowedMismatchFromSFTLength(self):

        long_Tsft_params = default_Writer_params.copy()
        long_Tsft_params["Tsft"] = 3600
        long_Tsft_params["duration"] = 4 * long_Tsft_params["Tsft"]
        long_Tsft_params["label"] = "long_Tsft"
        long_Tsft_params["F0"] = 1500
        long_Tsft_params["Band"] = 2.0
        long_Tsft_Writer = pyfstat.Writer(**long_Tsft_params)
        long_Tsft_Writer.run_makefakedata()

        search = pyfstat.ComputeFstat(
            tref=long_Tsft_Writer.tref,
            sftfilepattern=long_Tsft_Writer.sftfilepath,
            minCoverFreq=1499.5,
            maxCoverFreq=1500.5,
            allowedMismatchFromSFTLength=0.1,
        )
        with pytest.raises(RuntimeError):
            search.get_fullycoherent_twoF(F0=1500, F1=0, F2=0, Alpha=0, Delta=0)

        search = pyfstat.ComputeFstat(
            tref=long_Tsft_Writer.tref,
            sftfilepattern=long_Tsft_Writer.sftfilepath,
            minCoverFreq=1499.5,
            maxCoverFreq=1500.5,
            allowedMismatchFromSFTLength=0.5,
        )
        search.get_fullycoherent_twoF(F0=1500, F1=0, F2=0, Alpha=0, Delta=0)

    def test_run_computefstatistic_single_point_injectSources(self):

        predicted_FS = self.Writer.predict_fstat()

        injectSources = self.Writer.config_file_name
        search = pyfstat.ComputeFstat(
            tref=self.Writer.tref,
            assumeSqrtSX=1,
            injectSources=injectSources,
            minCoverFreq=28,
            maxCoverFreq=32,
            minStartTime=self.Writer.tstart,
            maxStartTime=self.Writer.tend(),
            detectors=self.Writer.detectors,
        )
        FS_from_file = search.get_fullycoherent_twoF(
            F0=self.Writer.F0,
            F1=self.Writer.F1,
            F2=self.Writer.F2,
            Alpha=self.Writer.Alpha,
            Delta=self.Writer.Delta,
        )
        self.assertTrue(np.abs(predicted_FS - FS_from_file) / FS_from_file < 0.3)

        injectSourcesdict = search.read_par(filename=injectSources)
        injectSourcesdict["F0"] = injectSourcesdict.pop("Freq")
        injectSourcesdict["F1"] = injectSourcesdict.pop("f1dot")
        injectSourcesdict["F2"] = injectSourcesdict.pop("f2dot")
        injectSourcesdict["phi"] = injectSourcesdict.pop("phi0")
        search = pyfstat.ComputeFstat(
            tref=self.Writer.tref,
            assumeSqrtSX=1,
            injectSources=injectSourcesdict,
            minCoverFreq=28,
            maxCoverFreq=32,
            minStartTime=self.Writer.tstart,
            maxStartTime=self.Writer.tend(),
            detectors=self.Writer.detectors,
        )
        FS_from_dict = search.get_fullycoherent_twoF(
            F0=self.Writer.F0,
            F1=self.Writer.F1,
            F2=self.Writer.F2,
            Alpha=self.Writer.Alpha,
            Delta=self.Writer.Delta,
        )
        self.assertTrue(FS_from_dict == FS_from_file)

    def test_get_fully_coherent_BSGL(self):
        # first pure noise, expect log10BSGL<0
        search_H1L1_noBSGL = pyfstat.ComputeFstat(
            tref=self.tref,
            minStartTime=self.tstart,
            maxStartTime=self.tstart + self.duration,
            detectors="H1,L1",
            injectSqrtSX=np.repeat(self.sqrtSX, 2),
            minCoverFreq=self.F0 - 0.1,
            maxCoverFreq=self.F0 + 0.1,
            BSGL=False,
            singleFstats=True,
            randSeed=self.randSeed,
        )
        twoF = search_H1L1_noBSGL.get_fullycoherent_detstat(
            F0=self.F0,
            F1=self.F1,
            F2=self.F2,
            Alpha=self.Alpha,
            Delta=self.Delta,
        )
        twoFX = search_H1L1_noBSGL.get_fullycoherent_single_IFO_twoFs()
        search_H1L1_BSGL = pyfstat.ComputeFstat(
            tref=self.tref,
            minStartTime=self.tstart,
            maxStartTime=self.tstart + self.duration,
            detectors="H1,L1",
            injectSqrtSX=np.repeat(self.sqrtSX, 2),
            minCoverFreq=self.F0 - 0.1,
            maxCoverFreq=self.F0 + 0.1,
            BSGL=True,
            randSeed=self.randSeed,
        )
        log10BSGL = search_H1L1_BSGL.get_fullycoherent_detstat(
            F0=self.F0,
            F1=self.F1,
            F2=self.F2,
            Alpha=self.Alpha,
            Delta=self.Delta,
        )
        self.assertTrue(log10BSGL < 0)
        self.assertTrue(
            log10BSGL == lalpulsar.ComputeBSGL(twoF, twoFX, search_H1L1_BSGL.BSGLSetup)
        )
        # now with an added signal, expect log10BSGL>0
        search_H1L1_noBSGL = pyfstat.ComputeFstat(
            tref=self.tref,
            minStartTime=self.tstart,
            maxStartTime=self.tstart + self.duration,
            detectors="H1,L1",
            injectSqrtSX=np.repeat(self.sqrtSX, 2),
            injectSources="{{Alpha={:g}; Delta={:g}; h0={:g}; cosi={:g}; Freq={:g}; f1dot={:g}; f2dot={:g}; refTime={:d};}}".format(
                self.Alpha,
                self.Delta,
                self.h0,
                self.cosi,
                self.F0,
                self.F1,
                self.F2,
                self.tref,
            ),
            minCoverFreq=self.F0 - 0.1,
            maxCoverFreq=self.F0 + 0.1,
            BSGL=False,
            singleFstats=True,
            randSeed=self.randSeed,
        )
        twoF = search_H1L1_noBSGL.get_fullycoherent_detstat(
            F0=self.F0,
            F1=self.F1,
            F2=self.F2,
            Alpha=self.Alpha,
            Delta=self.Delta,
        )
        twoFX = search_H1L1_noBSGL.get_fullycoherent_single_IFO_twoFs()
        search_H1L1_BSGL = pyfstat.ComputeFstat(
            tref=self.tref,
            minStartTime=self.tstart,
            maxStartTime=self.tstart + self.duration,
            detectors="H1,L1",
            injectSqrtSX=np.repeat(self.sqrtSX, 2),
            injectSources="{{Alpha={:g}; Delta={:g}; h0={:g}; cosi={:g}; Freq={:g}; f1dot={:g}; f2dot={:g}; refTime={:d};}}".format(
                self.Alpha,
                self.Delta,
                self.h0,
                self.cosi,
                self.F0,
                self.F1,
                self.F2,
                self.tref,
            ),
            minCoverFreq=self.F0 - 0.1,
            maxCoverFreq=self.F0 + 0.1,
            BSGL=True,
            randSeed=self.randSeed,
        )
        log10BSGL = search_H1L1_BSGL.get_fullycoherent_detstat(
            F0=self.F0,
            F1=self.F1,
            F2=self.F2,
            Alpha=self.Alpha,
            Delta=self.Delta,
        )
        self.assertTrue(log10BSGL > 0)
        self.assertTrue(
            log10BSGL == lalpulsar.ComputeBSGL(twoF, twoFX, search_H1L1_BSGL.BSGLSetup)
        )

    def test_cumulative_twoF(self):
        Nsft = 100
        # not using any SFTs on disk
        search = pyfstat.ComputeFstat(
            tref=self.tref,
            minStartTime=self.tstart,
            maxStartTime=self.tstart + Nsft * self.Tsft,
            detectors=self.detectors,
            injectSqrtSX=self.sqrtSX,
            injectSources=default_signal_params,
            minCoverFreq=self.F0 - 0.1,
            maxCoverFreq=self.F0 + 0.1,
        )
        start_time, taus, twoF_cumulative = search.calculate_twoF_cumulative(
            self.Writer.F0,
            self.Writer.F1,
            self.Writer.F2,
            self.Writer.Alpha,
            self.Writer.Delta,
            num_segments=Nsft + 1,
        )
        twoF = search.get_fullycoherent_detstat(
            F0=self.Writer.F0,
            F1=self.Writer.F1,
            F2=self.Writer.F2,
            Alpha=self.Writer.Alpha,
            Delta=self.Writer.Delta,
            tstart=self.Writer.tstart,
            tend=self.Writer.tstart + taus[-1],
        )
        reldiff = np.abs(twoF_cumulative[-1] - twoF) / twoF
        print(
            "2F from get_fullycoherent_detstat() is {:.4f}"
            " while last value from calculate_twoF_cumulative() is {:.4f};"
            " relative difference: {:.2f}".format(
                twoF, twoF_cumulative[-1], 100 * reldiff
            )
        )
        self.assertTrue(reldiff < 0.1)
        idx = int(Nsft / 2)
        partial_2F_expected = (taus[idx] / taus[-1]) * twoF
        reldiff = (
            np.abs(twoF_cumulative[idx] - partial_2F_expected) / partial_2F_expected
        )
        print(
            "Middle 2F value from calculate_twoF_cumulative() is {:.4f}"
            " while from duration ratio we'd expect {:.4f}*{:.4f}={:.4f};"
            " relative difference: {:.2f}%".format(
                twoF_cumulative[idx],
                taus[idx] / taus[-1],
                twoF,
                partial_2F_expected,
                100 * reldiff,
            )
        )
        self.assertTrue(reldiff < 0.1)
        _, _, pfs, pfs_sigma = search.predict_twoF_cumulative(
            F0=self.Writer.F0,
            Alpha=self.Writer.Alpha,
            Delta=self.Writer.Delta,
            h0=self.Writer.h0,
            cosi=self.Writer.cosi,
            psi=self.Writer.psi,
            tstart=self.tstart,
            tend=self.tstart + Nsft * self.Tsft,
            IFOs=self.detectors,
            assumeSqrtSX=self.sqrtSX,
            num_segments=3,  # this is slow, so only do start,mid,end
        )
        reldiffmid = 100 * (twoF_cumulative[idx] - pfs[1]) / pfs[1]
        reldiffend = 100 * (twoF_cumulative[-1] - pfs[2]) / pfs[2]
        print(
            "Predicted 2F values from predict_twoF_cumulative() are"
            " {:.4f}+-{:.4f}(+-{:.2f}%) at midpoint of data"
            " and {:.4f}+-{:.4f}(+-{:.2f}%) after full data,"
            " , relative differences: {:.2f}% and {:.2f}%".format(
                pfs[1],
                pfs_sigma[1],
                100 * pfs_sigma[1] / pfs[1],
                pfs[2],
                pfs_sigma[2],
                100 * pfs_sigma[2] / pfs[2],
                reldiffmid,
                reldiffend,
            )
        )
        self.assertTrue(reldiffmid < 0.25)
        self.assertTrue(reldiffend < 0.25)


class TestComputeFstatNoNoise(BaseForTestsWithData):
    # FIXME: should be possible to merge into TestComputeFstat with smart
    # defaults handlingf
    label = "TestComputeFstatSinglePointNoNoise"
    sqrtSX = 0

    def test_run_computefstatistic_single_point_no_noise(self):

        predicted_FS = self.Writer.predict_fstat(assumeSqrtSX=1)
        search = pyfstat.ComputeFstat(
            tref=self.Writer.tref,
            assumeSqrtSX=1,
            sftfilepattern=self.Writer.sftfilepath,
            search_ranges=self.search_ranges,
        )
        FS = search.get_fullycoherent_twoF(
            F0=self.Writer.F0,
            F1=self.Writer.F1,
            F2=self.Writer.F2,
            Alpha=self.Writer.Alpha,
            Delta=self.Writer.Delta,
        )
        self.assertTrue(np.abs(predicted_FS - FS) / FS < 0.3)

    def test_run_computefstatistic_single_point_no_noise_manual_ephem(self):

        predicted_FS = self.Writer.predict_fstat(assumeSqrtSX=1)

        # let's get the default ephemeris files (to be sure their paths exist)
        # and then pretend we pass them manually, to test those class options
        (
            earth_ephem_default,
            sun_ephem_default,
        ) = pyfstat.helper_functions.get_ephemeris_files()

        search = pyfstat.ComputeFstat(
            tref=self.Writer.tref,
            assumeSqrtSX=1,
            sftfilepattern=self.Writer.sftfilepath,
            earth_ephem=earth_ephem_default,
            sun_ephem=sun_ephem_default,
            search_ranges=self.search_ranges,
        )
        FS = search.get_fullycoherent_twoF(
            F0=self.Writer.F0,
            F1=self.Writer.F1,
            F2=self.Writer.F2,
            Alpha=self.Writer.Alpha,
            Delta=self.Writer.Delta,
        )
        self.assertTrue(np.abs(predicted_FS - FS) / FS < 0.3)


class TestSearchForSignalWithJumps(TestBaseSearchClass):
    def test_shift_matrix(self):
        search = pyfstat.SearchForSignalWithJumps()
        dT = 10
        a = search._shift_matrix(4, dT)
        b = np.array(
            [
                [
                    1,
                    2 * np.pi * dT,
                    2 * np.pi * dT**2 / 2.0,
                    2 * np.pi * dT**3 / 6.0,
                ],
                [0, 1, dT, dT**2 / 2.0],
                [0, 0, 1, dT],
                [0, 0, 0, 1],
            ]
        )
        self.assertTrue(np.array_equal(a, b))

    def test_shift_coefficients(self):
        search = pyfstat.SearchForSignalWithJumps()
        thetaA = np.array([10.0, 1e2, 10.0, 1e2])
        dT = 100

        # Calculate the 'long' way
        thetaB = np.zeros(len(thetaA))
        thetaB[3] = thetaA[3]
        thetaB[2] = thetaA[2] + thetaA[3] * dT
        thetaB[1] = thetaA[1] + thetaA[2] * dT + 0.5 * thetaA[3] * dT**2
        thetaB[0] = thetaA[0] + 2 * np.pi * (
            thetaA[1] * dT + 0.5 * thetaA[2] * dT**2 + thetaA[3] * dT**3 / 6.0
        )

        self.assertTrue(np.array_equal(thetaB, search._shift_coefficients(thetaA, dT)))

    def test_shift_coefficients_loop(self):
        search = pyfstat.SearchForSignalWithJumps()
        thetaA = np.array([10.0, 1e2, 10.0, 1e2])
        dT = 1e1
        thetaB = search._shift_coefficients(thetaA, dT)
        self.assertTrue(
            np.allclose(
                thetaA, search._shift_coefficients(thetaB, -dT), rtol=1e-9, atol=1e-9
            )
        )


class TestSemiCoherentSearch(BaseForTestsWithData):
    label = "TestSemiCoherentSearch"
    detectors = "H1,L1"
    nsegs = 2

    def test_get_semicoherent_twoF(self):

        search = pyfstat.SemiCoherentSearch(
            label=self.label,
            outdir=self.outdir,
            nsegs=self.nsegs,
            sftfilepattern=self.Writer.sftfilepath,
            tref=self.Writer.tref,
            search_ranges=self.search_ranges,
            BSGL=False,
        )

        twoF_sc = search.get_semicoherent_det_stat(
            self.Writer.F0,
            self.Writer.F1,
            self.Writer.F2,
            self.Writer.Alpha,
            self.Writer.Delta,
            record_segments=True,
        )
        twoF_per_seg_computed = np.array(search.twoF_per_segment)

        twoF_predicted = self.Writer.predict_fstat()
        # now compute the predicted semi-coherent Fstat for each segment
        print(self.Writer.duration)
        self.Writer.duration /= self.nsegs
        tstart = self.Writer.tstart
        print(tstart)
        twoF_per_seg_predicted = np.zeros(self.nsegs)
        for n in range(self.nsegs):
            self.Writer.tstart = tstart + n * self.Writer.duration
            print(self.Writer.tstart)
            print(self.Writer.duration)
            twoF_per_seg_predicted[n] = self.Writer.predict_fstat()

        self.assertTrue(len(twoF_per_seg_computed) == len(twoF_per_seg_predicted))
        diffs = (
            np.abs(twoF_per_seg_computed - twoF_per_seg_predicted)
            / twoF_per_seg_predicted
        )
        print(
            (
                "Predicted twoF per segment are {}"
                " while recovered values are {},"
                " relative difference: {}".format(
                    twoF_per_seg_predicted, twoF_per_seg_computed, diffs
                )
            )
        )
        self.assertTrue(np.all(diffs < 0.3))
        diff = np.abs(twoF_sc - twoF_predicted) / twoF_predicted
        print(
            (
                "Predicted semicoherent twoF is {}"
                " while recovered value is {},"
                " relative difference: {}".format(twoF_predicted, twoF_sc, diff)
            )
        )
        self.assertTrue(diff < 0.3)

    def _test_get_semicoherent_BSGL(self, **dataopts):
        search_noBSGL = pyfstat.SemiCoherentSearch(
            label=self.label,
            outdir=self.outdir,
            nsegs=self.nsegs,
            BSGL=False,
            singleFstats=True,
            **dataopts,
        )
        twoF = search_noBSGL.get_semicoherent_det_stat(
            self.Writer.F0,
            self.Writer.F1,
            self.Writer.F2,
            self.Writer.Alpha,
            self.Writer.Delta,
        )
        twoFX = search_noBSGL.get_semicoherent_single_IFO_twoFs()
        search_BSGL = pyfstat.SemiCoherentSearch(
            label=self.label,
            outdir=self.outdir,
            nsegs=self.nsegs,
            BSGL=True,
            **dataopts,
        )
        log10BSGL = search_BSGL.get_semicoherent_det_stat(
            self.Writer.F0,
            self.Writer.F1,
            self.Writer.F2,
            self.Writer.Alpha,
            self.Writer.Delta,
            record_segments=True,
        )
        self.assertTrue(log10BSGL > 0)
        self.assertTrue(
            log10BSGL == lalpulsar.ComputeBSGL(twoF, twoFX, search_BSGL.BSGLSetup)
        )

    def test_get_semicoherent_BSGL_SFTs(self):
        dataopts = {
            "sftfilepattern": self.Writer.sftfilepath,
            "tref": self.Writer.tref,
            "search_ranges": self.search_ranges,
        }
        self._test_get_semicoherent_BSGL(**dataopts)

    def test_get_semicoherent_BSGL_inject(self):
        dataopts = {
            "tref": self.tref,
            "minStartTime": self.tstart,
            "maxStartTime": self.tstart + self.duration,
            "detectors": "H1,L1",
            "injectSqrtSX": np.repeat(self.sqrtSX, 2),
            "minCoverFreq": self.F0 - 0.1,
            "maxCoverFreq": self.F0 + 0.1,
            "injectSources": self.Writer.config_file_name,
            "randSeed": self.randSeed,
        }
        self._test_get_semicoherent_BSGL(**dataopts)

    def test_get_semicoherent_twoF_allowedMismatchFromSFTLength(self):

        long_Tsft_params = default_Writer_params.copy()
        long_Tsft_params["Tsft"] = 3600
        long_Tsft_params["duration"] = 4 * long_Tsft_params["Tsft"]
        long_Tsft_params["label"] = "long_Tsft"
        long_Tsft_params["F0"] = 1500
        long_Tsft_params["Band"] = 2.0
        long_Tsft_Writer = pyfstat.Writer(**long_Tsft_params)
        long_Tsft_Writer.run_makefakedata()

        search = pyfstat.SemiCoherentSearch(
            label=self.label,
            outdir=self.outdir,
            tref=long_Tsft_Writer.tref,
            sftfilepattern=long_Tsft_Writer.sftfilepath,
            nsegs=self.nsegs,
            minCoverFreq=1499.5,
            maxCoverFreq=1500.5,
            allowedMismatchFromSFTLength=0.1,
        )
        with pytest.raises(RuntimeError):
            search.get_semicoherent_twoF(F0=1500, F1=0, F2=0, Alpha=0, Delta=0)

        search = pyfstat.SemiCoherentSearch(
            label=self.label,
            outdir=self.outdir,
            tref=long_Tsft_Writer.tref,
            sftfilepattern=long_Tsft_Writer.sftfilepath,
            nsegs=self.nsegs,
            minCoverFreq=1499.5,
            maxCoverFreq=1500.5,
            allowedMismatchFromSFTLength=0.5,
        )
        search.get_semicoherent_twoF(F0=1500, F1=0, F2=0, Alpha=0, Delta=0)


class TestSemiCoherentGlitchSearch(BaseForTestsWithData):
    label = "TestSemiCoherentGlitchSearch"
    dtglitch = 3600
    Band = 1

    def _run_test(self, delta_F0):

        Writer = pyfstat.GlitchWriter(
            self.label,
            outdir=self.outdir,
            tstart=self.tstart,
            duration=self.duration,
            dtglitch=self.dtglitch,
            delta_F0=delta_F0,
            detectors=self.detectors,
            sqrtSX=self.sqrtSX,
            **default_signal_params,
            SFTWindowType=self.SFTWindowType,
            SFTWindowBeta=self.SFTWindowBeta,
            randSeed=self.randSeed,
            Band=self.Band,
        )

        Writer.make_data(verbose=True)

        vanilla_search = pyfstat.SemiCoherentSearch(
            label=self.label,
            outdir=self.outdir,
            nsegs=2,
            sftfilepattern=self.Writer.sftfilepath,
            tref=Writer.tref,
            search_ranges=self.search_ranges,
        )

        # Compute the predicted semi-coherent glitch Fstat for the first half
        Writer.transientStartTime = Writer.tstart
        Writer.transientTau = self.dtglitch
        FSA = Writer.predict_fstat()
        # same for the second half (tau stays the same)
        Writer.transientStartTime += self.dtglitch
        FSB = Writer.predict_fstat()
        predicted_FS = FSA + FSB

        # vanilla semicoherent search not knowing about potential glitch
        twoF_sc_vanilla = vanilla_search.get_semicoherent_det_stat(
            Writer.F0,
            Writer.F1,
            Writer.F2,
            Writer.Alpha,
            Writer.Delta,
            record_segments=True,
        )
        twoF_per_seg_vanilla = vanilla_search.twoF_per_segment
        diff = np.abs(twoF_sc_vanilla - predicted_FS) / predicted_FS
        print(
            (
                "Predicted twoF is {}+{}={}"
                " while recovered value from SemiCoherentSearch is {}+{}={},"
                " relative difference: {}".format(
                    FSA, FSB, predicted_FS, *twoF_per_seg_vanilla, twoF_sc_vanilla, diff
                )
            )
        )
        if delta_F0 == 0:
            self.assertTrue(diff < 0.3)
        else:
            self.assertFalse(diff < 0.3)

        # glitch-robust search
        keys = ["F0", "F1", "F2", "Alpha", "Delta"]
        search_ranges = {
            key: [
                getattr(Writer, key),
                getattr(Writer, key) + getattr(Writer, "delta_" + key, 0.0),
            ]
            for key in keys
        }
        glitch_search = pyfstat.SemiCoherentGlitchSearch(
            label=self.label,
            outdir=self.outdir,
            sftfilepattern=Writer.sftfilepath,
            tref=Writer.tref,
            minStartTime=Writer.tstart,
            maxStartTime=Writer.tend(),
            nglitch=1,
            search_ranges=search_ranges,
        )
        twoF_glitch = glitch_search.get_semicoherent_nglitch_twoF(
            Writer.F0,
            Writer.F1,
            Writer.F2,
            Writer.Alpha,
            Writer.Delta,
            Writer.delta_F0,
            Writer.delta_F1,
            glitch_search.minStartTime + self.dtglitch,
        )
        diff = np.abs(twoF_glitch - predicted_FS) / predicted_FS
        print(
            (
                "Predicted twoF is {}+{}={}"
                " while recovered value from SemiCoherentGlitchSearch is {},"
                " relative difference: {}".format(
                    FSA, FSB, predicted_FS, twoF_glitch, diff
                )
            )
        )
        self.assertTrue(diff < 0.3)
        diff2 = np.abs((twoF_glitch - twoF_sc_vanilla) / twoF_sc_vanilla)
        print(
            "Relative difference between SemiCoherentSearch"
            "and SemiCoherentGlitchSearch: {}".format(diff2)
        )
        if delta_F0 == 0:
            self.assertTrue(diff2 < 0.01)
        else:
            self.assertTrue(twoF_glitch > twoF_sc_vanilla)
            self.assertTrue(diff2 > 0.3)

    def test_get_semicoherent_nglitch_twoF_no_glitch(self):
        self._run_test(delta_F0=0)

    def test_get_semicoherent_nglitch_twoF_with_glitch(self):
        self._run_test(delta_F0=0.1)


class BaseForMCMCSearchTests(BaseForTestsWithData):
    # this class is only used for common utilities for MCMCSearch-based classes
    # and doesn't run any tests itself
    label = "TestMCMCSearch"
    Band = 1

    def _check_twoF_predicted(self, assertTrue=True):
        self.twoF_predicted = self.Writer.predict_fstat()
        self.max_dict, self.maxTwoF = self.search.get_max_twoF()
        diff = np.abs((self.maxTwoF - self.twoF_predicted)) / self.twoF_predicted
        print(
            (
                "Predicted twoF is {} while recovered is {},"
                " relative difference: {}".format(
                    self.twoF_predicted, self.maxTwoF, diff
                )
            )
        )
        if assertTrue:
            self.assertTrue(diff < 0.3)

    def _check_mcmc_quantiles(self, transient=False, assertTrue=True):
        summary_stats = self.search.get_summary_stats()
        nsigmas = 3
        conf = "99"

        if not transient:
            inj = {k: getattr(self.Writer, k) for k in self.max_dict}
        else:
            inj = {
                "transient_tstart": self.Writer.transientStartTime,
                "transient_duration": self.Writer.transientTau,
            }

        for k in self.max_dict.keys():
            reldiff = np.abs((self.max_dict[k] - inj[k]) / inj[k])
            print("max2F  {:s} reldiff: {:.2e}".format(k, reldiff))
            reldiff = np.abs((summary_stats[k]["mean"] - inj[k]) / inj[k])
            print("mean   {:s} reldiff: {:.2e}".format(k, reldiff))
            reldiff = np.abs((summary_stats[k]["median"] - inj[k]) / inj[k])
            print("median {:s} reldiff: {:.2e}".format(k, reldiff))
        for k in self.max_dict.keys():
            lower = summary_stats[k]["mean"] - nsigmas * summary_stats[k]["std"]
            upper = summary_stats[k]["mean"] + nsigmas * summary_stats[k]["std"]
            within = (inj[k] >= lower) and (inj[k] <= upper)
            print(
                "{:s} in mean+-{:d}std ({} in [{},{}])? {}".format(
                    k, nsigmas, inj[k], lower, upper, within
                )
            )
            if assertTrue:
                self.assertTrue(within)
            within = (inj[k] >= summary_stats[k]["lower" + conf]) and (
                inj[k] <= summary_stats[k]["upper" + conf]
            )
            print(
                "{:s} in {:s}% quantiles ({} in [{},{}])? {}".format(
                    k,
                    conf,
                    inj[k],
                    summary_stats[k]["lower" + conf],
                    summary_stats[k]["upper" + conf],
                    within,
                )
            )
            if assertTrue:
                self.assertTrue(within)

    def _test_plots(self):
        self.search.plot_corner(add_prior=True)
        self.search.plot_prior_posterior()
        self.search.plot_cumulative_max()
        self.search.plot_chainconsumer()


class TestMCMCSearch(BaseForMCMCSearchTests):
    label = "TestMCMCSearch"
    BSGL = False

    def test_fully_coherent_MCMC(self):

        # use a single test case with loop over multiple prior choices
        # this could be much more elegantly done with @pytest.mark.parametrize
        # but that cannot be mixed with unittest classes
        thetas = {
            "uniformF0-uniformF1-fixedSky": {
                "F0": {
                    "type": "unif",
                    "lower": self.F0 - 1e-6,
                    "upper": self.F0 + 1e-6,
                },
                "F1": {
                    "type": "unif",
                    "lower": self.F1 - 1e-10,
                    "upper": self.F1 + 1e-10,
                },
                "F2": self.F2,
                "Alpha": self.Alpha,
                "Delta": self.Delta,
            },
            "log10uniformF0-uniformF1-fixedSky": {
                "F0": {
                    "type": "log10unif",
                    "log10lower": np.log10(self.F0 - 1e-6),
                    "log10upper": np.log10(self.F0 + 1e-6),
                },
                "F1": {
                    "type": "unif",
                    "lower": self.F1 - 1e-10,
                    "upper": self.F1 + 1e-10,
                },
                "F2": self.F2,
                "Alpha": self.Alpha,
                "Delta": self.Delta,
            },
            "normF0-normF1-fixedSky": {
                "F0": {"type": "norm", "loc": self.F0, "scale": 1e-6},
                "F1": {"type": "norm", "loc": self.F1, "scale": 1e-10},
                "F2": self.F2,
                "Alpha": self.Alpha,
                "Delta": self.Delta,
            },
            "lognormF0-halfnormF1-fixedSky": {
                # lognorm parametrization is weird, from the scipy docs:
                # "A common parametrization for a lognormal random variable Y
                # is in terms of the mean, mu, and standard deviation, sigma,
                # of the unique normally distributed random variable X
                # such that exp(X) = Y.
                # This parametrization corresponds to setting s = sigma
                # and scale = exp(mu)."
                # Hence, to set up a "lognorm" prior, we need
                # to give "loc" in log scale but "scale" in linear scale
                # Also, "lognorm" makes no sense for negative F1,
                # hence combining this with "halfnorm" into a single case.
                "F0": {"type": "lognorm", "loc": np.log(self.F0), "scale": 1e-6},
                "F1": {"type": "halfnorm", "loc": self.F1 - 1e-10, "scale": 1e-10},
                "F2": self.F2,
                "Alpha": self.Alpha,
                "Delta": self.Delta,
            },
            "normF0-normF1-uniformSky": {
                # norm in sky is too dangerous, can easily jump out of range
                "F0": {"type": "norm", "loc": self.F0, "scale": 1e-6},
                "F1": {"type": "norm", "loc": self.F1, "scale": 1e-10},
                "F2": self.F2,
                "Alpha": {
                    "type": "unif",
                    "lower": self.Alpha - 0.01,
                    "upper": self.Alpha + 0.01,
                },
                "Delta": {
                    "type": "unif",
                    "lower": self.Delta - 0.01,
                    "upper": self.Delta + 0.01,
                },
            },
        }

        for prior_choice in thetas:
            self.search = pyfstat.MCMCSearch(
                label=self.label + "-" + prior_choice,
                outdir=self.outdir,
                theta_prior=thetas[prior_choice],
                tref=self.tref,
                sftfilepattern=self.Writer.sftfilepath,
                nsteps=[20, 20],
                nwalkers=20,
                ntemps=2,
                log10beta_min=-1,
                BSGL=self.BSGL,
            )
            self.search.run(plot_walkers=False)
            self.search.print_summary()
            self._check_twoF_predicted()
            self._check_mcmc_quantiles()
            self._test_plots()


class TestMCMCSearchBSGL(TestMCMCSearch):
    label = "TestMCMCSearch"
    detectors = "H1,L1"
    BSGL = True

    def test_MCMC_search_on_data_with_line(self):
        # We reuse the default multi-IFO SFTs
        # but add an additional single-detector artifact to H1 only.
        # For simplicity, this is modelled here as a fully modulated CW-like signal,
        # just restricted to the single detector.
        SFTs_H1 = self.Writer.sftfilepath.split(";")[0]
        SFTs_L1 = self.Writer.sftfilepath.split(";")[1]
        extra_writer = pyfstat.Writer(
            label=self.label + "_with_line",
            outdir=self.outdir,
            tref=self.tref,
            F0=self.Writer.F0 + 0.5e-2,
            F1=0,
            F2=0,
            Alpha=self.Writer.Alpha,
            Delta=self.Writer.Delta,
            h0=10 * self.Writer.h0,
            cosi=self.Writer.cosi,
            sqrtSX=0,  # don't add yet another set of Gaussian noise
            noiseSFTs=SFTs_H1,
            SFTWindowType=self.Writer.SFTWindowType,
            SFTWindowBeta=self.Writer.SFTWindowBeta,
        )
        extra_writer.make_data()
        data_with_line = ";".join([SFTs_L1, extra_writer.sftfilepath])
        # use a single fixed prior and search F0 only for speed
        thetas = {
            "F0": {
                "type": "unif",
                "lower": self.F0 - 1e-2,
                "upper": self.F0 + 1e-2,
            },
            "F1": self.F1,
            "F2": self.F2,
            "Alpha": self.Alpha,
            "Delta": self.Delta,
        }
        # now run a standard F-stat search over this data
        self.search = pyfstat.MCMCSearch(
            label=self.label + "-F",
            outdir=self.outdir,
            theta_prior=thetas,
            tref=self.tref,
            sftfilepattern=data_with_line,
            nsteps=[20, 20],
            nwalkers=20,
            ntemps=2,
            log10beta_min=-1,
            BSGL=False,
        )
        self.search.run(plot_walkers=True)
        self.search.print_summary()
        # The standard checks here are expected to fail,
        # as the F-search will get confused by the line
        # and recover a much higher maxTwoF than predicted.
        self._check_twoF_predicted(assertTrue=False)
        mode_F0_Fsearch = self.max_dict["F0"]
        maxTwoF_Fsearch = self.maxTwoF
        self._check_mcmc_quantiles(assertTrue=False)
        self.assertTrue(maxTwoF_Fsearch > self.twoF_predicted)
        self._test_plots()
        # also run a BSGL search over the same data
        self.search = pyfstat.MCMCSearch(
            label=self.label + "-BSGL",
            outdir=self.outdir,
            theta_prior=thetas,
            tref=self.tref,
            sftfilepattern=data_with_line,
            nsteps=[20, 20],
            nwalkers=20,
            ntemps=2,
            log10beta_min=-1,
            BSGL=True,
        )
        self.search.run(plot_walkers=True)
        self.search.print_summary()
        # Still skipping the standard checks,
        # as we're using too cheap a MCMC setup here for them to be robust.
        self._check_twoF_predicted(assertTrue=False)
        mode_F0_BSGLsearch = self.max_dict["F0"]
        maxTwoF_BSGLsearch = self.maxTwoF
        self._check_mcmc_quantiles(assertTrue=False)
        # But for sure, the BSGL search should find a lower-F mode
        # closer to the true multi-IFO signal.
        self.assertTrue(maxTwoF_BSGLsearch < maxTwoF_Fsearch)
        self.assertTrue(mode_F0_BSGLsearch < mode_F0_Fsearch)
        self.assertTrue(
            np.abs(mode_F0_BSGLsearch - self.F0) < np.abs(mode_F0_Fsearch - self.F0)
        )
        self.assertTrue(maxTwoF_BSGLsearch < self.twoF_predicted)
        self._test_plots()


class TestMCMCSemiCoherentSearch(BaseForMCMCSearchTests):
    label = "TestMCMCSemiCoherentSearch"

    def test_semi_coherent_MCMC(self):

        theta = {
            "F0": {
                "type": "unif",
                "lower": self.F0 - 1e-6,
                "upper": self.F0 + 1e-6,
            },
            "F1": {
                "type": "unif",
                "lower": self.F1 - 1e-10,
                "upper": self.F1 + 1e-10,
            },
            "F2": self.F2,
            "Alpha": self.Alpha,
            "Delta": self.Delta,
        }
        nsegs = 2
        self.search = pyfstat.MCMCSemiCoherentSearch(
            label=self.label,
            outdir=self.outdir,
            theta_prior=theta,
            tref=self.tref,
            sftfilepattern=self.Writer.sftfilepath,
            nsteps=[100, 100],
            nwalkers=100,
            ntemps=2,
            log10beta_min=-1,
            nsegs=nsegs,
        )
        self.search.run(plot_walkers=False)
        self.search.print_summary()

        self._check_twoF_predicted()

        # recover per-segment twoF values at max point
        twoF_sc = self.search.search.get_semicoherent_det_stat(
            self.max_dict["F0"],
            self.max_dict["F1"],
            self.F2,
            self.Alpha,
            self.Delta,
            record_segments=True,
        )
        self.assertTrue(np.abs(twoF_sc - self.maxTwoF) / self.maxTwoF < 0.01)
        twoF_per_seg = np.array(self.search.search.twoF_per_segment)
        self.assertTrue(len(twoF_per_seg) == nsegs)
        twoF_summed = twoF_per_seg.sum()
        self.assertTrue(np.abs(twoF_summed - twoF_sc) / twoF_sc < 0.01)

        self._check_mcmc_quantiles()
        self._test_plots()


class TestMCMCFollowUpSearch(BaseForMCMCSearchTests):
    label = "TestMCMCFollowUpSearch"
    # Supersky metric cannot be computed for segment lengths <= ~24 hours
    duration = 10 * 86400
    # FIXME: if h0 too high for given duration, offsets to PFS become too large
    h0 = 0.1

    def test_MCMC_followup_search(self):

        theta = {
            "F0": {
                "type": "unif",
                "lower": self.F0 - 1e-6,
                "upper": self.F0 + 1e-6,
            },
            "F1": {
                "type": "unif",
                "lower": self.F1 - 1e-10,
                "upper": self.F1 + 1e-10,
            },
            "F2": self.F2,
            "Alpha": self.Alpha,
            "Delta": self.Delta,
        }
        nsegs = 10
        NstarMax = 1000
        self.search = pyfstat.MCMCFollowUpSearch(
            label=self.label,
            outdir=self.outdir,
            theta_prior=theta,
            tref=self.tref,
            sftfilepattern=self.Writer.sftfilepath,
            nsteps=[100, 100],
            nwalkers=100,
            ntemps=2,
            log10beta_min=-1,
        )
        self.search.run(
            plot_walkers=False,
            NstarMax=NstarMax,
            Nsegs0=nsegs,
        )
        self.search.print_summary()
        self._check_twoF_predicted()
        self._check_mcmc_quantiles()
        self._test_plots()


class TestMCMCTransientSearch(BaseForMCMCSearchTests):
    label = "TestMCMCTransientSearch"
    duration = 86400

    def setup_method(self, method):
        self.transientWindowType = "rect"
        self.transientStartTime = int(self.tstart + 0.25 * self.duration)
        self.transientTau = int(0.5 * self.duration)
        self.Writer = pyfstat.Writer(
            label=self.label,
            tstart=self.tstart,
            duration=self.duration,
            tref=self.tref,
            **default_signal_params,
            outdir=self.outdir,
            sqrtSX=self.sqrtSX,
            Band=self.Band,
            detectors=self.detectors,
            SFTWindowType=self.SFTWindowType,
            SFTWindowBeta=self.SFTWindowBeta,
            randSeed=self.randSeed,
            transientWindowType=self.transientWindowType,
            transientStartTime=self.transientStartTime,
            transientTau=self.transientTau,
        )
        self.Writer.make_data(verbose=True)
        self.basic_theta = {
            "F0": self.F0,
            "F1": self.F1,
            "F2": self.F2,
            "Alpha": self.Alpha,
            "Delta": self.Delta,
        }
        self.MCMC_params = {
            "nsteps": [50, 50],
            "nwalkers": 50,
            "ntemps": 2,
            "log10beta_min": -1,
        }

    def test_transient_MCMC_t0only(self):

        theta = {
            **self.basic_theta,
            "transient_tstart": {
                "type": "unif",
                "lower": self.Writer.tstart,
                "upper": self.Writer.tend() - 2 * self.Writer.Tsft,
            },
            "transient_duration": self.transientTau,
        }
        self.search = pyfstat.MCMCTransientSearch(
            label=self.label,
            outdir=self.outdir,
            theta_prior=theta,
            tref=self.tref,
            sftfilepattern=self.Writer.sftfilepath,
            **self.MCMC_params,
            transientWindowType=self.transientWindowType,
        )
        self.search.run(plot_walkers=False)
        self.search.print_summary()
        self._check_twoF_predicted()
        self._check_mcmc_quantiles(transient=True)
        self._test_plots()

    def test_transient_MCMC_tauonly(self):

        theta = {
            **self.basic_theta,
            "transient_tstart": self.transientStartTime,
            "transient_duration": {
                "type": "unif",
                "lower": 2 * self.Writer.Tsft,
                "upper": self.Writer.duration - 2 * self.Writer.Tsft,
            },
        }
        self.search = pyfstat.MCMCTransientSearch(
            label=self.label,
            outdir=self.outdir,
            theta_prior=theta,
            tref=self.tref,
            sftfilepattern=self.Writer.sftfilepath,
            **self.MCMC_params,
            transientWindowType=self.transientWindowType,
        )
        self.search.run(plot_walkers=False)
        self.search.print_summary()
        self._check_twoF_predicted()
        self._check_mcmc_quantiles(transient=True)
        self._test_plots()

    def test_transient_MCMC_t0_tau(self):

        theta = {
            **self.basic_theta,
            "transient_tstart": {
                "type": "unif",
                "lower": self.Writer.tstart,
                "upper": self.Writer.tend() - 2 * self.Writer.Tsft,
            },
            "transient_duration": {
                "type": "unif",
                "lower": 2 * self.Writer.Tsft,
                "upper": self.Writer.duration - 2 * self.Writer.Tsft,
            },
        }
        self.search = pyfstat.MCMCTransientSearch(
            label=self.label,
            outdir=self.outdir,
            theta_prior=theta,
            tref=self.tref,
            sftfilepattern=self.Writer.sftfilepath,
            **self.MCMC_params,
            transientWindowType=self.transientWindowType,
        )
        self.search.run(plot_walkers=False)
        self.search.print_summary()
        self._check_twoF_predicted()
        self._check_mcmc_quantiles(transient=True)
        self._test_plots()


class TestGridSearch(BaseForTestsWithData):
    label = "TestGridSearch"
    # Need to hand-pick values F0s here for the CFSv2 comparison:
    # that code sometimes includes endpoints, sometimes not.
    # For the F0s here, it happens to match our convention (include endpoint).
    F0s = [29.999, 30.001, 1e-4]
    F1s = [-1e-10, 0, 1e-11]
    Band = 0.5
    BSGL = False

    def _test_plots(self, search_keys):
        for key in search_keys:
            self.search.plot_1D(xkey=key, savefig=True)
        if len(search_keys) == 2:
            self.search.plot_2D(xkey=search_keys[0], ykey=search_keys[1], colorbar=True)
        vals = [
            np.unique(self.search.data[key]) - getattr(self.Writer, key)
            for key in search_keys
        ]
        twoF = self.search.data["twoF"].reshape([len(kval) for kval in vals])
        corner_labels = [f"${key} - {key}_0$" for key in search_keys]
        corner_labels.append("2F")
        gridcorner_fig, gridcorner_axes = pyfstat.gridcorner(
            twoF,
            vals,
            projection="log_mean",
            labels=corner_labels,
            whspace=0.1,
            factor=1.8,
        )
        gridcorner_fig.savefig(
            os.path.join(self.search.outdir, self.search.label + "_corner.png")
        )

    def test_grid_search_1D(self):
        self.search = pyfstat.GridSearch(
            "grid_search_F0",
            self.outdir,
            self.Writer.sftfilepath,
            F0s=self.F0s,
            F1s=[self.Writer.F1],
            F2s=[self.Writer.F2],
            Alphas=[self.Writer.Alpha],
            Deltas=[self.Writer.Delta],
            tref=self.tref,
            BSGL=self.BSGL,
        )
        self.search.run()
        self.assertTrue(os.path.isfile(self.search.out_file))
        max2F_point = self.search.get_max_twoF()
        self.assertTrue(np.all(max2F_point["twoF"] >= self.search.data["twoF"]))
        search_keys = ["F0"]  # only the ones that aren't 0-width
        self._test_plots(search_keys)

    def test_grid_search_2D(self):
        self.search = pyfstat.GridSearch(
            "grid_search_F0F1",
            self.outdir,
            self.Writer.sftfilepath,
            F0s=self.F0s,
            F1s=self.F1s,
            F2s=[self.Writer.F2],
            Alphas=[self.Writer.Alpha],
            Deltas=[self.Writer.Delta],
            tref=self.tref,
            BSGL=self.BSGL,
        )
        self.search.run()
        self.assertTrue(os.path.isfile(self.search.out_file))
        max2F_point = self.search.get_max_twoF()
        self.assertTrue(np.all(max2F_point["twoF"] >= self.search.data["twoF"]))
        search_keys = ["F0", "F1"]  # only the ones that aren't 0-width
        self._test_plots(search_keys)

    def test_grid_search_against_CFSv2(self):
        self.search = pyfstat.GridSearch(
            "grid_search",
            self.outdir,
            self.Writer.sftfilepath,
            F0s=self.F0s,
            F1s=[self.Writer.F1],
            F2s=[self.Writer.F2],
            Alphas=[self.Writer.Alpha],
            Deltas=[self.Writer.Delta],
            tref=self.tref,
        )
        self.search.run()
        self.assertTrue(os.path.isfile(self.search.out_file))
        pyfstat_out = pyfstat.helper_functions.read_txt_file_with_header(
            self.search.out_file, comments="#"
        )
        CFSv2_out_file = os.path.join(self.outdir, "CFSv2_Fstat_out.txt")
        CFSv2_loudest_file = os.path.join(self.outdir, "CFSv2_Fstat_loudest.txt")
        cl_CFSv2 = []
        cl_CFSv2.append("lalapps_ComputeFstatistic_v2")
        cl_CFSv2.append("--Alpha {} --AlphaBand 0".format(self.Alpha))
        cl_CFSv2.append("--Delta {} --DeltaBand 0".format(self.Delta))
        cl_CFSv2.append("--Freq {}".format(self.F0s[0]))
        cl_CFSv2.append("--FreqBand {}".format(self.F0s[1] - self.F0s[0]))
        cl_CFSv2.append("--dFreq {}".format(self.F0s[2]))
        cl_CFSv2.append("--f1dot {} --f1dotBand 0".format(self.F1))
        cl_CFSv2.append("--DataFiles '{}'".format(self.Writer.sftfilepath))
        cl_CFSv2.append("--refTime {}".format(self.tref))
        cl_CFSv2.append("--outputFstat " + CFSv2_out_file)
        cl_CFSv2.append("--outputLoudest " + CFSv2_loudest_file)
        # to match ComputeFstat default (and hence PyFstat) defaults on older
        # lalapps_CFSv2 versions, set the RngMedWindow manually:
        cl_CFSv2.append("--RngMedWindow=101")
        cl_CFSv2 = " ".join(cl_CFSv2)
        pyfstat.helper_functions.run_commandline(cl_CFSv2)
        self.assertTrue(os.path.isfile(CFSv2_out_file))
        self.assertTrue(os.path.isfile(CFSv2_loudest_file))
        CFSv2_out = pyfstat.helper_functions.read_txt_file_with_header(
            CFSv2_out_file, comments="%"
        )
        self.assertTrue(
            len(np.atleast_1d(CFSv2_out["2F"]))
            == len(np.atleast_1d(pyfstat_out["twoF"]))
        )
        self.assertTrue(np.max(np.abs(CFSv2_out["freq"] - pyfstat_out["F0"]) < 1e-16))
        self.assertTrue(np.max(np.abs(CFSv2_out["2F"] - pyfstat_out["twoF"]) < 1))
        self.assertTrue(np.max(CFSv2_out["2F"]) == np.max(pyfstat_out["twoF"]))
        self.search.generate_loudest()
        self.assertTrue(os.path.isfile(self.search.loudest_file))
        loudest = {}
        for run, f in zip(
            ["CFSv2", "PyFstat"], [CFSv2_loudest_file, self.search.loudest_file]
        ):
            loudest[run] = pyfstat.helper_functions.read_par(
                filename=f,
                suffix="loudest",
                raise_error=False,
            )
        for key in ["Alpha", "Delta", "Freq", "f1dot", "f2dot", "f3dot"]:
            self.assertTrue(
                np.abs(loudest["CFSv2"][key] - loudest["PyFstat"][key]) < 1e-16
            )
        self.assertTrue(
            np.abs(loudest["CFSv2"]["twoF"] - loudest["PyFstat"]["twoF"]) < 1
        )

    def test_semicoherent_grid_search(self):
        # FIXME this one doesn't check the results at all yet
        self.search = pyfstat.GridSearch(
            "sc_grid_search",
            self.outdir,
            self.Writer.sftfilepath,
            F0s=self.F0s,
            F1s=[self.Writer.F1],
            F2s=[self.Writer.F2],
            Alphas=[self.Writer.Alpha],
            Deltas=[self.Writer.Delta],
            tref=self.tref,
            nsegs=2,
            BSGL=self.BSGL,
        )
        self.search.run()
        self.assertTrue(os.path.isfile(self.search.out_file))
        search_keys = ["F0"]  # only the ones that aren't 0-width
        self._test_plots(search_keys)

    def test_glitch_grid_search(self):
        self.search = pyfstat.GridGlitchSearch(
            "grid_glitch_search",
            self.outdir,
            self.Writer.sftfilepath,
            F0s=self.F0s,
            F1s=self.F1s,
            F2s=[self.Writer.F2],
            Alphas=[self.Writer.Alpha],
            Deltas=[self.Writer.Delta],
            tref=self.tref,
            tglitchs=[self.tref],
            # BSGL=self.BSGL,  # not supported by this class
        )
        self.search.run()
        self.assertTrue(os.path.isfile(self.search.out_file))
        search_keys = ["F0", "F1"]  # only the ones that aren't 0-width
        self._test_plots(search_keys)


class TestGridSearchBSGL(TestGridSearch):
    label = "TestGridSearchBSGL"
    detectors = "H1,L1"
    BSGL = True

    def test_grid_search_on_data_with_line(self):
        # We reuse the default multi-IFO SFTs
        # but add an additional single-detector artifact to H1 only.
        # For simplicity, this is modelled here as a fully modulated CW-like signal,
        # just restricted to the single detector.
        SFTs_H1 = self.Writer.sftfilepath.split(";")[0]
        SFTs_L1 = self.Writer.sftfilepath.split(";")[1]
        extra_writer = pyfstat.Writer(
            label=self.label + "_with_line",
            outdir=self.outdir,
            tref=self.tref,
            F0=self.Writer.F0 + 0.0005,
            F1=self.Writer.F1,
            F2=self.Writer.F2,
            Alpha=self.Writer.Alpha,
            Delta=self.Writer.Delta,
            h0=10 * self.Writer.h0,
            cosi=self.Writer.cosi,
            sqrtSX=0,  # don't add yet another set of Gaussian noise
            noiseSFTs=SFTs_H1,
            SFTWindowType=self.Writer.SFTWindowType,
            SFTWindowBeta=self.Writer.SFTWindowBeta,
        )
        extra_writer.make_data()
        data_with_line = ";".join([SFTs_L1, extra_writer.sftfilepath])
        # now run a standard F-stat search over this data
        searchF = pyfstat.GridSearch(
            label="grid_search",
            outdir=self.outdir,
            sftfilepattern=data_with_line,
            F0s=self.F0s,
            F1s=[self.Writer.F1],
            F2s=[self.Writer.F2],
            Alphas=[self.Writer.Alpha],
            Deltas=[self.Writer.Delta],
            tref=self.tref,
            BSGL=False,
        )
        searchF.run()
        self.assertTrue(os.path.isfile(searchF.out_file))
        max2F_point_searchF = searchF.get_max_twoF()
        self.assertTrue(np.all(max2F_point_searchF["twoF"] >= searchF.data["twoF"]))
        # also run a BSGL search over the same data
        searchBSGL = pyfstat.GridSearch(
            label="grid_search",
            outdir=self.outdir,
            sftfilepattern=data_with_line,
            F0s=self.F0s,
            F1s=[self.Writer.F1],
            F2s=[self.Writer.F2],
            Alphas=[self.Writer.Alpha],
            Deltas=[self.Writer.Delta],
            tref=self.tref,
            BSGL=True,
        )
        searchBSGL.run()
        self.assertTrue(os.path.isfile(searchBSGL.out_file))
        max2F_point_searchBSGL = searchBSGL.get_max_twoF()
        self.assertTrue(
            np.all(max2F_point_searchBSGL["twoF"] >= searchBSGL.data["twoF"])
        )
        # Since we search the same grids and store all output,
        # the twoF from both searches should be the same.
        self.assertTrue(max2F_point_searchBSGL["twoF"] == max2F_point_searchF["twoF"])
        maxBSGL_point = searchBSGL.get_max_det_stat()
        self.assertTrue(
            np.all(maxBSGL_point["log10BSGL"] >= searchBSGL.data["log10BSGL"])
        )
        # The BSGL search should produce a lower max2F value than the F search.
        self.assertTrue(maxBSGL_point["twoF"] < max2F_point_searchF["twoF"])
        # But the maxBSGL_point should be the true multi-IFO signal
        # while max2F_point_searchF should have fallen for the single-IFO line.
        self.assertTrue(
            np.abs(maxBSGL_point["F0"] - self.F0)
            < np.abs(max2F_point_searchF["F0"] - self.F0)
        )


class TestTransientGridSearch(BaseForTestsWithData):
    label = "TestTransientGridSearch"
    F0s = [29.95, 30.05, 0.01]
    Band = 0.2

    def test_transient_grid_search(self):
        search = pyfstat.TransientGridSearch(
            "grid_search",
            self.outdir,
            self.Writer.sftfilepath,
            F0s=self.F0s,
            F1s=[self.Writer.F1],
            F2s=[self.Writer.F2],
            Alphas=[self.Writer.Alpha],
            Deltas=[self.Writer.Delta],
            tref=self.tref,
            minStartTime=self.Writer.tstart,
            maxStartTime=self.Writer.tend(),
            transientWindowType="rect",
            t0Band=self.Writer.duration - 2 * self.Writer.Tsft,
            tauBand=self.Writer.duration,
            outputTransientFstatMap=True,
            tCWFstatMapVersion="lal",
        )
        search.run()
        self.assertTrue(os.path.isfile(search.out_file))
        max2F_point = search.get_max_twoF()
        self.assertTrue(np.all(max2F_point["twoF"] >= search.data["twoF"]))
        tCWfile = search.get_transient_fstat_map_filename(max2F_point)
        tCW_out = pyfstat.helper_functions.read_txt_file_with_header(
            tCWfile, comments="#"
        )
        max2Fidx = np.argmax(tCW_out["2F"])
        self.assertTrue(
            np.isclose(max2F_point["twoF"], tCW_out["2F"][max2Fidx], rtol=1e-6, atol=0)
        )
        self.assertTrue(max2F_point["t0"] == tCW_out["t0s"][max2Fidx])
        self.assertTrue(max2F_point["tau"] == tCW_out["taus"][max2Fidx])


if __name__ == "__main__":
    unittest.main()
