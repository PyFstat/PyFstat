import unittest
import numpy as np
import os
import shutil
import pyfstat
import lalpulsar
import logging
import time


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


default_signal_params = {
    "F0": 30,
    "F1": -1e-10,
    "F2": 0,
    "h0": 5,
    "cosi": 0,
    "Alpha": 5e-3,
    "Delta": 1.2,
}

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
    **default_signal_params,
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
        for key, val in default_Writer_params.items():
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


class TestWriter(BaseForTestsWithData):
    label = "TestWriter"
    writer_class_to_test = pyfstat.Writer

    # def setup_method(self, method):
    ## only setting up here, not actually running anything
    # self.Writer = self.writer_class_to_test(
    # label=self.label,
    # outdir=self.outdir,
    # tstart=self.tstart,
    # duration=self.duration,
    # detectors=self.detectors,
    # **default_signal_params,
    # )

    def test_make_cff(self):
        self.Writer.make_cff(verbose=True)
        self.assertTrue(
            os.path.isfile(os.path.join(".", self.outdir, self.label + ".cff"))
        )

    def test_run_makefakedata(self):
        self.Writer.make_data(verbose=True)
        expected_outfile = os.path.join(
            self.Writer.outdir,
            "{:1s}-{:d}_{:2s}_{:d}SFT_{:s}-{:d}-{:d}.sft".format(
                self.detectors[0],
                int(self.duration / self.Tsft),
                self.detectors,
                self.Tsft,
                self.Writer.label,
                self.Writer.tstart,
                self.Writer.duration,
            ),
        )
        self.assertTrue(os.path.isfile(expected_outfile))
        cl_validate = "lalapps_SFTvalidate " + expected_outfile
        pyfstat.helper_functions.run_commandline(
            cl_validate, raise_error=True, return_output=False
        )

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

    def test_noise_sfts(self):
        randSeed = 69420
        detectors = "L1,H1"

        # create SFTs with both noise and a signal in them
        noise_and_signal_writer = self.writer_class_to_test(
            label="test_noiseSFTs_noise_and_signal",
            outdir=self.outdir,
            duration=self.duration,
            Tsft=self.Tsft,
            tstart=self.tstart,
            detectors=detectors,
            randSeed=randSeed,
            SFTWindowType=self.SFTWindowType,
            SFTWindowBeta=self.SFTWindowBeta,
            sqrtSX=self.sqrtSX,
            **default_signal_params,
        )
        noise_and_signal_writer.make_data(verbose=True)
        # FIXME: here and everywhere below,
        # get_sft_array() only looks at first detector
        times, freqs, data = pyfstat.helper_functions.get_sft_array(
            noise_and_signal_writer.sftfilepath
        )
        max_values_noise_and_signal = np.max(data, axis=0)
        max_freqs_noise_and_signal = freqs[np.argmax(data, axis=0)]
        self.assertTrue(len(times) == self.duration / self.Tsft)
        # with signal: all SFTs should peak at same freq
        self.assertTrue(len(np.unique(max_freqs_noise_and_signal)) == 1)

        # create noise-only SFTs
        noise_writer = self.writer_class_to_test(
            label="test_noiseSFTs_only_noise",
            outdir=self.outdir,
            duration=self.duration,
            Tsft=self.Tsft,
            tstart=self.tstart,
            detectors=detectors,
            randSeed=randSeed,
            SFTWindowType=self.SFTWindowType,
            SFTWindowBeta=self.SFTWindowBeta,
            sqrtSX=self.sqrtSX,
            h0=0,
            F0=self.F0,
        )
        noise_writer.make_data(verbose=True)
        times, freqs, data = pyfstat.helper_functions.get_sft_array(
            noise_writer.sftfilepath
        )
        max_values_noise = np.max(data, axis=0)
        max_freqs_noise = freqs[np.argmax(data, axis=0)]
        self.assertEqual(len(max_freqs_noise), len(max_freqs_noise_and_signal))
        # pure noise: random peak freq in each SFT, lower max values
        self.assertFalse(len(np.unique(max_freqs_noise)) == 1)
        self.assertTrue(np.all(max_values_noise < max_values_noise_and_signal))

        # inject into noise-only SFTs without additional SFT loading constraints
        add_signal_writer = self.writer_class_to_test(
            label="test_noiseSFTs_add_signal",
            outdir=self.outdir,
            duration=None,
            Tsft=self.Tsft,
            tstart=None,
            SFTWindowType=self.SFTWindowType,
            SFTWindowBeta=self.SFTWindowBeta,
            sqrtSX=0,
            noiseSFTs=noise_writer.sftfilepath,
            **default_signal_params,
        )
        add_signal_writer.make_data(verbose=True)
        times, freqs, data = pyfstat.helper_functions.get_sft_array(
            add_signal_writer.sftfilepath
        )
        max_values_added_signal = np.max(data, axis=0)
        max_freqs_added_signal = freqs[np.argmax(data, axis=0)]
        self.assertEqual(len(max_freqs_added_signal), len(max_freqs_noise_and_signal))
        # peak freqs expected exactly equal to first case,
        # peak values can have a bit of numerical diff
        self.assertTrue(np.all(max_freqs_added_signal == max_freqs_noise_and_signal))
        self.assertTrue(
            np.allclose(
                max_values_added_signal, max_values_noise_and_signal, rtol=1e-6, atol=0
            )
        )

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
            **default_signal_params,
        )
        add_signal_writer_constr.make_data(verbose=True)
        times, freqs, data = pyfstat.helper_functions.get_sft_array(
            add_signal_writer_constr.sftfilepath
        )
        max_values_added_signal_constr = np.max(data, axis=0)
        max_freqs_added_signal_constr = freqs[np.argmax(data, axis=0)]
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

    def test_data_with_gaps(self):
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


class TestBinaryModulatedWriter(TestWriter):
    label = "TestBinaryModulatedWriter"
    writer_class_to_test = pyfstat.BinaryModulatedWriter


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
            times_vanilla,
            freqs_vanilla,
            data_vanilla,
        ) = pyfstat.helper_functions.get_sft_array(vanillaWriter.sftfilepath)
        (
            times_noglitch,
            freqs_noglitch,
            data_noglitch,
        ) = pyfstat.helper_functions.get_sft_array(noGlitchWriter.sftfilepath)
        (
            times_glitch,
            freqs_glitch,
            data_glitch,
        ) = pyfstat.helper_functions.get_sft_array(glitchWriter.sftfilepath)
        max_freq_vanilla = freqs_vanilla[np.argmax(data_vanilla, axis=0)]
        max_freq_noglitch = freqs_noglitch[np.argmax(data_noglitch, axis=0)]
        max_freq_glitch = freqs_glitch[np.argmax(data_glitch, axis=0)]
        print([max_freq_vanilla, max_freq_noglitch, max_freq_glitch])
        self.assertEqual(times_noglitch, times_vanilla)
        self.assertEqual(times_glitch, times_vanilla)
        self.assertEqual(len(np.unique(max_freq_vanilla)), 1)
        self.assertEqual(len(np.unique(max_freq_noglitch)), 1)
        self.assertEqual(len(np.unique(max_freq_glitch)), 2)
        self.assertEqual(max_freq_noglitch[0], max_freq_vanilla[0])
        self.assertEqual(max_freq_glitch[0], max_freq_noglitch[0])
        self.assertTrue(max_freq_glitch[-1] > max_freq_noglitch[-1])


class TestBunch(unittest.TestCase):
    def test_bunch(self):
        b = pyfstat.core.Bunch(dict(x=10))
        self.assertTrue(b.x == 10)


class TestPar(BaseForTestsWithOutdir):
    label = "TestPar"

    def test(self):
        parfile = os.path.join(self.outdir, self.label + ".par")
        os.system('echo "x=100\ny=10" > ' + parfile)

        par = pyfstat.core.read_par(parfile, return_type="Bunch")
        self.assertTrue(par.x == 100)
        self.assertTrue(par.y == 10)

        par = pyfstat.core.read_par(
            outdir=self.outdir, label=self.label, return_type="dict"
        )
        self.assertTrue(par["x"] == 100)
        self.assertTrue(par["y"] == 10)


class TestBaseSearchClass(BaseForTestsWithData):
    def test_shift_matrix(self):
        BSC = pyfstat.BaseSearchClass()
        dT = 10
        a = BSC._shift_matrix(4, dT)
        b = np.array(
            [
                [
                    1,
                    2 * np.pi * dT,
                    2 * np.pi * dT ** 2 / 2.0,
                    2 * np.pi * dT ** 3 / 6.0,
                ],
                [0, 1, dT, dT ** 2 / 2.0],
                [0, 0, 1, dT],
                [0, 0, 0, 1],
            ]
        )
        self.assertTrue(np.array_equal(a, b))

    def test_shift_coefficients(self):
        BSC = pyfstat.BaseSearchClass()
        thetaA = np.array([10.0, 1e2, 10.0, 1e2])
        dT = 100

        # Calculate the 'long' way
        thetaB = np.zeros(len(thetaA))
        thetaB[3] = thetaA[3]
        thetaB[2] = thetaA[2] + thetaA[3] * dT
        thetaB[1] = thetaA[1] + thetaA[2] * dT + 0.5 * thetaA[3] * dT ** 2
        thetaB[0] = thetaA[0] + 2 * np.pi * (
            thetaA[1] * dT + 0.5 * thetaA[2] * dT ** 2 + thetaA[3] * dT ** 3 / 6.0
        )

        self.assertTrue(np.array_equal(thetaB, BSC._shift_coefficients(thetaA, dT)))

    def test_shift_coefficients_loop(self):
        BSC = pyfstat.BaseSearchClass()
        thetaA = np.array([10.0, 1e2, 10.0, 1e2])
        dT = 1e1
        thetaB = BSC._shift_coefficients(thetaA, dT)
        self.assertTrue(
            np.allclose(
                thetaA, BSC._shift_coefficients(thetaB, -dT), rtol=1e-9, atol=1e-9
            )
        )


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
            tstart=self.tstart,
            tend=self.tstart + self.duration,
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
            self.Writer.tstart,
            self.Writer.tend(),
            self.Writer.F0,
            self.Writer.F1,
            self.Writer.F2,
            self.Writer.Alpha,
            self.Writer.Delta,
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
            self.Writer.tstart,
            self.Writer.tend(),
            self.Writer.F0,
            self.Writer.F1,
            self.Writer.F2,
            self.Writer.Alpha,
            self.Writer.Delta,
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
            self.Writer.tstart,
            self.Writer.tend(),
            self.Writer.F0,
            self.Writer.F1,
            self.Writer.F2,
            self.Writer.Alpha,
            self.Writer.Delta,
        )
        self.assertTrue(np.abs(predicted_FS - FS_from_file) / FS_from_file < 0.3)

        injectSourcesdict = pyfstat.core.read_par(injectSources)
        injectSourcesdict["F0"] = injectSourcesdict["Freq"]
        injectSourcesdict["F1"] = injectSourcesdict["f1dot"]
        injectSourcesdict["F2"] = injectSourcesdict["f2dot"]
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
            self.Writer.tstart,
            self.Writer.tend(),
            self.Writer.F0,
            self.Writer.F1,
            self.Writer.F2,
            self.Writer.Alpha,
            self.Writer.Delta,
        )
        self.assertTrue(FS_from_dict == FS_from_file)

    def test_get_fully_coherent_BSGL(self):
        # first pure noise, expect lnBSGL<0
        search_H1L1 = pyfstat.ComputeFstat(
            tref=self.tref,
            minStartTime=self.tstart,
            maxStartTime=self.tstart + self.duration,
            detectors="H1,L1",
            injectSqrtSX=np.repeat(self.sqrtSX, 2),
            minCoverFreq=self.F0 - 0.1,
            maxCoverFreq=self.F0 + 0.1,
            BSGL=True,
        )
        lnBSGL = search_H1L1.get_fullycoherent_twoF(
            tstart=self.tstart,
            tend=self.tstart + self.duration,
            F0=self.F0,
            F1=self.F1,
            F2=self.F2,
            Alpha=self.Alpha,
            Delta=self.Delta,
        )
        self.assertTrue(lnBSGL < 0)
        # now with an added signal, expect lnBSGL>0
        search_H1L1 = pyfstat.ComputeFstat(
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
        )
        lnBSGL = search_H1L1.get_fullycoherent_twoF(
            tstart=self.tstart,
            tend=self.tstart + self.duration,
            F0=self.F0,
            F1=self.F1,
            F2=self.F2,
            Alpha=self.Alpha,
            Delta=self.Delta,
        )
        self.assertTrue(lnBSGL > 0)


class TestComputeFstatNoNoise(BaseForTestsWithData):
    # FIXME: should be possible to merge into TestComputeFstat with smart
    # defaults handlingf
    label = "TestComputeFstatSinglePointNoNoise"
    sqrtSX = 0

    def test_run_computefstatistic_single_point_no_noise(self):

        predicted_FS = self.Writer.predict_fstat(assumeSqrtSX=1)
        shutil.copy(self.Writer.sftfilepath, ".")
        search = pyfstat.ComputeFstat(
            tref=self.Writer.tref,
            assumeSqrtSX=1,
            sftfilepattern=self.Writer.sftfilepath,
            search_ranges=self.search_ranges,
        )
        FS = search.get_fullycoherent_twoF(
            self.Writer.tstart,
            self.Writer.tend(),
            self.Writer.F0,
            self.Writer.F1,
            self.Writer.F2,
            self.Writer.Alpha,
            self.Writer.Delta,
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
            self.Writer.tstart,
            self.Writer.tend(),
            self.Writer.F0,
            self.Writer.F1,
            self.Writer.F2,
            self.Writer.Alpha,
            self.Writer.Delta,
        )
        self.assertTrue(np.abs(predicted_FS - FS) / FS < 0.3)


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

    def test_get_semicoherent_BSGL(self):

        search = pyfstat.SemiCoherentSearch(
            label=self.label,
            outdir=self.outdir,
            nsegs=self.nsegs,
            sftfilepattern=self.Writer.sftfilepath,
            tref=self.Writer.tref,
            search_ranges=self.search_ranges,
            BSGL=True,
        )

        BSGL = search.get_semicoherent_det_stat(
            self.Writer.F0,
            self.Writer.F1,
            self.Writer.F2,
            self.Writer.Alpha,
            self.Writer.Delta,
            record_segments=True,
        )
        self.assertTrue(BSGL > 0)


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

    def _check_twoF_predicted(self):
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
        self.assertTrue(diff < 0.3)

    def _check_mcmc_quantiles(self, transient=False):
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

        for k in inj.keys():
            reldiff = np.abs((self.max_dict[k] - inj[k]) / inj[k])
            print("max2F  {:s} reldiff: {:.2e}".format(k, reldiff))
            reldiff = np.abs((summary_stats[k]["mean"] - inj[k]) / inj[k])
            print("mean   {:s} reldiff: {:.2e}".format(k, reldiff))
            reldiff = np.abs((summary_stats[k]["median"] - inj[k]) / inj[k])
            print("median {:s} reldiff: {:.2e}".format(k, reldiff))
        for k in inj.keys():
            lower = summary_stats[k]["mean"] - nsigmas * summary_stats[k]["std"]
            upper = summary_stats[k]["mean"] + nsigmas * summary_stats[k]["std"]
            within = (inj[k] >= lower) and (inj[k] <= upper)
            print(
                "{:s} in mean+-{:d}std ({} in [{},{}])? {}".format(
                    k, nsigmas, inj[k], lower, upper, within
                )
            )
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
            self.assertTrue(within)


class TestMCMCSearch(BaseForMCMCSearchTests):
    label = "TestMCMCSearch"

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
                nsteps=[100, 100],
                nwalkers=100,
                ntemps=2,
                log10beta_min=-1,
            )
            self.search.run(plot_walkers=False)
            self.search.print_summary()
            self._check_twoF_predicted()
            self._check_mcmc_quantiles()


class TestMCMCSemiCoherentSearch(BaseForMCMCSearchTests):
    label = "TestMCMCSemiCoherentSearch"

    def test_semi_coherent_MCMC(self):

        theta = {
            "F0": {"type": "unif", "lower": self.F0 - 1e-6, "upper": self.F0 + 1e-6,},
            "F1": {"type": "unif", "lower": self.F1 - 1e-10, "upper": self.F1 + 1e-10,},
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


class TestMCMCFollowUpSearch(BaseForMCMCSearchTests):
    label = "TestMCMCFollowUpSearch"
    # Supersky metric cannot be computed for segment lengths <= ~24 hours
    duration = 10 * 86400
    # FIXME: if h0 too high for given duration, offsets to PFS become too large
    h0 = 0.1

    def test_MCMC_followup_search(self):

        theta = {
            "F0": {"type": "unif", "lower": self.F0 - 1e-6, "upper": self.F0 + 1e-6,},
            "F1": {"type": "unif", "lower": self.F1 - 1e-10, "upper": self.F1 + 1e-10,},
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
            plot_walkers=False, NstarMax=NstarMax, Nsegs0=nsegs,
        )
        self.search.print_summary()
        self._check_twoF_predicted()
        self._check_mcmc_quantiles()


class TestMCMCTransientSearch(BaseForMCMCSearchTests):
    label = "TestMCMCTransientSearch"
    duration = 86400

    def setup_method(self, method):
        self.transientWindowType = "rect"
        self.transientStartTime = self.tstart + 0.25 * self.duration
        self.transientTau = 0.5 * self.duration
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

    def test_transient_MCMC(self):

        theta = {
            "F0": self.F0,
            "F1": self.F1,
            "F2": self.F2,
            "Alpha": self.Alpha,
            "Delta": self.Delta,
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
        nsegs = 10
        self.search = pyfstat.MCMCTransientSearch(
            label=self.label,
            outdir=self.outdir,
            theta_prior=theta,
            tref=self.tref,
            sftfilepattern=self.Writer.sftfilepath,
            nsteps=[100, 100],
            nwalkers=100,
            ntemps=2,
            log10beta_min=-1,
            transientWindowType=self.transientWindowType,
        )
        self.search.run(plot_walkers=False)
        self.search.print_summary()
        self._check_twoF_predicted()
        self._check_mcmc_quantiles(transient=True)


class TestGridSearch(BaseForTestsWithData):
    label = "TestGridSearch"
    F0s = [29, 31, 0.1]
    F1s = [-1e-10, 0, 1e-11]
    Band = 2.5

    def test_grid_search(self):
        search = pyfstat.GridSearch(
            "grid_search",
            self.outdir,
            self.Writer.sftfilepath,
            F0s=self.F0s,
            F1s=[0],
            F2s=[0],
            Alphas=[0],
            Deltas=[0],
            tref=self.tref,
        )
        search.run()
        self.assertTrue(os.path.isfile(search.out_file))
        max2F_point = search.get_max_twoF()
        self.assertTrue(
            np.all(max2F_point["twoF"] >= search.data[:, search.keys.index("twoF")])
        )

    def test_grid_search_against_CFSv2(self):
        search = pyfstat.GridSearch(
            "grid_search",
            self.outdir,
            self.Writer.sftfilepath,
            F0s=self.F0s,
            F1s=[0],
            F2s=[0],
            Alphas=[0],
            Deltas=[0],
            tref=self.tref,
        )
        search.run()
        self.assertTrue(os.path.isfile(search.out_file))
        pyfstat_out = pyfstat.helper_functions.read_txt_file_with_header(
            search.out_file, comments="#"
        )
        CFSv2_out_file = os.path.join(self.outdir, "CFSv2_Fstat_out.txt")
        CFSv2_loudest_file = os.path.join(self.outdir, "CFSv2_Fstat_loudest.txt")
        cl_CFSv2 = []
        cl_CFSv2.append("lalapps_ComputeFstatistic_v2")
        cl_CFSv2.append("--Alpha 0 --Delta 0")
        cl_CFSv2.append("--AlphaBand 0 --DeltaBand 0")
        cl_CFSv2.append("--Freq {}".format(self.F0s[0]))
        cl_CFSv2.append("--f1dot 0 --f1dotBand 0 --df1dot 0")
        cl_CFSv2.append("--FreqBand {}".format(self.F0s[1] - self.F0s[0]))
        cl_CFSv2.append("--dFreq {}".format(self.F0s[2]))
        cl_CFSv2.append("--DataFiles " + self.Writer.sftfilepath)
        cl_CFSv2.append("--refTime {}".format(self.tref))
        earth_ephem, sun_ephem = pyfstat.helper_functions.get_ephemeris_files()
        if earth_ephem is not None:
            cl_CFSv2.append('--ephemEarth="{}"'.format(earth_ephem))
        if sun_ephem is not None:
            cl_CFSv2.append('--ephemSun="{}"'.format(sun_ephem))
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

    def test_semicoherent_grid_search(self):
        # FIXME this one doesn't check the results at all yet
        search = pyfstat.GridSearch(
            "sc_grid_search",
            self.outdir,
            self.Writer.sftfilepath,
            F0s=self.F0s,
            F1s=[0],
            F2s=[0],
            Alphas=[0],
            Deltas=[0],
            tref=self.tref,
            nsegs=2,
        )
        search.run()
        self.assertTrue(os.path.isfile(search.out_file))

    def test_slice_grid_search(self):
        # FIXME this one doesn't check the results at all yet
        search = pyfstat.SliceGridSearch(
            "slice_grid_search",
            self.outdir,
            self.Writer.sftfilepath,
            F0s=self.F0s,
            F1s=self.F1s,
            F2s=[0.0],
            Alphas=[0.0],
            Deltas=[0.0],
            tref=self.tref,
            Lambda0=[30.0, 0.0, 0.0, 0.0],
        )
        fig, axes = search.run(save=False)
        self.assertTrue(fig is not None)

    def test_glitch_grid_search(self):
        search = pyfstat.GridGlitchSearch(
            "grid_grid_search",
            self.outdir,
            self.Writer.sftfilepath,
            F0s=self.F0s,
            F1s=self.F1s,
            F2s=[0],
            Alphas=[0],
            Deltas=[0],
            tref=self.tref,
            tglitchs=[self.tref],
        )
        search.run()
        self.assertTrue(os.path.isfile(search.out_file))

    def test_sliding_window(self):
        # FIXME this one doesn't check the results at all yet
        search = pyfstat.FrequencySlidingWindow(
            "grid_grid_search",
            self.outdir,
            self.Writer.sftfilepath,
            F0s=self.F0s,
            F1=0,
            F2=0,
            Alpha=0,
            Delta=0,
            tref=self.tref,
            minStartTime=self.Writer.tstart,
            maxStartTime=self.Writer.tend(),
        )
        search.run()
        self.assertTrue(os.path.isfile(search.out_file))


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
            F1s=[0],
            F2s=[0],
            Alphas=[0],
            Deltas=[0],
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
        self.assertTrue(
            np.all(max2F_point["twoF"] >= search.data[:, search.keys.index("twoF")])
        )
        tCWfile = (
            search.tCWfilebase
            + "{:.16f}_{:.16f}_{:.16f}_{:.16g}_{:.16g}.dat".format(
                max2F_point["F0"],
                max2F_point["Alpha"],
                max2F_point["Delta"],
                max2F_point["F1"],
                max2F_point["F2"],
            )
        )
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
