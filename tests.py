import unittest
import numpy as np
import os
import shutil
import pyfstat
import lalpulsar
import logging
import time


class Test(unittest.TestCase):
    outdir = "TestData"

    @classmethod
    def setUpClass(self):
        if os.path.isdir(self.outdir):
            try:
                shutil.rmtree(self.outdir)
            except OSError:
                logging.warning("{} not removed prior to tests".format(self.outdir))
        self.h0 = 1
        self.cosi = 0
        self.sqrtSX = 1
        self.F0 = 30
        self.F1 = -1e-10
        self.F2 = 0
        self.Tsft = 1800
        self.minStartTime = 700000000
        self.duration = 2 * 86400
        self.maxStartTime = self.minStartTime + self.duration
        self.Alpha = 5e-3
        self.Delta = 1.2
        self.tref = self.minStartTime
        self.detectors = "H1"
        self.SFTWindowType = "tukey"
        self.SFTWindowBeta = 1.0
        Writer = pyfstat.Writer(
            F0=self.F0,
            F1=self.F1,
            F2=self.F2,
            label="test",
            h0=self.h0,
            cosi=self.cosi,
            sqrtSX=self.sqrtSX,
            outdir=self.outdir,
            tstart=self.minStartTime,
            Alpha=self.Alpha,
            Delta=self.Delta,
            tref=self.tref,
            duration=self.duration,
            detectors=self.detectors,
            SFTWindowType=self.SFTWindowType,
            SFTWindowBeta=self.SFTWindowBeta,
            randSeed=None,
        )
        Writer.make_data()
        self.sftfilepath = Writer.sftfilepath
        self.search_keys = ["F0", "F1", "F2", "Alpha", "Delta"]
        self.search_ranges = {key: [getattr(self, key)] for key in self.search_keys}

    @classmethod
    def tearDownClass(self):
        if os.path.isdir(self.outdir):
            try:
                shutil.rmtree(self.outdir)
            except OSError:
                logging.warning("{} not removed prior to tests".format(self.outdir))


class TestWriter(Test):
    label = "TestWriter"
    writer_class_to_test = pyfstat.Writer
    tstart = 1094809861
    duration = 4 * 1800

    def test_make_cff(self):
        Writer = self.writer_class_to_test(
            label=self.label,
            outdir=self.outdir,
            tstart=self.tstart,
            duration=self.duration,
        )
        Writer.make_cff()
        self.assertTrue(
            os.path.isfile(os.path.join(".", self.outdir, self.label + ".cff"))
        )

    def test_run_makefakedata(self):
        duration = 4 * 1800
        Writer = self.writer_class_to_test(
            label=self.label, outdir=self.outdir, duration=duration, tstart=self.tstart
        )
        Writer.make_cff()
        Writer.run_makefakedata()
        expected_outfile = os.path.join(
            ".",
            Writer.outdir,
            "H-4_H1_1800SFT_{}-{}-{}.sft".format(
                Writer.label, Writer.tstart, Writer.duration
            ),
        )
        self.assertTrue(os.path.isfile(expected_outfile))
        cl_validate = "lalapps_SFTvalidate " + expected_outfile
        pyfstat.helper_functions.run_commandline(
            cl_validate, raise_error=True, return_output=False
        )

    def test_makefakedata_usecached(self):
        Writer = self.writer_class_to_test(
            label=self.label, outdir=self.outdir, duration=3600, tstart=self.tstart
        )
        if os.path.isfile(Writer.sftfilepath):
            os.remove(Writer.sftfilepath)
        # first run: make everything from scratch
        Writer.make_cff()
        Writer.run_makefakedata()
        time_first = os.path.getmtime(Writer.sftfilepath)
        # second run: should re-use .cff and .sft
        Writer.make_cff()
        Writer.run_makefakedata()
        time_second = os.path.getmtime(Writer.sftfilepath)
        self.assertTrue(time_first == time_second)
        # third run: touch the .cff to force regeneration
        time.sleep(1)  # make sure timestamp is actually different!
        os.system("touch {}".format(Writer.config_file_name))
        Writer.run_makefakedata()
        time_third = os.path.getmtime(Writer.sftfilepath)
        self.assertFalse(time_first == time_third)

    def test_noise_sfts(self):
        duration = 4 * self.Tsft
        h0 = 1000
        randSeed = 69420
        window = "tukey"
        window_beta = 0.01
        detectors = "L1,H1"

        # create sfts with a strong signal in them
        noise_and_signal_writer = self.writer_class_to_test(
            label="test_noiseSFTs_noise_and_signal",
            outdir=self.outdir,
            h0=h0,
            duration=duration,
            Tsft=self.Tsft,
            tstart=self.tstart,
            detectors=detectors,
            randSeed=randSeed,
            SFTWindowType=window,
            SFTWindowBeta=window_beta,
        )
        noise_and_signal_writer.make_data()

        # compute Fstat
        coherent_search = pyfstat.ComputeFstat(
            tref=noise_and_signal_writer.tref,
            sftfilepattern=noise_and_signal_writer.sftfilepath,
            search_ranges=self.search_ranges,
        )
        FS_1 = coherent_search.get_fullycoherent_twoF(
            noise_and_signal_writer.tstart,
            noise_and_signal_writer.tend,
            noise_and_signal_writer.F0,
            noise_and_signal_writer.F1,
            noise_and_signal_writer.F2,
            noise_and_signal_writer.Alpha,
            noise_and_signal_writer.Delta,
        )

        # create noise-only sfts first and then inject a strong signal
        noise_writer = self.writer_class_to_test(
            label="test_noiseSFTs_only_noise",
            outdir=self.outdir,
            h0=0,
            duration=duration,
            Tsft=self.Tsft,
            tstart=self.tstart,
            detectors=detectors,
            randSeed=randSeed,
            SFTWindowType=window,
            SFTWindowBeta=window_beta,
            Band=1,
        )
        noise_writer.make_data()

        # first inject without additional SFT loading constraints
        add_signal_writer = self.writer_class_to_test(
            label="test_noiseSFTs_add_signal",
            outdir=self.outdir,
            h0=h0,
            duration=None,
            Tsft=self.Tsft,
            tstart=None,
            detectors=detectors,
            sqrtSX=0,
            SFTWindowType=window,
            SFTWindowBeta=window_beta,
            noiseSFTs=noise_writer.sftfilepath,
        )
        add_signal_writer.make_data()

        # compute Fstat
        coherent_search = pyfstat.ComputeFstat(
            tref=add_signal_writer.tref,
            sftfilepattern=add_signal_writer.sftfilepath,
            search_ranges=self.search_ranges,
        )
        FS_2 = coherent_search.get_fullycoherent_twoF(
            add_signal_writer.tstart,
            add_signal_writer.tend,
            add_signal_writer.F0,
            add_signal_writer.F1,
            add_signal_writer.F2,
            add_signal_writer.Alpha,
            add_signal_writer.Delta,
        )

        self.assertTrue(np.abs(FS_1 - FS_2) / FS_1 < 0.01)

        # same again but with explicit (tstart,duration) to build constraints
        add_signal_writer_constr = self.writer_class_to_test(
            label="test_noiseSFTs_add_signal_with_constraints",
            outdir=self.outdir,
            h0=h0,
            duration=duration / 2,
            Tsft=self.Tsft,
            tstart=self.tstart,
            detectors=detectors,
            sqrtSX=0,
            SFTWindowType=window,
            SFTWindowBeta=window_beta,
            noiseSFTs=noise_writer.sftfilepath,
        )
        add_signal_writer_constr.make_data()

        # compute Fstat
        coherent_search = pyfstat.ComputeFstat(
            tref=add_signal_writer_constr.tref,
            sftfilepattern=add_signal_writer_constr.sftfilepath,
            search_ranges=self.search_ranges,
        )
        FS_3 = coherent_search.get_fullycoherent_twoF(
            add_signal_writer_constr.tstart,
            add_signal_writer_constr.tend,
            add_signal_writer_constr.F0,
            add_signal_writer_constr.F1,
            add_signal_writer_constr.F2,
            add_signal_writer_constr.Alpha,
            add_signal_writer_constr.Delta,
        )

        # we've used half the data, so expect half the FS
        self.assertTrue(np.abs(FS_1 - 2 * FS_3) / FS_1 < 0.05)

    def test_data_with_gaps(self):
        duration = 10 * self.Tsft
        gap_time = 4 * self.Tsft
        window = "tukey"
        window_beta = 0.01
        detectors = "H1"

        first_chunk_of_data = self.writer_class_to_test(
            label="first_chunk_of_data",
            outdir=self.outdir,
            duration=duration,
            Tsft=self.Tsft,
            tstart=self.tstart,
            detectors=detectors,
            SFTWindowType=window,
            SFTWindowBeta=window_beta,
        )
        first_chunk_of_data.make_data()

        second_chunk_of_data = self.writer_class_to_test(
            label="second_chunk_of_data",
            outdir=self.outdir,
            duration=duration,
            Tsft=self.Tsft,
            tstart=self.tstart + duration + gap_time,
            tref=self.tstart,
            detectors=detectors,
            SFTWindowType=window,
            SFTWindowBeta=window_beta,
        )
        second_chunk_of_data.make_data()

        both_chunks_of_data = self.writer_class_to_test(
            label="both_chunks_of_data",
            outdir=self.outdir,
            noiseSFTs=first_chunk_of_data.sftfilepath
            + ";"
            + second_chunk_of_data.sftfilepath,
            SFTWindowType=window,
            SFTWindowBeta=window_beta,
        )
        both_chunks_of_data.make_data()

        Tsft = both_chunks_of_data.Tsft
        total_duration = 2 * duration + gap_time
        Nsft = int((total_duration - gap_time) / Tsft)
        expected_SFT_filepath = (
            self.outdir
            + "/H-{}_H1_{}SFT_both_chunks_of_data-{}-{}.sft".format(
                Nsft, Tsft, self.tstart, total_duration
            )
        )
        self.assertTrue(os.path.isfile(expected_SFT_filepath))


class TestBinaryModulatedWriter(TestWriter):
    label = "TestBinaryModulatedWriter"
    writer_class_to_test = pyfstat.BinaryModulatedWriter


class TestGlitchWriter(TestWriter):

    label = "TestGlitchWriter"
    writer_class_to_test = pyfstat.GlitchWriter

    def test_glitch_injection(self):
        duration = 4 * 1800
        Band = 1
        vanillaWriter = pyfstat.Writer(
            label=self.label + "_vanilla",
            outdir=self.outdir,
            duration=duration,
            tstart=self.tstart,
            Band=Band,
        )
        vanillaWriter.make_cff()
        vanillaWriter.run_makefakedata()
        noGlitchWriter = self.writer_class_to_test(
            label=self.label + "_noglitch",
            outdir=self.outdir,
            duration=duration,
            tstart=self.tstart,
            Band=Band,
        )
        noGlitchWriter.make_cff()
        noGlitchWriter.run_makefakedata()
        glitchWriter = self.writer_class_to_test(
            label=self.label + "_glitch",
            outdir=self.outdir,
            duration=duration,
            tstart=self.tstart,
            Band=Band,
            dtglitch=2 * 1800,
            delta_F0=0.1,
        )
        glitchWriter.make_cff()
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
        self.assertEqual(times_noglitch, times_vanilla)
        self.assertEqual(times_glitch, times_vanilla)
        self.assertEqual(len(np.unique(max_freq_vanilla)), 1)
        self.assertEqual(len(np.unique(max_freq_noglitch)), 1)
        self.assertEqual(len(np.unique(max_freq_glitch)), 2)
        self.assertEqual(max_freq_noglitch[0], max_freq_vanilla[0])
        self.assertEqual(max_freq_glitch[0], max_freq_noglitch[0])
        self.assertTrue(max_freq_glitch[-1] > max_freq_noglitch[-1])


class TestBunch(Test):
    def test_bunch(self):
        b = pyfstat.core.Bunch(dict(x=10))
        self.assertTrue(b.x == 10)


class TestPar(Test):
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
        os.system("rm -r {}".format(self.outdir))


class TestBaseSearchClass(Test):
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


class TestComputeFstat(Test):
    label = "TestComputeFstat"

    def setup_method(self, method):
        self.Writer = pyfstat.Writer(
            label=self.label,
            outdir=self.outdir,
            duration=86400,
            tstart=1094809861,
            h0=1,
            sqrtSX=1,
            detectors="H1",
        )
        self.Writer.make_data()
        self.predicted_FS = self.Writer.predict_fstat()
        self.sftfilepattern = self.Writer.sftfilepath

    def test_run_computefstatistic_single_point_injectSqrtSX(self):

        search = pyfstat.ComputeFstat(
            tref=self.minStartTime,
            minStartTime=self.minStartTime,
            maxStartTime=self.maxStartTime,
            detectors=self.detectors,
            injectSqrtSX=self.sqrtSX,
            minCoverFreq=self.F0 - 0.1,
            maxCoverFreq=self.F0 + 0.1,
        )
        FS = search.get_fullycoherent_twoF(
            tstart=self.minStartTime,
            tend=self.maxStartTime,
            F0=self.F0,
            F1=self.F1,
            F2=self.F2,
            Alpha=self.Alpha,
            Delta=self.Delta,
        )
        self.assertTrue(FS > 0.0)

    def test_run_computefstatistic_single_point_with_SFTs(self):

        search = pyfstat.ComputeFstat(
            tref=self.Writer.tref,
            sftfilepattern=self.sftfilepattern,
            search_ranges=self.search_ranges,
        )
        FS = search.get_fullycoherent_twoF(
            self.Writer.tstart,
            self.Writer.tend,
            self.Writer.F0,
            self.Writer.F1,
            self.Writer.F2,
            self.Writer.Alpha,
            self.Writer.Delta,
        )
        self.assertTrue(np.abs(self.predicted_FS - FS) / FS < 0.3)

        # the following seems to be a leftover from when this test case was
        # doing separate H1 vs H1,L1 searches, but now only really tests the
        # SSBprec. But well, it still does add a tiny bit of coverage, can still
        # be replaced by something more systematic later.
        search = pyfstat.ComputeFstat(
            tref=self.Writer.tref,
            detectors=self.Writer.detectors,
            sftfilepattern=self.sftfilepattern,
            SSBprec=lalpulsar.SSBPREC_RELATIVISTIC,
            search_ranges=self.search_ranges,
        )
        FS2 = search.get_fullycoherent_twoF(
            self.Writer.tstart,
            self.Writer.tend,
            self.Writer.F0,
            self.Writer.F1,
            self.Writer.F2,
            self.Writer.Alpha,
            self.Writer.Delta,
        )
        self.assertTrue(np.abs(self.predicted_FS - FS2) / FS2 < 0.3)
        self.assertTrue(np.abs(FS2 - FS) / FS < 0.001)

    def test_run_computefstatistic_single_point_injectSources(self):

        injectSources = self.Writer.config_file_name
        search = pyfstat.ComputeFstat(
            tref=self.Writer.tref,
            assumeSqrtSX=1,
            injectSources=injectSources,
            minCoverFreq=28,
            maxCoverFreq=32,
            minStartTime=self.Writer.tstart,
            maxStartTime=self.Writer.tstart + self.Writer.duration,
            detectors=self.Writer.detectors,
        )
        FS_from_file = search.get_fullycoherent_twoF(
            self.Writer.tstart,
            self.Writer.tend,
            self.Writer.F0,
            self.Writer.F1,
            self.Writer.F2,
            self.Writer.Alpha,
            self.Writer.Delta,
        )
        # Writer.make_data()
        # predicted_FS = Writer.predict_fstat()
        self.assertTrue(np.abs(self.predicted_FS - FS_from_file) / FS_from_file < 0.3)

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
            maxStartTime=self.Writer.tstart + self.Writer.duration,
            detectors=self.Writer.detectors,
        )
        FS_from_dict = search.get_fullycoherent_twoF(
            self.Writer.tstart,
            self.Writer.tend,
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
            tref=self.minStartTime,
            minStartTime=self.minStartTime,
            maxStartTime=self.maxStartTime,
            detectors="H1,L1",
            injectSqrtSX=np.repeat(self.sqrtSX, 2),
            minCoverFreq=self.F0 - 0.1,
            maxCoverFreq=self.F0 + 0.1,
            BSGL=True,
        )
        lnBSGL = search_H1L1.get_fullycoherent_twoF(
            tstart=self.minStartTime,
            tend=self.maxStartTime,
            F0=self.F0,
            F1=self.F1,
            F2=self.F2,
            Alpha=self.Alpha,
            Delta=self.Delta,
        )
        self.assertTrue(lnBSGL < 0)
        # now with an added signal, expect lnBSGL>0
        search_H1L1 = pyfstat.ComputeFstat(
            tref=self.minStartTime,
            minStartTime=self.minStartTime,
            maxStartTime=self.maxStartTime,
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
            tstart=self.minStartTime,
            tend=self.maxStartTime,
            F0=self.F0,
            F1=self.F1,
            F2=self.F2,
            Alpha=self.Alpha,
            Delta=self.Delta,
        )
        self.assertTrue(lnBSGL > 0)


class TestComputeFstatNoNoise(Test):
    label = "TestComputeFstatSinglePointNoNoise"

    def setup_method(self, method):
        self.Writer = pyfstat.Writer(
            label=self.label,
            outdir=self.outdir,
            duration=86400,
            tstart=1094809861,
            h0=1,
        )
        self.Writer.make_data()
        self.predicted_FS = self.Writer.predict_fstat(assumeSqrtSX=1)

    def test_nonoise_Writer(self):
        # just a paranoia test that by using the default above,
        # we really get SFTs without noise
        Writer2 = pyfstat.Writer(
            label=self.label + "_sqrtSX0",
            outdir=self.outdir,
            duration=86400,
            tstart=1094809861,
            h0=1,
            sqrtSX=0,  # this should be the default, but test it explicitly
        )
        Writer2.make_data()
        predicted_FS_W2 = Writer2.predict_fstat(assumeSqrtSX=1)
        self.assertTrue(predicted_FS_W2 == self.predicted_FS)

    def test_run_computefstatistic_single_point_no_noise(self):

        search = pyfstat.ComputeFstat(
            tref=self.Writer.tref,
            assumeSqrtSX=1,
            sftfilepattern=os.path.join(
                self.Writer.outdir, "*{}-*sft".format(self.Writer.label)
            ),
            search_ranges=self.search_ranges,
        )
        FS = search.get_fullycoherent_twoF(
            self.Writer.tstart,
            self.Writer.tend,
            self.Writer.F0,
            self.Writer.F1,
            self.Writer.F2,
            self.Writer.Alpha,
            self.Writer.Delta,
        )
        self.assertTrue(np.abs(self.predicted_FS - FS) / FS < 0.3)

    def test_run_computefstatistic_single_point_no_noise_manual_ephem(self):

        # let's get the default ephemeris files (to be sure their paths exist)
        # and then pretend we pass them manually, to test those class options
        (
            earth_ephem_default,
            sun_ephem_default,
        ) = pyfstat.helper_functions.get_ephemeris_files()

        search = pyfstat.ComputeFstat(
            tref=self.Writer.tref,
            assumeSqrtSX=1,
            sftfilepattern=os.path.join(
                self.Writer.outdir, "*{}-*.sft".format(self.Writer.label)
            ),
            earth_ephem=earth_ephem_default,
            sun_ephem=sun_ephem_default,
            search_ranges=self.search_ranges,
        )
        FS = search.get_fullycoherent_twoF(
            self.Writer.tstart,
            self.Writer.tend,
            self.Writer.F0,
            self.Writer.F1,
            self.Writer.F2,
            self.Writer.Alpha,
            self.Writer.Delta,
        )
        self.assertTrue(np.abs(self.predicted_FS - FS) / FS < 0.3)


class TestSemiCoherentSearch(Test):
    label = "TestSemiCoherentSearch"

    def setup_method(self, method):
        self.Writer = pyfstat.Writer(
            label=self.label,
            outdir=self.outdir,
            duration=10 * 86400,
            tstart=1094809861,
            h0=1,
            sqrtSX=1,
            detectors="H1,L1",
        )
        self.Writer.make_data()

    def test_get_semicoherent_twoF(self):

        nsegs = 2
        search = pyfstat.SemiCoherentSearch(
            label=self.label,
            outdir=self.outdir,
            nsegs=nsegs,
            sftfilepattern=os.path.join(
                self.Writer.outdir, "*{}-*sft".format(self.Writer.label)
            ),
            tref=self.Writer.tref,
            minStartTime=self.Writer.tstart,
            maxStartTime=self.Writer.tend,
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
        self.Writer.duration /= nsegs
        tstart = self.Writer.tstart
        twoF_per_seg_predicted = np.zeros(nsegs)
        for n in range(nsegs):
            self.Writer.tstart = tstart + n * self.Writer.duration
            self.Writer.tend = tstart + (n + 1) * self.Writer.duration
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
        self.assertTrue(np.all(diffs < 0.2))
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
            nsegs=2,
            sftfilepattern=os.path.join(
                self.Writer.outdir, "*{}-*sft".format(self.Writer.label)
            ),
            tref=self.Writer.tref,
            minStartTime=self.Writer.tstart,
            maxStartTime=self.Writer.tend,
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


class TestSemiCoherentGlitchSearch(Test):
    label = "TestSemiCoherentGlitchSearch"

    def test_get_semicoherent_nglitch_twoF(self):
        duration = 10 * 86400
        dtglitch = 0.5 * duration
        delta_F0 = 0
        h0 = 1
        sqrtSX = 1
        Writer = pyfstat.GlitchWriter(
            self.label,
            outdir=self.outdir,
            tstart=self.minStartTime,
            duration=duration,
            dtglitch=dtglitch,
            delta_F0=delta_F0,
            sqrtSX=sqrtSX,
            h0=h0,
        )

        Writer.make_data()

        keys = ["F0", "F1", "F2", "Alpha", "Delta"]
        search_ranges = {
            key: [
                getattr(Writer, key),
                getattr(Writer, key) + getattr(Writer, "delta_" + key, 0.0),
            ]
            for key in keys
        }
        search = pyfstat.SemiCoherentGlitchSearch(
            label=self.label,
            outdir=self.outdir,
            sftfilepattern=os.path.join(Writer.outdir, "*{}-*sft".format(Writer.label)),
            tref=Writer.tref,
            minStartTime=Writer.tstart,
            maxStartTime=Writer.tend,
            nglitch=1,
            search_ranges=search_ranges,
        )

        FS = search.get_semicoherent_nglitch_twoF(
            Writer.F0,
            Writer.F1,
            Writer.F2,
            Writer.Alpha,
            Writer.Delta,
            Writer.delta_F0,
            Writer.delta_F1,
            search.minStartTime + dtglitch,
        )

        # Compute the predicted semi-coherent glitch Fstat for the first half
        Writer.transientStartTime = Writer.tstart
        Writer.transientTau = dtglitch
        FSA = Writer.predict_fstat()
        # same for the second half (tau stays the same)
        Writer.transientStartTime += dtglitch
        FSB = Writer.predict_fstat()
        predicted_FS = FSA + FSB

        self.assertTrue(np.abs((FS - predicted_FS)) / predicted_FS < 0.3)


class BaseForMCMCSearchTests(Test):
    # this class is only used for common setup of
    # tests for all MCMCSearch-based classes
    label = "TestMCMCSearch"
    randSeed = 42  # reduce chance of random failures in parameter recovery
    Band = 1

    def setup_method(self, method):
        self.Writer = pyfstat.Writer(
            F0=self.F0,
            F1=self.F1,
            F2=self.F2,
            label=self.label,
            h0=self.h0,
            sqrtSX=self.sqrtSX,
            outdir=self.outdir,
            tstart=self.minStartTime,
            Alpha=self.Alpha,
            Delta=self.Delta,
            tref=self.tref,
            duration=self.duration,
            randSeed=self.randSeed,
            Band=self.Band,
        )
        self.Writer.make_data()
        self.sftfilepath = self.Writer.sftfilepath
        self.twoF_predicted = self.Writer.predict_fstat()

    def _check_twoF_predicted(self):
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
                sftfilepattern=os.path.join(self.outdir, "*{}-*sft".format(self.label)),
                minStartTime=self.minStartTime,
                maxStartTime=self.maxStartTime,
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
        nsegs = 10
        self.search = pyfstat.MCMCSemiCoherentSearch(
            label=self.label,
            outdir=self.outdir,
            theta_prior=theta,
            tref=self.tref,
            sftfilepattern=os.path.join(self.outdir, "*{}-*sft".format(self.label)),
            minStartTime=self.minStartTime,
            maxStartTime=self.maxStartTime,
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

    def setup_method(self, method):
        self.duration = (
            10 * 86400
        )  # Supersky metric cannot be computed for segment lengths <= ~24 hours
        self.Writer = pyfstat.Writer(
            F0=self.F0,
            F1=self.F1,
            F2=self.F2,
            label=self.label,
            h0=self.h0,
            sqrtSX=self.sqrtSX,
            outdir=self.outdir,
            tstart=self.minStartTime,
            Alpha=self.Alpha,
            Delta=self.Delta,
            tref=self.tref,
            duration=self.duration,
            randSeed=self.randSeed,
            Band=self.Band,
        )
        self.Writer.make_data()
        self.sftfilepath = self.Writer.sftfilepath
        self.twoF_predicted = self.Writer.predict_fstat()

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
            sftfilepattern=os.path.join(self.outdir, "*{}-*sft".format(self.label)),
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

    def setup_method(self, method):
        self.transientWindowType = "rect"
        self.transientStartTime = self.minStartTime + 0.25 * self.duration
        self.transientTau = 0.5 * self.duration
        self.Writer = pyfstat.Writer(
            F0=self.F0,
            F1=self.F1,
            F2=self.F2,
            label=self.label,
            h0=self.h0,
            sqrtSX=self.sqrtSX,
            outdir=self.outdir,
            tstart=self.minStartTime,
            Alpha=self.Alpha,
            Delta=self.Delta,
            tref=self.tref,
            duration=self.duration,
            transientWindowType=self.transientWindowType,
            transientStartTime=self.transientStartTime,
            transientTau=self.transientTau,
            randSeed=self.randSeed,
            Band=self.Band,
        )
        self.Writer.make_data()
        self.sftfilepath = self.Writer.sftfilepath
        self.twoF_predicted = self.Writer.predict_fstat()

    def test_transient_MCMC(self):

        theta = {
            "F0": self.F0,
            "F1": self.F1,
            "F2": self.F2,
            "Alpha": self.Alpha,
            "Delta": self.Delta,
            "transient_tstart": {
                "type": "unif",
                "lower": self.minStartTime,
                "upper": self.maxStartTime - 2 * self.Tsft,
            },
            "transient_duration": {
                "type": "unif",
                "lower": 2 * self.Tsft,
                "upper": self.duration - 2 * self.Tsft,
            },
        }
        nsegs = 10
        self.search = pyfstat.MCMCTransientSearch(
            label=self.label,
            outdir=self.outdir,
            theta_prior=theta,
            tref=self.tref,
            sftfilepattern=os.path.join(self.outdir, "*{}-*sft".format(self.label)),
            minStartTime=self.minStartTime,
            maxStartTime=self.maxStartTime,
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


class BaseForGridSearchTests(Test):
    # this class is only used for common setup of
    # TestGridSearchCW and TestTransientGridSearch
    label = "TestGridSearch"

    def setup_method(self, method):
        self.Band = 2.5
        self.Writer = pyfstat.Writer(
            F0=self.F0,
            F1=self.F1,
            F2=self.F2,
            label=self.label,
            h0=self.h0,
            cosi=self.cosi,
            sqrtSX=self.sqrtSX,
            outdir=self.outdir,
            tstart=self.minStartTime,
            Alpha=self.Alpha,
            Delta=self.Delta,
            tref=self.tref,
            duration=self.duration,
            detectors=self.detectors,
            SFTWindowType=self.SFTWindowType,
            SFTWindowBeta=self.SFTWindowBeta,
            randSeed=None,
            Band=self.Band,
        )
        self.Writer.make_data()
        self.sftfilepath = self.Writer.sftfilepath


class TestGridSearch(BaseForGridSearchTests):
    label = "TestGridSearch"
    F0s = [29, 31, 0.1]
    F1s = [-1e-10, 0, 1e-11]

    def test_grid_search(self):
        search = pyfstat.GridSearch(
            "grid_search",
            self.outdir,
            self.sftfilepath,
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
            self.sftfilepath,
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
        cl_CFSv2.append("--DataFiles " + self.sftfilepath)
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
        search = pyfstat.GridSearch(
            "sc_grid_search",
            self.outdir,
            self.sftfilepath,
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
        search = pyfstat.SliceGridSearch(
            "slice_grid_search",
            self.outdir,
            self.sftfilepath,
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
            self.sftfilepath,
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
        search = pyfstat.FrequencySlidingWindow(
            "grid_grid_search",
            self.outdir,
            self.sftfilepath,
            F0s=self.F0s,
            F1=0,
            F2=0,
            Alpha=0,
            Delta=0,
            tref=self.tref,
            minStartTime=self.minStartTime,
            maxStartTime=self.maxStartTime,
        )
        search.run()
        self.assertTrue(os.path.isfile(search.out_file))


class TestTransientGridSearch(BaseForGridSearchTests):
    label = "TestTransientGridSearch"
    F0s = [29, 31, 0.1]

    def test_transient_grid_search(self):
        search = pyfstat.TransientGridSearch(
            "grid_search",
            self.outdir,
            self.sftfilepath,
            F0s=self.F0s,
            F1s=[0],
            F2s=[0],
            Alphas=[0],
            Deltas=[0],
            tref=self.tref,
            minStartTime=self.minStartTime,
            maxStartTime=self.maxStartTime,
            transientWindowType="rect",
            t0Band=self.duration - 3600,
            tauBand=self.duration,
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
