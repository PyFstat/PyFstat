import unittest
import pyfstat
import numpy as np
import os


class TestWriter(unittest.TestCase):

    def test_make_cff(self):
        label = "Test"
        Writer = pyfstat.Writer(label, outdir='TestData')
        Writer.make_cff()
        self.assertTrue(os.path.isfile('./TestData/Test.cff'))

    def test_run_makefakedata(self):
        label = "Test"
        Writer = pyfstat.Writer(label, outdir='TestData')
        Writer.make_cff()
        Writer.run_makefakedata()
        self.assertTrue(os.path.isfile(
            './TestData/H-4800_H1_1800SFT_Test-700000000-8640000.sft'))

    def test_makefakedata_usecached(self):
        label = "Test"
        Writer = pyfstat.Writer(label, outdir='TestData')
        if os.path.isfile(Writer.sft_filepath):
            os.remove(Writer.sft_filepath)
        Writer.run_makefakedata()
        time_first = os.path.getmtime(Writer.sft_filepath)
        Writer.run_makefakedata()
        time_second = os.path.getmtime(Writer.sft_filepath)
        self.assertTrue(time_first == time_second)
        os.system('touch {}'.format(Writer.config_file_name))
        Writer.run_makefakedata()
        time_third = os.path.getmtime(Writer.sft_filepath)
        self.assertFalse(time_first == time_third)


class TestBaseSearchClass(unittest.TestCase):
    def test_shift_matrix(self):
        BSC = pyfstat.BaseSearchClass()
        dT = 10
        a = BSC.shift_matrix(4, dT)
        b = np.array([[1, 2*np.pi*dT, 2*np.pi*dT**2/2.0, 2*np.pi*dT**3/6.0],
                      [0, 1, dT, dT**2/2.0],
                      [0, 0, 1, dT],
                      [0, 0, 0, 1]])
        self.assertTrue(np.array_equal(a, b))

    def test_shift_coefficients(self):
        BSC = pyfstat.BaseSearchClass()
        thetaA = np.array([10., 1e2, 10., 1e2])
        dT = 100

        # Calculate the 'long' way
        thetaB = np.zeros(len(thetaA))
        thetaB[3] = thetaA[3]
        thetaB[2] = thetaA[2] + thetaA[3]*dT
        thetaB[1] = thetaA[1] + thetaA[2]*dT + .5*thetaA[3]*dT**2
        thetaB[0] = thetaA[0] + 2*np.pi*(thetaA[1]*dT + .5*thetaA[2]*dT**2
                                         + thetaA[3]*dT**3 / 6.0)

        self.assertTrue(
            np.array_equal(
                thetaB, BSC.shift_coefficients(thetaA, dT)))

    def test_shift_coefficients_loop(self):
        BSC = pyfstat.BaseSearchClass()
        thetaA = np.array([10., 1e2, 10., 1e2])
        dT = 1e1
        thetaB = BSC.shift_coefficients(thetaA, dT)
        self.assertTrue(
            np.allclose(
                thetaA, BSC.shift_coefficients(thetaB, -dT),
                rtol=1e-9, atol=1e-9))


class TestFullyCoherentNarrowBandSearch(unittest.TestCase):
    label = "Test"
    outdir = 'TestData'

    def test_compute_fstat(self):
        Writer = glitch_tools.Writer(self.label, outdir=self.outdir)
        Writer.make_data()

        search = glitch_searches.FullyCoherentNarrowBandSearch(
            self.label, self.outdir, tref=Writer.tref, Alpha=Writer.Alpha,
            Delta=Writer.Delta, duration=Writer.duration, tstart=Writer.tstart,
            Writer=Writer)
        search.run_computefstatistic_slow(m=1e-3, n=0)
        _, _, _, FS_max_slow = search.get_FS_max()

        search.run_computefstatistic(dFreq=0, numFreqBins=1)
        _, _, _, FS_max = search.get_FS_max()
        self.assertTrue(
            np.abs(FS_max-FS_max_slow)/FS_max_slow < 0.1)

    def test_compute_fstat_against_predict_fstat(self):
        Writer = glitch_tools.Writer(self.label, outdir=self.outdir)
        Writer.make_data()
        Writer.run_makefakedata()
        predicted_FS = Writer.predict_fstat()

        search = glitch_searches.FullyCoherentNarrowBandSearch(
            self.label, self.outdir, tref=Writer.tref, Alpha=Writer.Alpha,
            Delta=Writer.Delta, duration=Writer.duration, tstart=Writer.tstart,
            Writer=Writer)
        search.run_computefstatistic(dFreq=0, numFreqBins=1)
        _, _, _, FS_max = search.get_FS_max()
        self.assertTrue(np.abs(predicted_FS-FS_max)/FS_max < 0.5)


class TestSemiCoherentGlitchSearch(unittest.TestCase):
    label = "Test"
    outdir = 'TestData'

    def test_run_computefstatistic_single_point(self):
        Writer = pyfstat.Writer(self.label, outdir=self.outdir)
        Writer.make_data()
        predicted_FS = Writer.predict_fstat()

        search = pyfstat.SemiCoherentGlitchSearch(
            label=Writer.label, outdir=Writer.outdir, tref=Writer.tref,
            tstart=Writer.tstart, tend=Writer.tend)
        FS = search.run_computefstatistic_single_point(search.tref,
                                                       search.tstart,
                                                       search.tend,
                                                       Writer.F0,
                                                       Writer.F1,
                                                       Writer.F2,
                                                       Writer.Alpha,
                                                       Writer.Delta)
        print predicted_FS, FS
        self.assertTrue(np.abs(predicted_FS-FS)/FS < 0.1)

    def test_run_computefstatistic_single_point_slow(self):
        Writer = pyfstat.Writer(self.label, outdir=self.outdir)
        Writer.make_data()
        predicted_FS = Writer.predict_fstat()

        search = pyfstat.SemiCoherentGlitchSearch(
            label=Writer.label, outdir=Writer.outdir, tref=Writer.tref,
            tstart=Writer.tstart, tend=Writer.tend)
        FS = search.run_computefstatistic_single_point_slow(search.tref,
                                                            search.tstart,
                                                            search.tend,
                                                            Writer.F0,
                                                            Writer.F1,
                                                            Writer.F2,
                                                            Writer.Alpha,
                                                            Writer.Delta)
        self.assertTrue(np.abs(predicted_FS-FS)/FS < 0.1)

    def test_compute_glitch_fstat_slow(self):
        duration = 100*86400
        dtglitch = 100*43200
        delta_F0 = 0
        Writer = pyfstat.Writer(self.label, outdir=self.outdir,
                                duration=duration, dtglitch=dtglitch,
                                delta_F0=delta_F0)
        Writer.make_data()

        search = pyfstat.SemiCoherentGlitchSearch(
            label=Writer.label, outdir=Writer.outdir, tref=Writer.tref,
            tstart=Writer.tstart, tend=Writer.tend)

        FS = search.compute_glitch_fstat_slow(Writer.F0, Writer.F1, Writer.F2,
                                              Writer.Alpha, Writer.Delta,
                                              Writer.delta_F0, Writer.delta_F1,
                                              Writer.tglitch)

        # Compute the predicted semi-coherent glitch Fstat
        tstart = Writer.tstart
        tend = Writer.tend

        Writer.tend = tstart + dtglitch
        FSA = Writer.predict_fstat()

        Writer.tstart = tstart + dtglitch
        Writer.tend = tend
        FSB = Writer.predict_fstat()

        predicted_FS = .5*(FSA + FSB)

        print(predicted_FS, FS)
        self.assertTrue(np.abs((FS - predicted_FS))/predicted_FS < 0.1)

    def test_compute_nglitch_fstat(self):
        duration = 100*86400
        dtglitch = 100*43200
        delta_F0 = 0
        Writer = pyfstat.Writer(self.label, outdir=self.outdir,
                                duration=duration, dtglitch=dtglitch,
                                delta_F0=delta_F0)

        Writer.make_data()

        search = pyfstat.SemiCoherentGlitchSearch(
            label=Writer.label, outdir=Writer.outdir, tref=Writer.tref,
            tstart=Writer.tstart, tend=Writer.tend, nglitch=1)

        FS = search.compute_nglitch_fstat(Writer.F0, Writer.F1, Writer.F2,
                                          Writer.Alpha, Writer.Delta,
                                          Writer.delta_F0, Writer.delta_F1,
                                          search.tstart+dtglitch)

        # Compute the predicted semi-coherent glitch Fstat
        tstart = Writer.tstart
        tend = Writer.tend

        Writer.tend = tstart + dtglitch
        FSA = Writer.predict_fstat()

        Writer.tstart = tstart + dtglitch
        Writer.tend = tend
        FSB = Writer.predict_fstat()

        predicted_FS = (FSA + FSB)

        print(predicted_FS, FS)
        self.assertTrue(np.abs((FS - predicted_FS))/predicted_FS < 0.1)


class TestMCMCGlitchSearch(unittest.TestCase):
    label = "MCMCTest"
    outdir = 'TestData'

    def test_fully_coherent(self):
        h0 = 1e-24
        sqrtSX = 1e-22
        F0 = 30
        F1 = -1e-10
        F2 = 0
        tstart = 700000000
        duration = 100 * 86400
        tend = tstart + duration
        Alpha = 5e-3
        Delta = 1.2
        tref = tstart
        dtglitch = duration
        delta_F0 = 0
        Writer = pyfstat.Writer(F0=F0, F1=F1, F2=F2, label=self.label,
                                h0=h0, sqrtSX=sqrtSX,
                                outdir=self.outdir, tstart=tstart,
                                Alpha=Alpha, Delta=Delta, tref=tref,
                                duration=duration, dtglitch=dtglitch,
                                delta_F0=delta_F0, Band=4)

        Writer.make_data()
        predicted_FS = Writer.predict_fstat()

        theta = {'delta_F0': 0, 'delta_F1': 0, 'tglitch': tend,
                 'F0': {'type': 'norm', 'loc': F0, 'scale': np.abs(1e-9*F0)},
                 'F1': {'type': 'norm', 'loc': F1, 'scale': np.abs(1e-9*F1)},
                 'F2': F2, 'Alpha': Alpha, 'Delta': Delta}

        search = pyfstat.MCMCGlitchSearch(
            label=self.label, outdir=self.outdir, theta=theta, tref=tref,
            sftlabel=self.label, sftdir=self.outdir,
            tstart=tstart, tend=tend, nsteps=[100, 100], nwalkers=100,
            ntemps=1)
        search.run()
        search.plot_corner(add_prior=True)
        _, FS = search.get_max_twoF()

        print('Predicted twoF is {} while recovered is {}'.format(
                predicted_FS, FS))
        self.assertTrue(np.abs((FS - predicted_FS))/predicted_FS < 0.1)


if __name__ == '__main__':
    unittest.main()
