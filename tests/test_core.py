import os
import unittest

import lalpulsar
import numpy as np
import pytest
from commons_for_tests import (
    default_binary_params,
    default_signal_params,
    default_Writer_params,
)
from scipy.stats import chi2

import pyfstat


@pytest.mark.usefixtures("outdir")
class TestReadParFile:
    label = "TestReadParFile"

    def test(self):
        parfile = os.path.join(self.outdir, self.label + ".par")
        os.system('echo "x=100\ny=10" > ' + parfile)

        par = pyfstat.utils.read_par(filename=parfile)
        assert par["x"] == 100
        assert par["y"] == 10

        par = pyfstat.utils.read_par(outdir=self.outdir, label=self.label)
        assert par["x"] == 100
        assert par["y"] == 10


@pytest.mark.usefixtures("outdir")
class TestPredictFstat:
    label = "TestPredictFstat"
    # here we only test the modes WITHOUT sftfilepattern,
    # which itself is tested through the Writer and Search classes

    def test_PFS_noise(self):
        twoF_expected, twoF_sigma = pyfstat.utils.predict_fstat(
            minStartTime=default_Writer_params["tstart"],
            duration=default_Writer_params["duration"],
            IFOs=default_Writer_params["detectors"],
            assumeSqrtSX=1,
        )
        print(
            "predict_fstat() returned: E[2F]={}+-{}".format(twoF_expected, twoF_sigma)
        )
        assert twoF_expected == 4
        assert np.isclose(twoF_sigma, chi2.std(df=4), rtol=1e-5)

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
        twoF_expected, twoF_sigma = pyfstat.utils.predict_fstat(
            timestampsFiles=",".join(TSfiles),
            IFOs=",".join(IFOs),
            assumeSqrtSX=1,
        )
        print(
            "predict_fstat() returned: E[2F]={}+-{}".format(twoF_expected, twoF_sigma)
        )
        assert twoF_expected == 4
        assert np.isclose(twoF_sigma, chi2.std(df=4), rtol=1e-5)

    def test_PFS_signal(self):
        duration = 10 * default_Writer_params["duration"]
        twoF_expected, twoF_sigma = pyfstat.utils.predict_fstat(
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
        assert twoF_expected > 4
        assert twoF_sigma > 0
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
        params = pyfstat.utils.get_predict_fstat_parameters_from_dict(params)
        twoF_expected_dict, twoF_sigma_dict = pyfstat.utils.predict_fstat(
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
        assert twoF_expected_dict == twoF_expected
        # add transient parameters
        params["transientWindowType"] = "rect"
        params["transient_tstart"] = default_Writer_params["tstart"]
        params["transient_duration"] = 0.5 * duration
        params = pyfstat.utils.get_predict_fstat_parameters_from_dict(params)
        (
            twoF_expected_transient,
            twoF_sigma_transient,
        ) = pyfstat.utils.predict_fstat(
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
        assert twoF_expected_transient < twoF_expected


class TestBaseSearchClass(unittest.TestCase):
    # TODO test the basic methods
    pass


@pytest.mark.usefixtures("data_fixture")
class TestComputeFstat:
    label = "TestComputeFstat"

    def test_run_computefstatistic_single_point_injectSqrtSX(self):
        # not using any SFTs
        search = pyfstat.ComputeFstat(
            tref=self.Writer.tref,
            minStartTime=self.Writer.tstart,
            maxStartTime=self.Writer.tstart + self.Writer.duration,
            detectors=self.Writer.detectors,
            injectSqrtSX=self.Writer.sqrtSX,
            minCoverFreq=self.signal_params["F0"] - 0.1,
            maxCoverFreq=self.signal_params["F0"] + 0.1,
        )
        # FIXME: This way is deprecated, remove test in future versions
        FS = search.get_fullycoherent_twoF(
            F0=self.signal_params["F0"],
            F1=self.signal_params["F1"],
            F2=self.signal_params["F2"],
            Alpha=self.signal_params["Alpha"],
            Delta=self.signal_params["Delta"],
        )
        assert FS > 0.0
        # now with new input params option
        FS_new = search.get_fullycoherent_twoF(
            params={
                "F0": self.signal_params["F0"],
                "F1": self.signal_params["F1"],
                "F2": self.signal_params["F2"],
                "Alpha": self.signal_params["Alpha"],
                "Delta": self.signal_params["Delta"],
            }
        )
        np.isclose(FS_new, FS, rtol=1e-6, atol=0)
        # now with higher spindowns
        FS_sd = search.get_fullycoherent_twoF(
            params={
                "F0": self.signal_params["F0"],
                "F1": self.signal_params["F1"],
                "F2": self.signal_params["F2"],
                # deliberately skipping F3 to test non-contiguous lists
                # (omtited terms should be assumed as 0)
                "F4": 1e-20,
                "Alpha": self.signal_params["Alpha"],
                "Delta": self.signal_params["Delta"],
            }
        )
        np.isclose(FS_sd, FS, rtol=1e-6, atol=0)
        # FIXME: extend test to properly test Fkdot matches?

    def test_run_computefstatistic_single_point_injectSqrtSX_binary(self):
        # not using any SFTs
        search = pyfstat.ComputeFstat(
            tref=self.Writer.tref,
            minStartTime=self.Writer.tstart,
            maxStartTime=self.Writer.tstart + self.Writer.duration,
            detectors=self.Writer.detectors,
            injectSqrtSX=self.Writer.sqrtSX,
            minCoverFreq=self.signal_params["F0"] - 0.1,
            maxCoverFreq=self.signal_params["F0"] + 0.1,
            binary=True,
        )
        # FIXME: This way is deprecated, remove test in future versions
        FS = search.get_fullycoherent_twoF(
            F0=self.signal_params["F0"],
            F1=self.signal_params["F1"],
            F2=self.signal_params["F2"],
            Alpha=self.signal_params["Alpha"],
            Delta=self.signal_params["Delta"],
            **default_binary_params,
        )
        assert FS > 0.0
        # now with new input params option
        params = {
            "F0": self.signal_params["F0"],
            "F1": self.signal_params["F1"],
            "F2": self.signal_params["F2"],
            "Alpha": self.signal_params["Alpha"],
            "Delta": self.signal_params["Delta"],
        }
        params.update(default_binary_params)
        FS = search.get_fullycoherent_twoF(params=params)
        assert FS > 0.0

    def test_run_computefstatistic_single_point_with_SFTs(self):
        twoF_predicted = self.Writer.predict_fstat()

        search = pyfstat.ComputeFstat(
            tref=self.Writer.tref,
            sftfilepattern=self.Writer.sftfilepath,
            search_ranges=self.search_ranges,
        )
        twoF = search.get_fullycoherent_twoF(
            F0=self.signal_params["F0"],
            F1=self.signal_params["F1"],
            F2=self.signal_params["F2"],
            Alpha=self.signal_params["Alpha"],
            Delta=self.signal_params["Delta"],
        )
        diff = np.abs(twoF - twoF_predicted) / twoF_predicted
        print(
            (
                "Predicted twoF is {}"
                " while recovered value is {},"
                " relative difference: {}".format(twoF_predicted, twoF, diff)
            )
        )
        assert diff < 0.3

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
            F0=self.signal_params["F0"],
            F1=self.signal_params["F1"],
            F2=self.signal_params["F2"],
            Alpha=self.signal_params["Alpha"],
            Delta=self.signal_params["Delta"],
        )
        diff = np.abs(twoF2 - twoF_predicted) / twoF_predicted
        print(
            (
                "Predicted twoF is {}"
                " while recovered value is {},"
                " relative difference: {}".format(twoF_predicted, twoF2, diff)
            )
        )
        assert diff < 0.3
        diff = np.abs(twoF2 - twoF) / twoF
        assert diff < 0.001

    def test_run_computefstatistic_allowedMismatchFromSFTLength(self):
        long_Tsft_params = default_Writer_params.copy()
        long_Tsft_params["Tsft"] = 3600
        long_Tsft_params["duration"] = 4 * long_Tsft_params["Tsft"]
        long_Tsft_params["label"] = "longTsft"
        long_Tsft_params["F0"] = 1500
        long_Tsft_params["Band"] = 2.0
        long_Tsft_Writer = pyfstat.Writer(
            outdir=self.outdir,
            **long_Tsft_params,
        )
        long_Tsft_Writer.run_makefakedata()
        search = pyfstat.ComputeFstat(
            tref=self.Writer.tref,
            sftfilepattern=long_Tsft_Writer.sftfilepath,
            minCoverFreq=long_Tsft_params["F0"] - 0.5,
            maxCoverFreq=long_Tsft_params["F0"] + 0.5,
            allowedMismatchFromSFTLength=0.1,
        )

        with pytest.raises(RuntimeError):
            search.get_fullycoherent_twoF(F0=1500, F1=0, F2=0, Alpha=0, Delta=0)

        search = pyfstat.ComputeFstat(
            tref=self.Writer.tref,
            sftfilepattern=long_Tsft_Writer.sftfilepath,
            minCoverFreq=long_Tsft_params["F0"] - 0.5,
            maxCoverFreq=long_Tsft_params["F0"] + 0.5,
            allowedMismatchFromSFTLength=0.5,
        )
        search.get_fullycoherent_twoF(
            F0=long_Tsft_params["F0"], F1=0, F2=0, Alpha=0, Delta=0
        )

    @pytest.mark.parametrize("binary", ["nobinary", "binary"])
    def test_run_computefstatistic_single_point_injectSources(self, binary):
        binary = binary == "binary"

        predicted_FS = self.Writer.predict_fstat()

        # first on-the-fly injection, via file loading,
        # and never with binary parameters
        injectSources = self.Writer.config_file_name
        search = pyfstat.ComputeFstat(
            tref=self.Writer.tref,
            assumeSqrtSX=1,
            injectSources=injectSources,
            minCoverFreq=self.signal_params["F0"] - 2,
            maxCoverFreq=self.signal_params["F0"] + 2,
            minStartTime=self.Writer.tstart,
            maxStartTime=self.Writer.tend,
            detectors=self.Writer.detectors,
        )
        # get Fstat also ignoring binary parameters
        FS_from_file = search.get_fullycoherent_twoF(
            F0=self.signal_params["F0"],
            F1=self.signal_params["F1"],
            F2=self.signal_params["F2"],
            Alpha=self.signal_params["Alpha"],
            Delta=self.signal_params["Delta"],
        )
        assert (
            np.abs(predicted_FS - FS_from_file) / FS_from_file < 0.3
        ), f"2F from on-the-fly CFS injection should be similar to predicted 2F, but got {FS_from_file} more than 30% off from {predicted_FS}"

        # second on-the-fly injection, this time with a dict,
        # and optionally with binary parameters
        injectSourcesdict = pyfstat.utils.read_par(filename=injectSources)
        injectSourcesdict["F0"] = injectSourcesdict.pop("Freq")
        injectSourcesdict["F1"] = injectSourcesdict.pop("f1dot")
        injectSourcesdict["F2"] = injectSourcesdict.pop("f2dot")
        injectSourcesdict["phi"] = injectSourcesdict.pop("phi0")
        if binary:
            injectSourcesdict.update(default_binary_params)
        search2 = pyfstat.ComputeFstat(
            tref=self.Writer.tref,
            assumeSqrtSX=1,
            injectSources=injectSourcesdict,
            minCoverFreq=self.signal_params["F0"] - 2,
            maxCoverFreq=self.signal_params["F0"] + 2,
            minStartTime=self.Writer.tstart,
            maxStartTime=self.Writer.tend,
            detectors=self.Writer.detectors,
            binary=True,
        )
        # recover without considering binary parameters
        FS_from_dict_nobinary = search2.get_fullycoherent_twoF(
            F0=self.signal_params["F0"],
            F1=self.signal_params["F1"],
            F2=self.signal_params["F2"],
            Alpha=self.signal_params["Alpha"],
            Delta=self.signal_params["Delta"],
            asini=0,
            period=0,
            ecc=0,
            tp=0,
            argp=0,
        )
        # recover with considering binary parameters
        FS_from_dict_binary = search2.get_fullycoherent_twoF(
            F0=self.signal_params["F0"],
            F1=self.signal_params["F1"],
            F2=self.signal_params["F2"],
            Alpha=self.signal_params["Alpha"],
            Delta=self.signal_params["Delta"],
            **default_binary_params,
        )
        if binary:
            assert (
                FS_from_dict_nobinary < FS_from_file
            ), f"2F from analysing binary injection while ignoring binary parameters should be smaller, but got {FS_from_dict_nobinary} >= {FS_from_file}"
            assert (
                FS_from_dict_binary > FS_from_dict_nobinary
            ), f"2F from analysing binary injection and searching over binary parameters should be bigger, but got {FS_from_dict_binary} <= {FS_from_dict_nobinary}"
            assert (
                np.abs(FS_from_dict_binary - predicted_FS) / predicted_FS < 0.3
            ), f"2F from on-the-fly CFS injection with correct binary treatment should be similar to predicted 2F, but got {FS_from_dict_binary} more than 30% off from {predicted_FS}"
        else:
            assert (
                FS_from_dict_nobinary == FS_from_file
            ), f"2F from injecting via dict should be the same as via file, but got {FS_from_dict_nobinary} != {FS_from_file}"
            assert (
                FS_from_dict_binary < FS_from_dict_nobinary
            ), f"2F from analysing non-binary injection with binary parameters should be smaller, but got {FS_from_dict_binary} >= {FS_from_dict_nobinary}"

    def test_get_fully_coherent_BSGL(self):
        # first pure noise, expect log10BSGL<0
        search_H1L1_noBSGL = pyfstat.ComputeFstat(
            tref=self.Writer.tref,
            minStartTime=self.Writer.tstart,
            maxStartTime=self.Writer.tstart + self.Writer.duration,
            detectors="H1,L1",
            injectSqrtSX=np.repeat(self.Writer.sqrtSX, 2),
            minCoverFreq=self.signal_params["F0"] - 0.1,
            maxCoverFreq=self.signal_params["F0"] + 0.1,
            BSGL=False,
            singleFstats=True,
            randSeed=self.Writer.randSeed,
        )
        twoF = search_H1L1_noBSGL.get_fullycoherent_detstat(
            F0=self.signal_params["F0"],
            F1=self.signal_params["F1"],
            F2=self.signal_params["F2"],
            Alpha=self.signal_params["Alpha"],
            Delta=self.signal_params["Delta"],
        )
        twoFX = search_H1L1_noBSGL.get_fullycoherent_single_IFO_twoFs()
        search_H1L1_BSGL = pyfstat.ComputeFstat(
            tref=self.Writer.tref,
            minStartTime=self.Writer.tstart,
            maxStartTime=self.Writer.tstart + self.Writer.duration,
            detectors="H1,L1",
            injectSqrtSX=np.repeat(self.Writer.sqrtSX, 2),
            minCoverFreq=self.signal_params["F0"] - 0.1,
            maxCoverFreq=self.signal_params["F0"] + 0.1,
            BSGL=True,
            randSeed=self.Writer.randSeed,
        )
        log10BSGL = search_H1L1_BSGL.get_fullycoherent_detstat(
            F0=self.signal_params["F0"],
            F1=self.signal_params["F1"],
            F2=self.signal_params["F2"],
            Alpha=self.signal_params["Alpha"],
            Delta=self.signal_params["Delta"],
        )
        assert log10BSGL < 0
        assert log10BSGL == lalpulsar.ComputeBSGL(
            twoF, twoFX, search_H1L1_BSGL.BSGLSetup
        )
        # now with an added signal, expect log10BSGL>0
        search_H1L1_noBSGL = pyfstat.ComputeFstat(
            tref=self.Writer.tref,
            minStartTime=self.Writer.tstart,
            maxStartTime=self.Writer.tstart + self.Writer.duration,
            detectors="H1,L1",
            injectSqrtSX=np.repeat(self.Writer.sqrtSX, 2),
            injectSources="{{Alpha={:g}; Delta={:g}; h0={:g}; cosi={:g}; Freq={:g}; f1dot={:g}; f2dot={:g}; refTime={:d};}}".format(
                self.signal_params["Alpha"],
                self.signal_params["Delta"],
                self.signal_params["h0"],
                self.signal_params["cosi"],
                self.signal_params["F0"],
                self.signal_params["F1"],
                self.signal_params["F2"],
                self.Writer.tref,
            ),
            minCoverFreq=self.signal_params["F0"] - 0.1,
            maxCoverFreq=self.signal_params["F0"] + 0.1,
            BSGL=False,
            singleFstats=True,
            randSeed=self.Writer.randSeed,
        )
        twoF = search_H1L1_noBSGL.get_fullycoherent_detstat(
            F0=self.signal_params["F0"],
            F1=self.signal_params["F1"],
            F2=self.signal_params["F2"],
            Alpha=self.signal_params["Alpha"],
            Delta=self.signal_params["Delta"],
        )
        twoFX = search_H1L1_noBSGL.get_fullycoherent_single_IFO_twoFs()
        search_H1L1_BSGL = pyfstat.ComputeFstat(
            tref=self.Writer.tref,
            minStartTime=self.Writer.tstart,
            maxStartTime=self.Writer.tstart + self.Writer.duration,
            detectors="H1,L1",
            injectSqrtSX=np.repeat(self.Writer.sqrtSX, 2),
            injectSources="{{Alpha={:g}; Delta={:g}; h0={:g}; cosi={:g}; Freq={:g}; f1dot={:g}; f2dot={:g}; refTime={:d};}}".format(
                self.signal_params["Alpha"],
                self.signal_params["Delta"],
                self.signal_params["h0"],
                self.signal_params["cosi"],
                self.signal_params["F0"],
                self.signal_params["F1"],
                self.signal_params["F2"],
                self.Writer.tref,
            ),
            minCoverFreq=self.signal_params["F0"] - 0.1,
            maxCoverFreq=self.signal_params["F0"] + 0.1,
            BSGL=True,
            randSeed=self.Writer.randSeed,
        )
        log10BSGL = search_H1L1_BSGL.get_fullycoherent_detstat(
            F0=self.signal_params["F0"],
            F1=self.signal_params["F1"],
            F2=self.signal_params["F2"],
            Alpha=self.signal_params["Alpha"],
            Delta=self.signal_params["Delta"],
        )
        assert log10BSGL > 0
        assert log10BSGL == lalpulsar.ComputeBSGL(
            twoF, twoFX, search_H1L1_BSGL.BSGLSetup
        )

    def test_transient_detstats(self):
        # first get maxTwoF and lnBtSG (from lalpulsar.ComputeTransientBstat) stats
        CFS_params = {
            "tref": self.Writer.tref,
            "minStartTime": self.Writer.tstart,
            "maxStartTime": self.Writer.tstart + self.Writer.duration,
            "detectors": "H1,L1",
            "injectSqrtSX": np.repeat(self.Writer.sqrtSX, 2),
            "randSeed": 42,
            "minCoverFreq": self.signal_params["F0"] - 0.1,
            "maxCoverFreq": self.signal_params["F0"] + 0.1,
            "transientWindowType": "rect",
            "t0Band": 2 * default_Writer_params["Tsft"],
            "tauBand": 2 * default_Writer_params["Tsft"],
            "tauMin": 2 * default_Writer_params["Tsft"],
            "tCWFstatMapVersion": "lal",
        }
        lambda_params = {
            "F0": self.signal_params["F0"],
            "F1": self.signal_params["F1"],
            "F2": self.signal_params["F2"],
            "Alpha": self.signal_params["Alpha"],
            "Delta": self.signal_params["Delta"],
        }
        search1 = pyfstat.ComputeFstat(
            **CFS_params,
            BtSG=True,
        )
        # standard way of getting the "main" detection statistic (as set by BtSG=True)
        lnBtSG1a = search1.get_fullycoherent_detstat(**lambda_params)
        # twoF1 = search1.twoF
        # maxTwoF1 = search1.maxTwoF
        print(f"twoF={search1.twoF}, maxTwoF={search1.maxTwoF}")
        # self.assertTrue(search1.maxTwoF >= search1.twoF) # FIXME: not always exactly true, can we define a more robust check?
        # recompute BtSG by calling one function level lower,
        # this should still redo the F-stat map and use the lalpulsar implementation
        lnBtSG1b = search1.get_transient_detstats()
        assert np.isclose(
            lnBtSG1a,
            lnBtSG1b,
            rtol=1e-4,
        ), f"lnBtSG: from get_fullycoherent_detstat() -> {lnBtSG1a}, from get_transient_detstats() -> {lnBtSG1b}"
        # recompute BtSG from the F-stat map using our own implementation
        lnBtSG1c = search1.FstatMap.get_lnBtSG()
        assert np.isclose(
            lnBtSG1a,
            lnBtSG1c,
            rtol=1e-4,
        ), f"lnBtSG: from get_fullycoherent_detstat() -> {lnBtSG1a}, from FstatMap.get_lnBtSG() -> {lnBtSG1c}"
        # recompute BtSG and other stats from a map saved to disk, using our own implementation
        tCWfile = os.path.join(self.outdir, "Fmn.txt")
        search1.FstatMap.write_F_mn_to_file(
            tCWfile, search1.windowRange, "testing a header"
        )
        Fmap_from_file = pyfstat.pyTransientFstatMap(from_file=tCWfile)
        Fmap_from_file.lnBtSG = Fmap_from_file.get_lnBtSG()
        maxidx = Fmap_from_file.get_maxF_idx()
        t0_ML = search1.windowRange.t0 + maxidx[0] * search1.windowRange.dt0
        tau_ML = search1.windowRange.tau + maxidx[1] * search1.windowRange.dtau
        shape1 = np.shape(search1.FstatMap.F_mn)
        shape2 = np.shape(Fmap_from_file.F_mn)
        assert (
            shape1 == shape2
        ), f"shape(search1.FstatMap.F_mn)={shape1}, shape(Fmap_from_file.F_mn)={shape2}"
        assert np.isclose(
            search1.FstatMap.maxF,
            Fmap_from_file.maxF,
            rtol=1e-4,
        ), f"search1.FstatMap.maxF={search1.FstatMap.maxF}, Fmap_from_file.maxF={Fmap_from_file.maxF}"
        assert np.isclose(
            search1.FstatMap.t0_ML,
            t0_ML,
            rtol=1e-4,
        ), f"search1.FstatMap.t0_ML={search1.FstatMap.t0_ML}, from file: t0_ML={t0_ML}"
        assert np.isclose(
            search1.FstatMap.tau_ML,
            tau_ML,
            rtol=1e-4,
        ), f"search1.FstatMap.tau_ML={search1.FstatMap.tau_ML}, from file: tau_ML={tau_ML}"
        assert np.isclose(
            search1.FstatMap.lnBtSG,
            Fmap_from_file.lnBtSG,
            rtol=1e-2,  # more tolerant than other checks due to implementation details
        ), f"search1.FstatMap.lnBtSG={search1.FstatMap.lnBtSG}, Fmap_from_file.lnBtSG={Fmap_from_file.lnBtSG}"

        # now set up for transient BSGL as "main" detection statistic instead
        search2 = pyfstat.ComputeFstat(
            **CFS_params,
            BSGL=True,
        )
        log10BSGL2a = search2.get_fullycoherent_detstat(**lambda_params)
        assert np.isclose(
            search1.twoF,
            search2.twoF,
            rtol=1e-4,
        ), f"search1.twoF={search1.twoF}, search2.twoF={search2.twoF}"
        Ndet = len(search2.detectors.split(","))
        for X in range(Ndet):
            print(search2.twoFX[X])
            assert (
                search2.twoFX[X] > 0
            ), f"search2.twoFX={search1.twoFX} but the first {Ndet} entries should be non-zero."
        assert np.isclose(
            search1.maxTwoF,
            search2.maxTwoF,
            rtol=1e-4,
        ), f"search1.maxTwoF={search1.maxTwoF}, search2.maxTwoF={search2.maxTwoF}"
        print(f"twoFXatMaxTwoF={search2.twoFXatMaxTwoF}")
        print(f"log10BSGL={search2.log10BSGL}")
        # recompute log10BSGL by calling one function level lower
        log10BSGL2b = search2.get_transient_detstats()
        assert np.isclose(
            log10BSGL2a,
            log10BSGL2b,
            rtol=1e-4,
        ), f"log10BSGL: from get_fullycoherent_detstat() -> {log10BSGL2a}, from get_transient_detstats() -> {log10BSGL2b}"
        # FIXME: add more quantitative tests of BSGL values

    def test_cumulative_twoF(self):
        Nsft = 100
        # not using any SFTs on disk
        search = pyfstat.ComputeFstat(
            tref=self.Writer.tref,
            minStartTime=self.Writer.tstart,
            maxStartTime=self.Writer.tstart + Nsft * self.Writer.Tsft,
            detectors=self.Writer.detectors,
            injectSqrtSX=self.Writer.sqrtSX,
            injectSources=default_signal_params,
            minCoverFreq=self.signal_params["F0"] - 0.1,
            maxCoverFreq=self.signal_params["F0"] + 0.1,
        )
        start_time, taus, twoF_cumulative = search.calculate_twoF_cumulative(
            self.signal_params["F0"],
            self.signal_params["F1"],
            self.signal_params["F2"],
            self.signal_params["Alpha"],
            self.signal_params["Delta"],
            num_segments=Nsft + 1,
        )
        twoF = search.get_fullycoherent_detstat(
            F0=self.signal_params["F0"],
            F1=self.signal_params["F1"],
            F2=self.signal_params["F2"],
            Alpha=self.signal_params["Alpha"],
            Delta=self.signal_params["Delta"],
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
        assert reldiff < 0.1
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
        assert reldiff < 0.1
        _, _, pfs, pfs_sigma = search.predict_twoF_cumulative(
            F0=self.signal_params["F0"],
            Alpha=self.signal_params["Alpha"],
            Delta=self.signal_params["Delta"],
            h0=self.signal_params["h0"],
            cosi=self.signal_params["cosi"],
            psi=self.signal_params["psi"],
            tstart=self.Writer.tstart,
            tend=self.Writer.tstart + Nsft * self.Writer.Tsft,
            IFOs=self.Writer.detectors,
            assumeSqrtSX=self.Writer.sqrtSX,
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
        assert reldiffmid < 0.25
        assert reldiffend < 0.25


@pytest.fixture
def CFS_default_params():
    # FIXME: `default_Writer_params` should be yet another fixture
    return {
        "tref": default_Writer_params["tstart"],
        "minStartTime": default_Writer_params["tstart"],
        "maxStartTime": default_Writer_params["tstart"]
        + default_Writer_params["duration"],
        "detectors": default_Writer_params["detectors"],
        "injectSqrtSX": default_Writer_params["sqrtSX"],
        "minCoverFreq": default_signal_params["F0"] - 0.1,
        "maxCoverFreq": default_signal_params["F0"] + 0.1,
    }


@pytest.fixture
def lambda_params():
    return {
        "F0": default_signal_params["F0"],
        "F1": default_signal_params["F1"],
        "F2": default_signal_params["F2"],
        "Alpha": default_signal_params["Alpha"],
        "Delta": default_signal_params["Delta"],
    }


@pytest.mark.parametrize("tCWFstatMapVersion", ["lal", "pycuda"])
@pytest.mark.parametrize("transientWindowType", [None, "rect"])
@pytest.mark.parametrize("cleanup", ["no", "manual", "contextmanager"])
def test_context_finalizer(
    CFS_default_params, lambda_params, tCWFstatMapVersion, transientWindowType, cleanup
):
    if cleanup == "manual" and not tCWFstatMapVersion == "pycuda":
        pytest.skip("Manual cleanup won't work in non-pycuda case.")

    # if GPU available, try the real thing;
    # else this should still set up the finalizer
    # but without actually trying to run on GPU
    if tCWFstatMapVersion == "pycuda":
        have_pycuda = pyfstat.tcw_fstat_map_funcs._optional_imports_pycuda()
        if not have_pycuda:
            pytest.skip("Optional imports failed, skipping actual pycuda test.")
        elif cleanup == "no":
            pytest.skip("This case might work but will sabotage others.")
    if cleanup == "contextmanager":
        with pyfstat.ComputeFstat(
            **CFS_default_params,
            tCWFstatMapVersion=tCWFstatMapVersion,
            transientWindowType=transientWindowType,
        ) as search:
            if tCWFstatMapVersion == "pycuda":
                assert search._finalizer is not None
                assert search._finalizer.alive
            detstat = search.get_fullycoherent_detstat(**lambda_params)
    else:
        search = pyfstat.ComputeFstat(
            **CFS_default_params,
            tCWFstatMapVersion=tCWFstatMapVersion,
            transientWindowType=transientWindowType,
        )
        if tCWFstatMapVersion == "pycuda":
            assert search._finalizer is not None
            assert search._finalizer.alive
        detstat = search.get_fullycoherent_detstat(**lambda_params)
        if cleanup == "manual":
            # calling finalizer manually should kill it
            search._finalizer()
            assert not search._finalizer.alive
    assert detstat > 0


def test_atoms_io(tmp_path, CFS_default_params, lambda_params):
    search = pyfstat.ComputeFstat(
        **CFS_default_params,
        computeAtoms=True,
    )
    search.get_fullycoherent_detstat(**lambda_params)
    atoms_orig = pyfstat.tcw_fstat_map_funcs.reshape_FstatAtomsVector(
        search.FstatResults.multiFatoms[0].data[0]
    )
    atomsdir = tmp_path
    search.write_atoms_to_file(atomsdir / "CFS")
    atomsfiles = [(atomsdir / f) for f in os.listdir(atomsdir) if "Fstatatoms" in f]
    assert len(atomsfiles) == 1
    atoms_read = pyfstat.utils.read_txt_file_with_header(atomsfiles[0], comments="%%")
    assert len(atoms_read.dtype.names) == 8
    assert len(atoms_read) == len(atoms_orig["timestamp"])
    for key_read in atoms_read.dtype.names:
        if key_read == "tGPS":
            key_orig = "timestamp"
        elif "_" in key_read:
            key_orig = key_read.replace("_", "_alpha_")
        else:
            key_orig = key_read + "_alpha"
        assert np.allclose(
            atoms_read[key_read], atoms_orig[key_orig], rtol=1e-6, atol=1e-6
        )


@pytest.mark.usefixtures("data_fixture")
class TestComputeFstatNoNoise:
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
            F0=self.signal_params["F0"],
            F1=self.signal_params["F1"],
            F2=self.signal_params["F2"],
            Alpha=self.signal_params["Alpha"],
            Delta=self.signal_params["Delta"],
        )
        assert np.abs(predicted_FS - FS) / FS < 0.3

    def test_run_computefstatistic_single_point_no_noise_manual_ephem(self):
        predicted_FS = self.Writer.predict_fstat(assumeSqrtSX=1)

        # let's get the default ephemeris files (to be sure their paths exist)
        # and then pretend we pass them manually, to test those class options
        (
            earth_ephem_default,
            sun_ephem_default,
        ) = pyfstat.utils.get_ephemeris_files()

        search = pyfstat.ComputeFstat(
            tref=self.Writer.tref,
            assumeSqrtSX=1,
            sftfilepattern=self.Writer.sftfilepath,
            earth_ephem=earth_ephem_default,
            sun_ephem=sun_ephem_default,
            search_ranges=self.search_ranges,
        )
        FS = search.get_fullycoherent_twoF(
            F0=self.signal_params["F0"],
            F1=self.signal_params["F1"],
            F2=self.signal_params["F2"],
            Alpha=self.signal_params["Alpha"],
            Delta=self.signal_params["Delta"],
        )
        assert np.abs(predicted_FS - FS) / FS < 0.3


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
        assert np.array_equal(a, b)

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

        assert np.array_equal(thetaB, search._shift_coefficients(thetaA, dT))

    def test_shift_coefficients_loop(self):
        search = pyfstat.SearchForSignalWithJumps()
        thetaA = np.array([10.0, 1e2, 10.0, 1e2])
        dT = 1e1
        thetaB = search._shift_coefficients(thetaA, dT)
        assert np.allclose(
            thetaA, search._shift_coefficients(thetaB, -dT), rtol=1e-9, atol=1e-9
        )


@pytest.mark.usefixtures("data_fixture")
class TestSemiCoherentSearch:
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
            self.signal_params["F0"],
            self.signal_params["F1"],
            self.signal_params["F2"],
            self.signal_params["Alpha"],
            self.signal_params["Delta"],
            record_segments=True,
        )
        twoF_per_seg_computed = np.array(search.twoF_per_segment)

        twoF_predicted = self.Writer.predict_fstat()
        # now compute the predicted semi-coherent Fstat for each segment
        twoF_per_seg_predicted = np.zeros(self.nsegs)
        for n in range(self.nsegs):
            twoF_per_seg_predicted[n], _ = pyfstat.utils.predict_fstat(
                h0=self.signal_params["h0"],
                cosi=self.signal_params["cosi"],
                psi=self.signal_params["psi"],
                Alpha=self.signal_params["Alpha"],
                Delta=self.signal_params["Delta"],
                F0=self.signal_params["F0"],
                sftfilepattern=self.Writer.sftfilepath,
                minStartTime=self.Writer.tstart
                + n * self.Writer.duration // self.nsegs,
                duration=self.Writer.duration // self.nsegs,
                IFOs=self.Writer.detectors,
                assumeSqrtSX=self.Writer.sqrtSX,
                tempory_filename=os.path.join(self.outdir, self.label + ".tmp"),
                transientWindowType=self.signal_params.get(
                    "transientWindowType", "none"
                ),
                transientStartTime=self.signal_params.get("transientStartTime", None),
                transientTau=self.signal_params.get("transientTau", None),
            )

        assert len(twoF_per_seg_computed) == len(twoF_per_seg_predicted)
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
        assert np.all(diffs < 0.3)
        diff = np.abs(twoF_sc - twoF_predicted) / twoF_predicted
        print(
            (
                "Predicted semicoherent twoF is {}"
                " while recovered value is {},"
                " relative difference: {}".format(twoF_predicted, twoF_sc, diff)
            )
        )
        assert diff < 0.3

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
            self.signal_params["F0"],
            self.signal_params["F1"],
            self.signal_params["F2"],
            self.signal_params["Alpha"],
            self.signal_params["Delta"],
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
            self.signal_params["F0"],
            self.signal_params["F1"],
            self.signal_params["F2"],
            self.signal_params["Alpha"],
            self.signal_params["Delta"],
            record_segments=True,
        )
        assert log10BSGL > 0
        assert log10BSGL == lalpulsar.ComputeBSGL(twoF, twoFX, search_BSGL.BSGLSetup)

    def test_get_semicoherent_BSGL_SFTs(self):
        dataopts = {
            "sftfilepattern": self.Writer.sftfilepath,
            "tref": self.Writer.tref,
            "search_ranges": self.search_ranges,
        }
        self._test_get_semicoherent_BSGL(**dataopts)

    def test_get_semicoherent_BSGL_inject(self):
        dataopts = {
            "tref": self.Writer.tref,
            "minStartTime": self.Writer.tstart,
            "maxStartTime": self.Writer.tstart + self.Writer.duration,
            "detectors": "H1,L1",
            "injectSqrtSX": np.repeat(self.Writer.sqrtSX, 2),
            "minCoverFreq": self.signal_params["F0"] - 0.1,
            "maxCoverFreq": self.signal_params["F0"] + 0.1,
            "injectSources": self.Writer.config_file_name,
            "randSeed": self.Writer.randSeed,
        }
        self._test_get_semicoherent_BSGL(**dataopts)

    def test_get_semicoherent_twoF_allowedMismatchFromSFTLength(self):
        long_Tsft_params = default_Writer_params.copy()
        long_Tsft_params["Tsft"] = 3600
        long_Tsft_params["duration"] = 4 * long_Tsft_params["Tsft"]
        long_Tsft_params["label"] = "longTsft"
        long_Tsft_params["F0"] = 1500
        long_Tsft_params["Band"] = 2.0
        long_Tsft_Writer = pyfstat.Writer(
            outdir=self.outdir,
            **long_Tsft_params,
        )
        long_Tsft_Writer.run_makefakedata()

        search = pyfstat.SemiCoherentSearch(
            label=self.label,
            outdir=self.outdir,
            tref=self.Writer.tref,
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
            tref=self.Writer.tref,
            sftfilepattern=long_Tsft_Writer.sftfilepath,
            nsegs=self.nsegs,
            minCoverFreq=1499.5,
            maxCoverFreq=1500.5,
            allowedMismatchFromSFTLength=0.5,
        )
        search.get_semicoherent_twoF(F0=1500, F1=0, F2=0, Alpha=0, Delta=0)


@pytest.mark.usefixtures("data_fixture")
class TestSemiCoherentGlitchSearch:
    label = "TestSemiCoherentGlitchSearch"
    dtglitch = 3600
    Band = 1

    def _run_test(self, delta_F0):
        Writer = pyfstat.GlitchWriter(
            self.label,
            outdir=self.outdir,
            tstart=self.Writer.tstart,
            duration=self.Writer.duration,
            dtglitch=self.dtglitch,
            delta_F0=delta_F0,
            detectors=self.Writer.detectors,
            sqrtSX=self.Writer.sqrtSX,
            **{
                k: v
                for k, v in default_signal_params.items()
                if not (k.startswith("F") and int(k[-1]) > 2)
            },
            SFTWindowType=self.Writer.SFTWindowType,
            SFTWindowParam=self.Writer.SFTWindowParam,
            randSeed=self.Writer.randSeed,
            Band=self.Writer.Band,
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
        Writer.signal_parameters["transientStartTime"] = Writer.tstart
        Writer.signal_parameters["transientTau"] = self.dtglitch
        FSA = Writer.predict_fstat()
        # same for the second half (tau stays the same)
        Writer.signal_parameters["transientStartTime"] += self.dtglitch
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
            assert diff < 0.3
        else:
            assert not diff < 0.3

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
            maxStartTime=Writer.tend,
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
        assert diff < 0.3
        diff2 = np.abs((twoF_glitch - twoF_sc_vanilla) / twoF_sc_vanilla)
        print(
            "Relative difference between SemiCoherentSearch"
            "and SemiCoherentGlitchSearch: {}".format(diff2)
        )
        if delta_F0 == 0:
            assert diff2 < 0.01
        else:
            assert twoF_glitch > twoF_sc_vanilla
            assert diff2 > 0.3

    def test_get_semicoherent_nglitch_twoF_no_glitch(self):
        self._run_test(delta_F0=0)

    def test_get_semicoherent_nglitch_twoF_with_glitch(self):
        self._run_test(delta_F0=0.1)
