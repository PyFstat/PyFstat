import os

import numpy as np

# FIXME this should be made cleaner with fixtures
from commons_for_tests import BaseForTestsWithData, default_binary_params

import pyfstat


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
            self.search.plot_1D(xkey=key, x0=self.Writer.F0, savefig=True)
        if len(search_keys) == 2:
            self.search.plot_2D(
                xkey=search_keys[0],
                ykey=search_keys[1],
                x0=self.Writer.F0,
                y0=self.Writer.F1,
                colorbar=True,
            )
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
        pyfstat_out = pyfstat.utils.read_txt_file_with_header(
            self.search.out_file, comments="#"
        )
        CFSv2_out_file = os.path.join(self.outdir, "CFSv2_Fstat_out.txt")
        CFSv2_loudest_file = os.path.join(self.outdir, "CFSv2_Fstat_loudest.txt")
        cl_CFSv2 = []
        cl_CFSv2.append("lalpulsar_ComputeFstatistic_v2")
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
        # CFSv2 versions, set the RngMedWindow manually:
        cl_CFSv2.append("--RngMedWindow=101")
        cl_CFSv2 = " ".join(cl_CFSv2)
        pyfstat.utils.run_commandline(cl_CFSv2)
        self.assertTrue(os.path.isfile(CFSv2_out_file))
        self.assertTrue(os.path.isfile(CFSv2_loudest_file))
        CFSv2_out = pyfstat.utils.read_txt_file_with_header(
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
            loudest[run] = pyfstat.utils.read_par(
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
            label=self.label + "WithLine",
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
            SFTWindowParam=self.Writer.SFTWindowParam,
        )
        extra_writer.make_data()
        data_with_line = ";".join([SFTs_L1, extra_writer.sftfilepath])
        # now run a standard F-stat search over this data
        searchF = pyfstat.GridSearch(
            label="GridSearch",
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
            label="GridSearch",
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

    def test_transient_grid_search(self, transient=True, BtSG=False):
        if transient:
            transient_params = {
                "minStartTime": self.Writer.tstart,
                "maxStartTime": self.Writer.tend,
                "transientWindowType": "rect",
                "t0Band": self.Writer.duration - 2 * self.Writer.Tsft,
                "tauBand": self.Writer.duration,
                "outputTransientFstatMap": True,
                "tCWFstatMapVersion": "lal",
                "BtSG": BtSG,
            }
        else:
            transient_params = {}
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
            **transient_params,
            outputAtoms=True,
        )
        search.run()
        self.assertTrue(os.path.isfile(search.out_file))
        max2F_point = search.get_max_twoF()
        self.assertTrue(np.all(max2F_point["twoF"] >= search.data["twoF"]))
        if transient:
            tCWfile = search.get_transient_fstat_map_filename(max2F_point)
            tCW_out = pyfstat.utils.read_txt_file_with_header(tCWfile, comments="#")
            max2Fidx = np.argmax(tCW_out["2F"])
            self.assertTrue(
                np.isclose(
                    max2F_point["twoF"], tCW_out["2F"][max2Fidx], rtol=1e-6, atol=0
                )
            )
            self.assertTrue(max2F_point["t0"] == tCW_out["t0s"][max2Fidx])
            self.assertTrue(max2F_point["tau"] == tCW_out["taus"][max2Fidx])
        if BtSG:
            self.assertTrue(hasattr(search.search, "lnBtSG"))

    def test_transient_grid_search_notransient(self):
        self.test_transient_grid_search(transient=False)

    def test_transient_grid_search_BtSG(self):
        self.test_transient_grid_search(transient=True, BtSG=True)


class TestSearchOverGridFile(BaseForTestsWithData):
    label = "TestSearchOverGridFile"
    # Need to hand-pick values F0s here for the CFSv2 comparison:
    # that code sometimes includes endpoints, sometimes not.
    # For the F0s here, it happens to match our convention (include endpoint).
    F0s = [29.999, 30.001, 1e-4]
    F3s = np.logspace(-29, -28, 2)
    Band = 0.5

    def _write_gridfile(self, binary=False):
        # This is a simple test just looping over two parameters, F0 and F3,
        # and writing the gridfile by hand.
        self.gridfile = os.path.join(self.outdir, "test_grid.txt")
        with open(self.gridfile, "w") as fp:
            fp.write("%% columns:\n")
            fp.write("%% freq alpha delta f1dot f2dot f3dot")
            if binary:
                for key in default_binary_params.keys():
                    fp.write(f" {key}")
            fp.write("\n")
            # to match CFSv2, F0 must be the innermost loop
            for F3 in self.F3s:
                for F0 in np.arange(*self.F0s):
                    fp.write(
                        f"{F0:.16g} {self.Writer.Alpha:.16g} {self.Writer.Delta:.16g} {self.Writer.F1:.16g} {self.Writer.F2:.16g}  {F3:.16g}"
                    )
                    if binary:
                        for key in default_binary_params.keys():
                            fp.write(f" {default_binary_params[key]:.16g}")
                    fp.write("\n")

    def _write_gridfile_with_CFSv2(self):
        # Here we let the CFSv2 itself create the gridfile
        # and loop over frequency plus three spin-down parameters.
        self.gridfile = os.path.join(self.outdir, "test_grid_CFSv2.txt")
        cl_CFSv2 = []
        cl_CFSv2.append("lalpulsar_ComputeFstatistic_v2")
        cl_CFSv2.append(f"--Freq {self.F0s[0]:.16g}")
        cl_CFSv2.append(f"--FreqBand {(self.F0s[1] - self.F0s[0]):.16g}")
        cl_CFSv2.append(f"--Alpha {self.Writer.Alpha:.16g}")
        cl_CFSv2.append(f"--Delta {self.Writer.Delta:.16g}")
        cl_CFSv2.append(f"--f1dot {self.Writer.F1:.16g}")
        cl_CFSv2.append(f"--f1dotBand {3e-7:.16g}")
        cl_CFSv2.append(f"--f2dot {self.Writer.F2:.16g}")
        cl_CFSv2.append(f"--f2dotBand {3e-11:.16g}")
        cl_CFSv2.append(f"--f3dot {self.F3s[0]:.16g}")
        cl_CFSv2.append(f"--f3dotBand {3e-15:.16g}")
        cl_CFSv2.append("--gridType 8")
        cl_CFSv2.append("--DataFiles '{}'".format(self.Writer.sftfilepath))
        cl_CFSv2.append("--refTime {}".format(self.tref))
        cl_CFSv2.append(f"--metricMismatch {0.02:.16g}")
        cl_CFSv2.append("--countTemplates TRUE")
        cl_CFSv2.append("--strictSpindownBounds TRUE")
        cl_CFSv2.append(f"--outputGrid {self.gridfile}")
        # to match ComputeFstat default (and hence PyFstat) defaults on older
        # CFSv2 versions, set the RngMedWindow manually:
        cl_CFSv2.append("--RngMedWindow=101")
        cl_CFSv2 = " ".join(cl_CFSv2)
        pyfstat.utils.run_commandline(cl_CFSv2)

    def test_gridfile_search(
        self, binary=False, transient=False, grid="manual", reading_method="numpy"
    ):
        if grid == "manual":
            self._write_gridfile(binary)
        elif grid == "CFSv2":
            self._write_gridfile_with_CFSv2()

        if transient:
            transient_search_params = {
                "transientWindowType": "rect",
                "t0Band": self.Writer.duration - 2 * self.Writer.Tsft,
                "tauBand": self.Writer.duration,
                "outputTransientFstatMap": True,
                "tCWFstatMapVersion": "lal",
            }
        else:
            transient_search_params = {}

        search = pyfstat.SearchOverGridFile(
            label=f"grid_search{'_binary' if binary else ''}",
            outdir=self.outdir,
            sftfilepattern=self.Writer.sftfilepath,
            gridfile=self.gridfile,
            tref=self.tref,
            reading_method=reading_method,
            **transient_search_params,
        )
        search.run()
        self.assertTrue(os.path.isfile(search.out_file))
        pyfstat_out = pyfstat.utils.read_txt_file_with_header(
            search.out_file, comments="#"
        )
        CFSv2_out_file = os.path.join(self.outdir, "CFSv2_Fstat_from_gridfile_out.txt")

        CFSv2_loudest_file = os.path.join(
            self.outdir, "CFSv2_Fstat_from_gridfile_loudest.txt"
        )
        cl_CFSv2 = []
        cl_CFSv2.append("lalpulsar_ComputeFstatistic_v2")
        if binary:
            # CFSv2 would ignore binary parameters in gridfile,
            # so set everything manually
            cl_CFSv2.append(f"--Freq {self.F0s[0]:.16g}")
            cl_CFSv2.append(f"--FreqBand {self.F0s[1] - self.F0s[0]:.16g}")
            cl_CFSv2.append(f"--dFreq {self.F0s[2]:.16g}")
            cl_CFSv2.append(f"--Alpha {self.Writer.Alpha:.16g}")
            cl_CFSv2.append(f"--Delta {self.Writer.Delta:.16g}")
            cl_CFSv2.append(f"--f1dot {self.Writer.F1:.16g}")
            cl_CFSv2.append(f"--f2dot {self.Writer.F2:.16g}")
            cl_CFSv2.append(f"--f3dot {self.F3s[0]:.16g}")
            cl_CFSv2.append(f"--f3dotBand {(self.F3s[1] - self.F3s[0]):.16g}")
            cl_CFSv2.append(f"--df3dot {(self.F3s[1] - self.F3s[0]):.16g}")
            translated_binary_params = search.translate_keys_to_lal(
                default_binary_params
            )
            for key in translated_binary_params.keys():
                cl_CFSv2.append(f"--{key} {translated_binary_params[key]:.16g}")
        else:
            cl_CFSv2.append("--gridType 6")
            cl_CFSv2.append(f"--gridFile {self.gridfile}")
        if transient:
            cl_CFSv2.append(
                f"--transient-WindowType {transient_search_params['transientWindowType']}"
            )
            cl_CFSv2.append("--transient-t0Offset 0")
            cl_CFSv2.append(f"--transient-t0Band {transient_search_params['t0Band']:d}")
            cl_CFSv2.append(f"--transient-tau {int(2 * self.Tsft):d}")
            cl_CFSv2.append(
                f"--transient-tauBand {transient_search_params['tauBand']:d}"
            )
            CFSv2_out_file_transient = CFSv2_out_file.replace(".txt", "_transient.txt")
            cl_CFSv2.append(f"--outputTransientStats {CFSv2_out_file_transient}")
        cl_CFSv2.append("--DataFiles '{}'".format(self.Writer.sftfilepath))
        cl_CFSv2.append("--refTime {}".format(self.tref))
        cl_CFSv2.append("--outputFstat " + CFSv2_out_file)
        cl_CFSv2.append("--outputLoudest " + CFSv2_loudest_file)
        # to match ComputeFstat default (and hence PyFstat) defaults on older
        # CFSv2 versions, set the RngMedWindow manually:
        cl_CFSv2.append("--RngMedWindow=101")
        cl_CFSv2 = " ".join(cl_CFSv2)
        pyfstat.utils.run_commandline(cl_CFSv2)
        self.assertTrue(os.path.isfile(CFSv2_out_file))
        self.assertTrue(os.path.isfile(CFSv2_loudest_file))
        CFSv2_out = pyfstat.utils.read_txt_file_with_header(
            CFSv2_out_file, comments="%"
        )
        self.assertTrue(
            len(np.atleast_1d(CFSv2_out["2F"]))
            == len(np.atleast_1d(pyfstat_out["twoF"]))
        )

        # we first compare the Doppler parameters, which are the first six columns of the output files
        for i in range(6):
            self.assertTrue(
                np.max(
                    np.abs(
                        CFSv2_out[CFSv2_out.dtype.names[i]]
                        - pyfstat_out[pyfstat_out.dtype.names[i]]
                    )
                )
                < 1e-16
            )
        self.assertTrue(np.max(np.abs(CFSv2_out["2F"] - pyfstat_out["twoF"]) < 1))
        self.assertTrue(np.max(CFSv2_out["2F"]) == np.max(pyfstat_out["twoF"]))
        if transient:
            self.assertTrue(os.path.isfile(CFSv2_out_file_transient))
            CFSv2_out_transient = pyfstat.utils.read_txt_file_with_header(
                CFSv2_out_file_transient, comments="%"
            )
            self.assertTrue(
                len(np.atleast_1d(CFSv2_out_transient["maxTwoF"]))
                == len(np.atleast_1d(pyfstat_out["maxTwoF"]))
            )
            self.assertTrue(
                np.max(
                    np.abs(CFSv2_out_transient["FreqHz"] - pyfstat_out["F0"]) < 1e-16
                )
            )
            self.assertTrue(
                np.max(
                    np.abs(CFSv2_out_transient["maxTwoF"] - pyfstat_out["maxTwoF"]) < 1
                )
            )
            self.assertTrue(
                np.max(CFSv2_out_transient["maxTwoF"])
                - np.max(pyfstat_out["maxTwoF"] < 1e-3)
            )

    def test_gridfile_search_binary(self):
        self.test_gridfile_search(binary=True)

    def test_gridfile_transient_search(self):
        self.test_gridfile_search(transient=True)

    def test_gridfile_from_CFSv2(self):
        self.test_gridfile_search(grid="CFSv2")

    def test_gridfile_from_CFSv2_pandas(self):
        self.test_gridfile_search(grid="CFSv2", reading_method="pandas")
