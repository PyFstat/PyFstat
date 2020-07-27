""" Searches using grid-based methods """


import os
import logging
import itertools
from collections import OrderedDict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import re

import pyfstat.helper_functions as helper_functions
from pyfstat.core import (
    BaseSearchClass,
    ComputeFstat,
    SemiCoherentGlitchSearch,
    SemiCoherentSearch,
    tqdm,
    args,
    read_par,
)
import lalpulsar
import lal


class GridSearch(BaseSearchClass):
    """ Gridded search using ComputeFstat """

    tex_labels = {
        "F0": r"$f$",
        "F1": r"$\dot{f}$",
        "F2": r"$\ddot{f}$",
        "Alpha": r"$\alpha$",
        "Delta": r"$\delta$",
        "twoF": r"$\widetilde{2\mathcal{F}}$",
        "BSGL": r"$\log_{10}{\mathcal{B}_{\mathrm{SGL}}$",
    }
    tex_labels0 = {
        "F0": r"$-f_0$",
        "F1": r"$-\dot{f}_0$",
        "F2": r"$-\ddot{f}_0$",
        "Alpha": r"$-\alpha_0$",
        "Delta": r"$-\delta_0$",
    }

    @helper_functions.initializer
    def __init__(
        self,
        label,
        outdir,
        sftfilepattern,
        F0s,
        F1s,
        F2s,
        Alphas,
        Deltas,
        tref=None,
        minStartTime=None,
        maxStartTime=None,
        nsegs=1,
        BSGL=False,
        minCoverFreq=None,
        maxCoverFreq=None,
        detectors=None,
        SSBprec=None,
        RngMedWindow=None,
        injectSources=None,
        input_arrays=False,
        assumeSqrtSX=None,
        earth_ephem=None,
        sun_ephem=None,
        estimate_covering_band=None,
    ):
        """
        Parameters
        ----------
        label, outdir: str
            A label and directory to read/write data from/to
        sftfilepattern: str
            Pattern to match SFTs using wildcards (*?) and ranges [0-9];
            mutiple patterns can be given separated by colons.
        F0s, F1s, F2s, Alphas, Deltas: tuple
            Length 3 tuple describing the grid for each parameter, e.g
            [F0min, F0max, dF0], for a fixed value simply give [F0]. Unless
            input_arrays == True, then these are the values to search at.
        tref, minStartTime, maxStartTime: int
            GPS seconds of the reference time, start time and end time
        input_arrays: bool
            if true, use the F0s, F1s, etc as is

        For all other parameters, see `pyfstat.ComputeFStat` for details

        Note: if a large number of grid points are used, checks against cached
        data may be slow as the array is loaded into memory. To avoid this, run
        with the `clean` option which uses a generator instead.
        """

        self._set_init_params_dict(locals())
        if os.path.isdir(outdir) is False:
            os.mkdir(outdir)
        self.set_out_file()
        self.search_keys = ["F0", "F1", "F2", "Alpha", "Delta"]
        self.input_keys = ["minStartTime", "maxStartTime"] + self.search_keys
        self.keys = self.input_keys.copy()
        if self.BSGL:
            self.detstat = "BSGL"
            self.keys += ["logBSGL"]
        else:
            self.detstat = "twoF"
            self.keys += ["twoF"]
        for k in self.search_keys:
            setattr(self, k, np.atleast_1d(getattr(self, k + "s")))
        self.output_file_header = self.get_output_file_header()

    def _get_search_ranges(self):
        if (self.minCoverFreq is None) or (self.maxCoverFreq is None):
            return {key: getattr(self, key + "s") for key in self.search_keys}
        else:
            return None

    def inititate_search_object(self):
        logging.info("Setting up search object")
        search_ranges = self._get_search_ranges()
        if self.nsegs == 1:
            self.search = ComputeFstat(
                tref=self.tref,
                sftfilepattern=self.sftfilepattern,
                minCoverFreq=self.minCoverFreq,
                maxCoverFreq=self.maxCoverFreq,
                search_ranges=search_ranges,
                detectors=self.detectors,
                minStartTime=self.minStartTime,
                maxStartTime=self.maxStartTime,
                BSGL=self.BSGL,
                SSBprec=self.SSBprec,
                RngMedWindow=self.RngMedWindow,
                injectSources=self.injectSources,
                assumeSqrtSX=self.assumeSqrtSX,
                earth_ephem=self.earth_ephem,
                sun_ephem=self.sun_ephem,
                estimate_covering_band=self.estimate_covering_band,
            )
            self.search.get_det_stat = self.search.get_fullycoherent_twoF
        else:
            self.search = SemiCoherentSearch(
                label=self.label,
                outdir=self.outdir,
                tref=self.tref,
                nsegs=self.nsegs,
                sftfilepattern=self.sftfilepattern,
                BSGL=self.BSGL,
                minStartTime=self.minStartTime,
                maxStartTime=self.maxStartTime,
                minCoverFreq=self.minCoverFreq,
                maxCoverFreq=self.maxCoverFreq,
                search_ranges=search_ranges,
                detectors=self.detectors,
                injectSources=self.injectSources,
                estimate_covering_band=self.estimate_covering_band,
            )

            def cut_out_tstart_tend(*vals):
                return self.search.get_semicoherent_det_stat(*vals[2:])

            self.search.get_det_stat = cut_out_tstart_tend

    def get_array_from_tuple(self, x):
        if len(x) == 1:
            return np.array(x)
        elif len(x) == 3 and self.input_arrays is False:
            return np.arange(x[0], x[1], x[2])
        else:
            logging.info("Using tuple as is")
            return np.array(x)

    def get_input_data_array(self):
        logging.info("Generating input data array")
        coord_arrays = []
        for sl in self.input_keys:
            coord_arrays.append(
                self.get_array_from_tuple(np.atleast_1d(getattr(self, sl)))
            )
        self.coord_arrays = coord_arrays
        self.total_iterations = np.prod([len(ca) for ca in coord_arrays])

        if args.clean is False:
            input_data = []
            for vals in itertools.product(*coord_arrays):
                input_data.append(vals)
            self.input_data = np.array(input_data)

    def check_old_data_is_okay_to_use(self):
        if args.clean:
            return False
        if os.path.isfile(self.out_file) is False:
            logging.info(
                "No old output file '{:s}' found, continuing with grid search.".format(
                    self.out_file
                )
            )
            return False
        if self.sftfilepattern is not None:
            oldest_sft = min(
                [os.path.getmtime(f) for f in self._get_list_of_matching_sfts()]
            )
            if os.path.getmtime(self.out_file) < oldest_sft:
                logging.info(
                    "Search output data outdates sft files,"
                    + " continuing with grid search."
                )
                return False

        logging.info("Checking header of '{:s}'".format(self.out_file))
        old_params_dict_str_list = helper_functions.read_parameters_dict_lines_from_file_header(
            self.out_file
        )
        new_params_dict_str_list = [
            l.strip(" ") for l in self.pprint_init_params_dict()[1:-1]
        ]
        unmatched = np.setxor1d(old_params_dict_str_list, new_params_dict_str_list)
        if len(unmatched) > 0:
            logging.info(
                "Parameters string in file header does not match"
                + " current search setup, continuing with grid search."
            )
            return False
        else:
            logging.info(
                "Parameters string in file header matches current search setup."
            )

        logging.info("Loading old data from '{:s}'.".format(self.out_file))
        old_data = np.atleast_2d(np.genfromtxt(self.out_file, delimiter=" "))
        # need to convert any "None" entries in input_data array safely to 0s
        # to make np.allclose() work reliably
        new_data = np.nan_to_num(self.input_data.astype(np.float64))
        rtol, atol = self._get_tolerance_from_savetxt_fmt()
        column_matches = [
            np.allclose(old_data[:, n], new_data[:, n], rtol=rtol[n], atol=atol[n],)
            for n in range(len(self.coord_arrays))
        ]
        if np.all(column_matches):
            logging.info(
                "Old data found in '{:s}' with matching input parameters grid,"
                " no search performed.".format(self.out_file)
            )
            return old_data
        else:
            logging.info(
                "Old data found in '{:s}', input parameters grid differs,h"
                "  continuing with grid search.".format(self.out_file)
            )
            return False
        return False

    def run(self, return_data=False):
        self.get_input_data_array()

        if args.clean:
            iterable = itertools.product(*self.coord_arrays)
        else:
            old_data = self.check_old_data_is_okay_to_use()
            iterable = self.input_data

            if old_data is not False:
                self.data = old_data
                return

        if hasattr(self, "search") is False:
            self.inititate_search_object()

        data = []
        for vals in tqdm(iterable, total=getattr(self, "total_iterations", None)):
            detstat = self.search.get_det_stat(*vals)
            thisCand = list(vals) + [detstat]
            data.append(thisCand)

        data = np.array(data, dtype=np.float)
        if return_data:
            return data
        else:
            self.save_array_to_disk(data)
            self.data = data

    def get_savetxt_fmt(self):
        fmt = []
        if "minStartTime" in self.keys:
            fmt += ["%d"]
        if "maxStartTime" in self.keys:
            fmt += ["%d"]
        fmt += helper_functions.get_doppler_params_output_format(self.keys)
        fmt += ["%.9g"]  # for detection statistic
        return fmt

    def _get_tolerance_from_savetxt_fmt(self):
        """ decide appropriate input grid comparison tolerance from fprintf formats """
        fmt = self.get_savetxt_fmt()
        rtol = np.zeros(len(fmt))
        atol = np.zeros(len(fmt))
        for n, f in enumerate(fmt):
            if f.endswith("d"):
                rtol[n] = 0
                atol[n] = 0
            elif f.endswith("g"):
                precision = int(re.findall(r"\d+", f)[-1])
                rtol[n] = 10 ** (1 - precision)
                atol[n] = 0
            elif f.endswith("f"):
                decimals = int(re.findall(r"\d+", f)[-1])
                rtol[n] = 0
                atol[n] = 10 ** -decimals
            else:
                raise ValueError(
                    "Cannot parse fprintf format '{:s}' to obtain recommended tolerance.".format(
                        f
                    )
                )
        return rtol, atol

    def save_array_to_disk(self, data):
        logging.info("Saving data to {}".format(self.out_file))
        header = "\n".join(self.output_file_header)
        header += "\n" + " ".join(self.keys)
        outfmt = self.get_savetxt_fmt()
        Ncols = np.shape(data)[1]
        if len(outfmt) != Ncols:
            raise RuntimeError(
                "Lengths of data rows ({:d})"
                " and output format ({:d})"
                " do not match."
                " If your search class uses different"
                " keys than the base GridSearch class,"
                " override the get_savetxt_fmt"
                " method.".format(Ncols, len(outfmt))
            )
        np.savetxt(
            self.out_file,
            np.nan_to_num(data),
            delimiter=" ",
            header=header,
            fmt=outfmt,
        )

    def convert_F0_to_mismatch(self, F0, F0hat, Tseg):
        DeltaF0 = F0[1] - F0[0]
        m_spacing = (np.pi * Tseg * DeltaF0) ** 2 / 12.0
        N = len(F0)
        return np.arange(-N * m_spacing / 2.0, N * m_spacing / 2.0, m_spacing)

    def convert_F1_to_mismatch(self, F1, F1hat, Tseg):
        DeltaF1 = F1[1] - F1[0]
        m_spacing = (np.pi * Tseg ** 2 * DeltaF1) ** 2 / 720.0
        N = len(F1)
        return np.arange(-N * m_spacing / 2.0, N * m_spacing / 2.0, m_spacing)

    def add_mismatch_to_ax(self, ax, x, y, xkey, ykey, xhat, yhat, Tseg):
        axX = ax.twiny()
        axX.zorder = -10
        axY = ax.twinx()
        axY.zorder = -10

        if xkey == "F0":
            m = self.convert_F0_to_mismatch(x, xhat, Tseg)
            axX.set_xlim(m[0], m[-1])

        if ykey == "F1":
            m = self.convert_F1_to_mismatch(y, yhat, Tseg)
            axY.set_ylim(m[0], m[-1])

    def plot_1D(
        self,
        xkey,
        ax=None,
        x0=None,
        xrescale=1,
        savefig=True,
        xlabel=None,
        ylabel=None,
        agg_chunksize=None,
    ):
        if agg_chunksize:
            # FIXME: workaround for matplotlib "Exceeded cell block limit" errors
            plt.rcParams["agg.path.chunksize"] = agg_chunksize
        if ax is None:
            fig, ax = plt.subplots()
        xidx = self.keys.index(xkey)
        # x = np.unique(self.data[:, xidx]) # this doesn't work for multi-dim searches!
        x = self.data[:, xidx]
        if x0:
            x = x - x0
        x = x * xrescale
        zidx = self.keys.index(self.detstat)
        z = self.data[:, zidx]
        ax.plot(x, z)
        if xlabel:
            ax.set_xlabel(xlabel)
        elif x0:
            ax.set_xlabel(self.tex_labels[xkey] + self.tex_labels0[xkey])
        else:
            ax.set_xlabel(self.tex_labels[xkey])
        if ylabel:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(self.tex_labels[self.detstat])
        if savefig:
            fig.tight_layout()
            fname = "{}_1D_{}_{}.png".format(self.label, xkey, self.detstat)
            fig.savefig(os.path.join(self.outdir, fname))
            plt.close(fig)
        else:
            return ax

    def plot_2D(
        self,
        xkey,
        ykey,
        ax=None,
        save=True,
        vmin=None,
        vmax=None,
        add_mismatch=None,
        xN=None,
        yN=None,
        flat_keys=[],
        rel_flat_idxs=[],
        flatten_method=np.max,
        title=None,
        predicted_twoF=None,
        cm=None,
        cbarkwargs={},
        x0=None,
        y0=None,
        colorbar=False,
        xrescale=1,
        yrescale=1,
        xlabel=None,
        ylabel=None,
        zlabel=None,
    ):
        """ Plots a 2D grid of 2F values

        Parameters
        ----------
        add_mismatch: tuple (xhat, yhat, Tseg)
            If not None, add a secondary axis with the metric mismatch from the
            point xhat, yhat with duration Tseg
        flatten_method: np.max
            Function to use in flattening the flat_keys

        FIXME: this will currently fail if the search went over >2 dimensions
        """
        if ax is None:
            fig, ax = plt.subplots()
        xidx = self.input_keys.index(xkey)
        yidx = self.input_keys.index(ykey)
        flat_idxs = [self.input_keys.index(k) for k in flat_keys]

        x = np.unique(self.data[:, xidx])
        if x0:
            x = x - x0
        y = np.unique(self.data[:, yidx])
        if y0:
            y = y - y0
        flat_vals = [np.unique(self.data[:, j]) for j in flat_idxs]

        zidx = self.keys.index(self.detstat)
        z = self.data[:, zidx]

        Y, X = np.meshgrid(y, x)
        shape = [len(x), len(y)] + [len(v) for v in flat_vals]
        Z = z.reshape(shape)

        if len(rel_flat_idxs) > 0:
            Z = flatten_method(Z, axis=tuple(rel_flat_idxs))

        if predicted_twoF:
            Z = (predicted_twoF - Z) / (predicted_twoF + 4)
            if cm is None:
                cm = plt.cm.viridis_r
        else:
            if cm is None:
                cm = plt.cm.viridis

        pax = ax.pcolormesh(
            X * xrescale, Y * yrescale, Z, cmap=cm, vmin=vmin, vmax=vmax
        )
        if colorbar:
            cb = plt.colorbar(pax, ax=ax, **cbarkwargs)
            if zlabel:
                cb.set_label(zlabel)
            else:
                cb.set_label(self.tex_labels[self.detstat])

        if add_mismatch:
            self.add_mismatch_to_ax(ax, x, y, xkey, ykey, *add_mismatch)

        if x[-1] > x[0]:
            ax.set_xlim(x[0] * xrescale, x[-1] * xrescale)
        if y[-1] > y[0]:
            ax.set_ylim(y[0] * yrescale, y[-1] * yrescale)

        if xlabel:
            ax.set_xlabel(xlabel)
        elif x0:
            ax.set_xlabel(self.tex_labels[xkey] + self.tex_labels0[xkey])
        else:
            ax.set_xlabel(self.tex_labels[xkey])

        if ylabel:
            ax.set_ylabel(ylabel)
        elif y0:
            ax.set_ylabel(self.tex_labels[ykey] + self.tex_labels0[ykey])
        else:
            ax.set_ylabel(self.tex_labels[ykey])

        if title:
            ax.set_title(title)

        if xN:
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(xN))
        if yN:
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(yN))

        if save:
            fig.tight_layout()
            fname = "{}_2D_{}_{}_{}.png".format(self.label, xkey, ykey, self.detstat)
            fig.savefig(os.path.join(self.outdir, fname))
        else:
            return ax

    def get_max_twoF(self):
        """ Get the maximum twoF over the grid

        Returns
        -------
        d: dict
            Dictionary containing, 'minStartTime', 'maxStartTime', 'F0', 'F1',
            'F2', 'Alpha', 'Delta' and 'twoF' of maximum

        """
        idx = np.argmax(self.data[:, self.keys.index("twoF")])
        d = OrderedDict([(key, self.data[idx, k]) for k, key in enumerate(self.keys)])
        return d

    def print_max_twoF(self):
        d = self.get_max_twoF()
        print("Grid point with max(twoF) for {}:".format(self.label))
        for k, v in d.items():
            print("  {}={}".format(k, v))

    def set_out_file(self, extra_label=None):
        if self.detectors:
            dets = self.detectors.replace(",", "")
        else:
            dets = "NA"
        if extra_label:
            self.out_file = os.path.join(
                self.outdir,
                "{}_{}_{}_{}.txt".format(
                    self.label, dets, type(self).__name__, extra_label
                ),
            )
        else:
            self.out_file = os.path.join(
                self.outdir,
                "{}_{}_{}.txt".format(self.label, dets, type(self).__name__),
            )


class TransientGridSearch(GridSearch):
    """ Gridded transient-continous search using ComputeFstat """

    @helper_functions.initializer
    def __init__(
        self,
        label,
        outdir,
        sftfilepattern,
        F0s,
        F1s,
        F2s,
        Alphas,
        Deltas,
        tref=None,
        minStartTime=None,
        maxStartTime=None,
        BSGL=False,
        minCoverFreq=None,
        maxCoverFreq=None,
        detectors=None,
        SSBprec=None,
        RngMedWindow=None,
        injectSources=None,
        input_arrays=False,
        assumeSqrtSX=None,
        transientWindowType=None,
        t0Band=None,
        tauBand=None,
        tauMin=None,
        dt0=None,
        dtau=None,
        outputTransientFstatMap=False,
        outputAtoms=False,
        tCWFstatMapVersion="lal",
        cudaDeviceName=None,
        earth_ephem=None,
        sun_ephem=None,
        estimate_covering_band=None,
    ):
        """
        Parameters
        ----------
        label, outdir: str
            A label and directory to read/write data from/to
        sftfilepattern: str
            Pattern to match SFTs using wildcards (*?) and ranges [0-9];
            mutiple patterns can be given separated by colons.
        F0s, F1s, F2s, Alphas, Deltas: tuple
            Length 3 tuple describing the grid for each parameter, e.g
            [F0min, F0max, dF0], for a fixed value simply give [F0]. Unless
            input_arrays == True, then these are the values to search at.
        tref, minStartTime, maxStartTime: int
            GPS seconds of the reference time, start time and end time
        input_arrays: bool
            if true, use the F0s, F1s, etc as is
        transientWindowType: str
            If 'rect' or 'exp', compute atoms so that a transient (t0,tau) map
            can later be computed.  ('none' instead of None explicitly calls
            the transient-window function, but with the full range, for
            debugging). Currently only supported for nsegs=1.
        t0Band, tauBand: int
            if >0, search t0 in (minStartTime,minStartTime+t0Band)
                   and tau in (tauMin,2*Tsft+tauBand).
            if =0, only compute CW Fstat with t0=minStartTime,
                   tau=maxStartTime-minStartTime.
        tauMin: int
            defaults to 2*Tsft
        dt0, dtau: int
            grid resolutions in transient start-time and duration,
            both default to Tsft
        outputTransientFstatMap: bool
            if true, write output files for (t0,tau) Fstat maps
            (one file for each doppler grid point!)
        tCWFstatMapVersion: str
            Choose between standard 'lal' implementation,
            'pycuda' for gpu, and some others for devel/debug.
        cudaDeviceName: str
            GPU name to be matched against drv.Device output.

        For all other parameters, see `pyfstat.ComputeFStat` for details
        """

        self._set_init_params_dict(locals())
        self.nsegs = 1
        if os.path.isdir(outdir) is False:
            os.mkdir(outdir)
        self.set_out_file()
        self.search_keys = ["F0", "F1", "F2", "Alpha", "Delta"]
        self.input_keys = ["minStartTime", "maxStartTime"] + self.search_keys
        self.keys = self.input_keys.copy()
        if self.BSGL:
            self.detstat = "BSGL"
            self.keys += ["logBSGL"]
        else:
            self.detstat = "twoF"
            self.keys += ["twoF"]
        # for consistency below, t0/tau must come after detstat
        self.keys += ["t0", "tau"]
        for k in self.search_keys:
            setattr(self, k, np.atleast_1d(getattr(self, k + "s")))
        self.output_file_header = self.get_output_file_header()

    def inititate_search_object(self):
        logging.info("Setting up search object")
        search_ranges = self._get_search_ranges()
        self.search = ComputeFstat(
            tref=self.tref,
            sftfilepattern=self.sftfilepattern,
            minCoverFreq=self.minCoverFreq,
            maxCoverFreq=self.maxCoverFreq,
            search_ranges=search_ranges,
            detectors=self.detectors,
            transientWindowType=self.transientWindowType,
            t0Band=self.t0Band,
            tauBand=self.tauBand,
            tauMin=self.tauMin,
            dt0=self.dt0,
            dtau=self.dtau,
            minStartTime=self.minStartTime,
            maxStartTime=self.maxStartTime,
            BSGL=self.BSGL,
            SSBprec=self.SSBprec,
            RngMedWindow=self.RngMedWindow,
            injectSources=self.injectSources,
            assumeSqrtSX=self.assumeSqrtSX,
            tCWFstatMapVersion=self.tCWFstatMapVersion,
            cudaDeviceName=self.cudaDeviceName,
            computeAtoms=self.outputAtoms,
            earth_ephem=self.earth_ephem,
            sun_ephem=self.sun_ephem,
            estimate_covering_band=self.estimate_covering_band,
        )
        self.search.get_det_stat = self.search.get_fullycoherent_twoF

    def run(self, return_data=False):
        self.get_input_data_array()
        old_data = self.check_old_data_is_okay_to_use()
        if old_data is not False:
            self.data = old_data
            return

        if hasattr(self, "search") is False:
            self.inititate_search_object()

        data = []
        if self.outputTransientFstatMap:
            self.tCWfilebase = os.path.splitext(self.out_file)[0] + "_tCW_"
            logging.info(
                "Will save per-Doppler Fstatmap"
                " results to {}*.dat".format(self.tCWfilebase)
            )
        self.timingFstatMap = 0.0
        for vals in tqdm(self.input_data):
            detstat = self.search.get_det_stat(*vals)
            windowRange = getattr(self.search, "windowRange", None)
            FstatMap = getattr(self.search, "FstatMap", None)
            self.timingFstatMap += getattr(self.search, "timingFstatMap", 0.0)
            thisCand = list(vals) + [detstat]
            if getattr(self, "transientWindowType", None):
                if self.tCWFstatMapVersion == "lal":
                    F_mn = FstatMap.F_mn.data
                else:
                    F_mn = FstatMap.F_mn
                if self.outputTransientFstatMap:
                    # per-Doppler filename convention:
                    # freq alpha delta f1dot f2dot
                    tCWfile = (
                        self.tCWfilebase
                        + "{:.16f}_{:.16f}_{:.16f}_{:.16g}_{:.16g}.dat".format(
                            vals[self.keys.index("F0")],
                            vals[self.keys.index("Alpha")],
                            vals[self.keys.index("Delta")],
                            vals[self.keys.index("F1")],
                            vals[self.keys.index("F2")],
                        )
                    )
                    if self.tCWFstatMapVersion == "lal":
                        fo = lal.FileOpen(tCWfile, "w")
                        for hline in self.output_file_header:
                            lal.FilePuts("# {:s}\n".format(hline), fo)
                        lal.FilePuts("# t0[s]      tau[s]      2F\n", fo)
                        lalpulsar.write_transientFstatMap_to_fp(
                            fo, FstatMap, windowRange, None
                        )
                        # instead of lal.FileClose(),
                        # which is not SWIG-exported:
                        del fo
                    else:
                        self.write_F_mn(tCWfile, F_mn, windowRange)
                maxidx = np.unravel_index(F_mn.argmax(), F_mn.shape)
                thisCand += [
                    windowRange.t0 + maxidx[0] * windowRange.dt0,
                    windowRange.tau + maxidx[1] * windowRange.dtau,
                ]
            data.append(thisCand)
            if self.outputAtoms:
                self.search.write_atoms_to_file(os.path.splitext(self.out_file)[0])

        logging.info(
            "Total time spent computing transient F-stat maps: {:.2f}s".format(
                self.timingFstatMap
            )
        )

        data = np.array(data, dtype=np.float)
        if return_data:
            return data
        else:
            self.save_array_to_disk(data)
            self.data = data

    def get_savetxt_fmt(self):
        fmt = []
        if "minStartTime" in self.keys:
            fmt += ["%d"]
        if "maxStartTime" in self.keys:
            fmt += ["%d"]
        fmt += helper_functions.get_doppler_params_output_format(self.keys)
        fmt += ["%.9g"]  # for detection statistic
        fmt += ["%d", "%d"]  # for t0, tau
        return fmt

    def write_F_mn(self, tCWfile, F_mn, windowRange):
        with open(tCWfile, "w") as tfp:
            for hline in self.output_file_header:
                tfp.write("# {:s}\n".format(hline))
            tfp.write("# t0 [s]     tau [s]     2F\n")
            for m, F_m in enumerate(F_mn):
                this_t0 = windowRange.t0 + m * windowRange.dt0
                for n, this_F in enumerate(F_m):
                    this_tau = windowRange.tau + n * windowRange.dtau
                    tfp.write(
                        "  %10d %10d %- 11.8g\n" % (this_t0, this_tau, 2.0 * this_F)
                    )

    def __del__(self):
        if hasattr(self, "search"):
            self.search.__del__()


class SliceGridSearch(GridSearch):
    """ Slice gridded search using ComputeFstat """

    @helper_functions.initializer
    def __init__(
        self,
        label,
        outdir,
        sftfilepattern,
        F0s,
        F1s,
        F2s,
        Alphas,
        Deltas,
        tref=None,
        minStartTime=None,
        maxStartTime=None,
        nsegs=1,
        minCoverFreq=None,
        maxCoverFreq=None,
        detectors=None,
        SSBprec=None,
        RngMedWindow=None,
        injectSources=None,
        input_arrays=False,
        assumeSqrtSX=None,
        Lambda0=None,
        earth_ephem=None,
        sun_ephem=None,
        estimate_covering_band=None,
    ):
        """
        Parameters
        ----------
        label, outdir: str
            A label and directory to read/write data from/to
        sftfilepattern: str
            Pattern to match SFTs using wildcards (*?) and ranges [0-9];
            mutiple patterns can be given separated by colons.
        F0s, F1s, F2s, Alphas, Deltas: tuple
            Length 3 tuple describing the grid for each parameter, e.g
            [F0min, F0max, dF0], for a fixed value simply give [F0]. Unless
            input_arrays == True, then these are the values to search at.
        tref, minStartTime, maxStartTime: int
            GPS seconds of the reference time, start time and end time
        input_arrays: bool
            if true, use the F0s, F1s, etc as is

        For all other parameters, see `pyfstat.ComputeFStat` for details
        """

        self._set_init_params_dict(locals())
        if os.path.isdir(outdir) is False:
            os.mkdir(outdir)
        self.set_out_file()
        self.search_keys = ["F0", "F1", "Alpha", "Delta"]
        self.input_keys = ["minStartTime", "maxStartTime"] + self.search_keys
        self.keys = self.input_keys.copy()
        self.ndim = 0
        self.thetas = [F0s, F1s, Alphas, Deltas]
        self.ndim = 4

        if self.Lambda0 is None:
            raise ValueError("Lambda0 undefined")
        if len(self.Lambda0) != len(self.search_keys):
            raise ValueError(
                "Lambda0 must be of length {}".format(len(self.search_keys))
            )
        self.Lambda0 = np.array(Lambda0)

    def run(self, factor=2, max_n_ticks=4, whspace=0.07, save=True, **kwargs):
        lbdim = 0.5 * factor  # size of left/bottom margin
        trdim = 0.4 * factor  # size of top/right margin
        plotdim = factor * self.ndim + factor * (self.ndim - 1.0) * whspace
        dim = lbdim + plotdim + trdim

        fig, axes = plt.subplots(self.ndim, self.ndim, figsize=(dim, dim))

        # Format the figure.
        lb = lbdim / dim
        tr = (lbdim + plotdim) / dim
        fig.subplots_adjust(
            left=lb, bottom=lb, right=tr, top=tr, wspace=whspace, hspace=whspace
        )

        search = GridSearch(
            self.label,
            self.outdir,
            self.sftfilepattern,
            F0s=[self.Lambda0[0]],
            F1s=[self.Lambda0[1]],
            F2s=[self.F2s[0]],
            Alphas=[self.Lambda0[2]],
            Deltas=[self.Lambda0[3]],
            tref=self.tref,
            minStartTime=self.minStartTime,
            maxStartTime=self.maxStartTime,
            earth_ephem=self.earth_ephem,
            sun_ephem=self.sun_ephem,
            estimate_covering_band=self.estimate_covering_band,
        )

        for i, ikey in enumerate(self.search_keys):
            setattr(search, ikey + "s", self.thetas[i])
            search.label = "{}_{}".format(self.label, ikey)
            search.set_out_file()
            search.run()
            axes[i, i] = search.plot_1D(
                ikey, ax=axes[i, i], savefig=False, x0=self.Lambda0[i]
            )
            setattr(search, ikey + "s", [self.Lambda0[i]])
            axes[i, i].yaxis.tick_right()
            axes[i, i].yaxis.set_label_position("right")
            axes[i, i].set_xlabel("")

            for j, jkey in enumerate(self.search_keys):
                ax = axes[i, j]

                if j > i:
                    ax.set_frame_on(False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                ax.get_shared_x_axes().join(axes[self.ndim - 1, j], ax)
                if i < self.ndim - 1:
                    ax.set_xticklabels([])
                if j < i:
                    ax.get_shared_y_axes().join(axes[i, i - 1], ax)
                    if j > 0:
                        ax.set_yticklabels([])
                if j == i:
                    continue

                ax.xaxis.set_major_locator(
                    matplotlib.ticker.MaxNLocator(max_n_ticks, prune="upper")
                )
                ax.yaxis.set_major_locator(
                    matplotlib.ticker.MaxNLocator(max_n_ticks, prune="upper")
                )

                setattr(search, ikey + "s", self.thetas[i])
                setattr(search, jkey + "s", self.thetas[j])
                search.label = "{}_{}".format(self.label, ikey + jkey)
                search.set_out_file()
                search.run()
                ax = search.plot_2D(
                    jkey,
                    ikey,
                    ax=ax,
                    save=False,
                    y0=self.Lambda0[i],
                    x0=self.Lambda0[j],
                    **kwargs
                )
                setattr(search, ikey + "s", [self.Lambda0[i]])
                setattr(search, jkey + "s", [self.Lambda0[j]])

                ax.grid(lw=0.2, ls="--", zorder=10)
                ax.set_xlabel("")
                ax.set_ylabel("")

        for i, ikey in enumerate(self.search_keys):
            axes[-1, i].set_xlabel(self.tex_labels[ikey] + self.tex_labels0[ikey])
            if i > 0:
                axes[i, 0].set_ylabel(self.tex_labels[ikey] + self.tex_labels0[ikey])
            axes[i, i].set_ylabel(self.tex_labels["twoF"])

        if save:
            fig.savefig(os.path.join(self.outdir, self.label + "_slice_projection.png"))
        else:
            return fig, axes


class GridUniformPriorSearch:
    @helper_functions.initializer
    def __init__(
        self,
        theta_prior,
        NF0,
        NF1,
        label,
        outdir,
        sftfilepattern,
        tref,
        minStartTime,
        maxStartTime,
        minCoverFreq=None,
        maxCoverFreq=None,
        BSGL=False,
        detectors=None,
        nsegs=1,
        SSBprec=None,
        RngMedWindow=None,
        injectSources=None,
        earth_ephem=None,
        sun_ephem=None,
        estimate_covering_band=None,
    ):
        dF0 = (theta_prior["F0"]["upper"] - theta_prior["F0"]["lower"]) / NF0
        dF1 = (theta_prior["F1"]["upper"] - theta_prior["F1"]["lower"]) / NF1
        F0s = [theta_prior["F0"]["lower"], theta_prior["F0"]["upper"], dF0]
        F1s = [theta_prior["F1"]["lower"], theta_prior["F1"]["upper"], dF1]
        self.search = GridSearch(
            label,
            outdir,
            sftfilepattern,
            F0s=F0s,
            F1s=F1s,
            tref=tref,
            Alphas=[theta_prior["Alpha"]],
            Deltas=[theta_prior["Delta"]],
            minStartTime=minStartTime,
            maxStartTime=maxStartTime,
            BSGL=BSGL,
            detectors=detectors,
            minCoverFreq=minCoverFreq,
            injectSources=injectSources,
            maxCoverFreq=maxCoverFreq,
            nsegs=nsegs,
            SSBprec=SSBprec,
            RngMedWindow=RngMedWindow,
            earth_ephem=earth_ephem,
            sun_ephem=sun_ephem,
            estimate_covering_band=self.estimate_covering_band,
        )

    def run(self):
        self.search.run()

    def get_2D_plot(self, **kwargs):
        return self.search.plot_2D("F0", "F1", **kwargs)


class GridGlitchSearch(GridSearch):
    """ Grid search using the SemiCoherentGlitchSearch """

    @helper_functions.initializer
    def __init__(
        self,
        label,
        outdir="data",
        sftfilepattern=None,
        F0s=[0],
        F1s=[0],
        F2s=[0],
        delta_F0s=[0],
        delta_F1s=[0],
        tglitchs=None,
        Alphas=[0],
        Deltas=[0],
        tref=None,
        minStartTime=None,
        maxStartTime=None,
        minCoverFreq=None,
        maxCoverFreq=None,
        detectors=None,
        earth_ephem=None,
        sun_ephem=None,
        estimate_covering_band=None,
    ):
        """
        Run a single-glitch grid search

        Parameters
        ----------
        label, outdir: str
            A label and directory to read/write data from/to
        sftfilepattern: str
            Pattern to match SFTs using wildcards (*?) and ranges [0-9];
            mutiple patterns can be given separated by colons.
        F0s, F1s, F2s, delta_F0s, delta_F1s, tglitchs, Alphas, Deltas: tuple
            Length 3 tuple describing the grid for each parameter, e.g
            [F0min, F0max, dF0], for a fixed value simply give [F0]. Note that
            tglitchs is referenced to zero at minStartTime.
        tref, minStartTime, maxStartTime: int
            GPS seconds of the reference time, start time and end time

        For all other parameters, see pyfstat.ComputeFStat.
        """

        self._set_init_params_dict(locals())
        self.BSGL = False
        self.input_arrays = False
        if tglitchs is None:
            raise ValueError("You must specify `tglitchs`")

        self.search_keys = [
            "F0",
            "F1",
            "F2",
            "Alpha",
            "Delta",
            "delta_F0",
            "delta_F1",
            "tglitch",
        ]
        self.input_keys = self.search_keys
        self.keys = self.input_keys.copy()
        self.keys += ["twoF"]
        for k in self.search_keys:
            setattr(self, k, np.atleast_1d(getattr(self, k + "s")))
        search_ranges = self._get_search_ranges()

        self.search = SemiCoherentGlitchSearch(
            label=label,
            outdir=outdir,
            sftfilepattern=self.sftfilepattern,
            tref=tref,
            minStartTime=minStartTime,
            maxStartTime=maxStartTime,
            minCoverFreq=minCoverFreq,
            maxCoverFreq=maxCoverFreq,
            search_ranges=search_ranges,
            BSGL=self.BSGL,
            earth_ephem=earth_ephem,
            sun_ephem=sun_ephem,
            estimate_covering_band=self.estimate_covering_band,
        )
        self.search.get_det_stat = self.search.get_semicoherent_nglitch_twoF

        if os.path.isdir(outdir) is False:
            os.mkdir(outdir)
        self.set_out_file()
        self.output_file_header = self.get_output_file_header()

    def get_savetxt_fmt(self):
        fmt = []
        if "minStartTime" in self.keys:
            fmt += ["%d"]
        if "maxStartTime" in self.keys:
            fmt += ["%d"]
        fmt += helper_functions.get_doppler_params_output_format(self.keys)
        fmt += ["%.16g", "%.16g", "%d"]  # for delta_F0, delta_F1, tglitch
        fmt += ["%.9g"]  # for detection statistic
        return fmt


class SlidingWindow(GridSearch):
    @helper_functions.initializer
    def __init__(
        self,
        label,
        outdir,
        sftfilepattern,
        F0,
        F1,
        F2,
        Alpha,
        Delta,
        tref,
        minStartTime=None,
        maxStartTime=None,
        window_size=10 * 86400,
        window_delta=86400,
        BSGL=False,
        minCoverFreq=None,
        maxCoverFreq=None,
        detectors=None,
        SSBprec=None,
        RngMedWindow=None,
        injectSources=None,
        earth_ephem=None,
        sun_ephem=None,
        estimate_covering_band=None,
    ):
        """
        Parameters
        ----------
        label, outdir: str
            A label and directory to read/write data from/to
        sftfilepattern: str
            Pattern to match SFTs using wildcards (*?) and ranges [0-9];
            mutiple patterns can be given separated by colons.
        F0, F1, F2, Alpha, Delta: float
            Fixed values to compute output over
        tref, minStartTime, maxStartTime: int
            GPS seconds of the reference time, start time and end time

        For all other parameters, see `pyfstat.ComputeFStat` for details
        """

        self._set_init_params_dict(locals())
        if os.path.isdir(outdir) is False:
            os.mkdir(outdir)
        self.set_out_file()
        self.nsegs = 1

        self.tstarts = [self.minStartTime]
        while self.tstarts[-1] + self.window_size < self.maxStartTime:
            self.tstarts.append(self.tstarts[-1] + self.window_delta)
        self.tmids = np.array(self.tstarts) + 0.5 * self.window_size

    def inititate_search_object(self):
        logging.info("Setting up search object")
        search_ranges = self._get_search_ranges()
        self.search = ComputeFstat(
            tref=self.tref,
            sftfilepattern=self.sftfilepattern,
            minCoverFreq=self.minCoverFreq,
            maxCoverFreq=self.maxCoverFreq,
            search_ranges=search_ranges,
            detectors=self.detectors,
            transient=True,
            minStartTime=self.minStartTime,
            maxStartTime=self.maxStartTime,
            BSGL=self.BSGL,
            SSBprec=self.SSBprec,
            RngMedWindow=self.RngMedWindow,
            injectSources=self.injectSources,
            earth_ephem=self.earth_ephem,
            sun_ephem=self.sun_ephem,
            estimate_covering_band=self.estimate_covering_band,
        )

    def check_old_data_is_okay_to_use(self, out_file):
        if os.path.isfile(out_file):
            tmids, vals, errvals = np.loadtxt(out_file).T
            if len(tmids) == len(self.tmids) and (tmids[0] == self.tmids[0]):
                self.vals = vals
                self.errvals = errvals
                return True
        return False

    def run(self, key="h0", errkey="dh0"):
        self.key = key
        self.errkey = errkey
        out_file = os.path.join(
            self.outdir, "{}_{}-sliding-window.txt".format(self.label, key)
        )

        if self.check_old_data_is_okay_to_use(out_file) is False:
            self.inititate_search_object()
            vals = []
            errvals = []
            for ts in self.tstarts:
                loudest = self.search.get_full_CFSv2_output(
                    ts,
                    ts + self.window_size,
                    self.F0,
                    self.F1,
                    self.F2,
                    self.Alpha,
                    self.Delta,
                    self.tref,
                )
                vals.append(loudest[key])
                errvals.append(loudest[errkey])

            np.savetxt(out_file, np.array([self.tmids, vals, errvals]).T)
            self.vals = np.array(vals)
            self.errvals = np.array(errvals)

    def plot_sliding_window(self, factor=1, fig=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        days = (self.tmids - self.minStartTime) / 86400
        ax.errorbar(days, self.vals * factor, yerr=self.errvals * factor)
        ax.set_ylabel(self.key)
        ax.set_xlabel(
            r"Mid-point (days after $t_\mathrm{{start}}$={})".format(self.minStartTime)
        )
        ax.set_title(
            "Sliding window of {} days in increments of {} days".format(
                self.window_size / 86400, self.window_delta / 86400
            )
        )

        if fig:
            fig.savefig(
                os.path.join(
                    self.outdir, "{}_{}-sliding-window.png".format(self.label, self.key)
                )
            )
        else:
            return ax


class FrequencySlidingWindow(GridSearch):
    """ A sliding-window search over the Frequency """

    @helper_functions.initializer
    def __init__(
        self,
        label,
        outdir,
        sftfilepattern,
        F0s,
        F1,
        F2,
        Alpha,
        Delta,
        tref,
        minStartTime=None,
        maxStartTime=None,
        window_size=10 * 86400,
        window_delta=86400,
        BSGL=False,
        minCoverFreq=None,
        maxCoverFreq=None,
        detectors=None,
        SSBprec=None,
        RngMedWindow=None,
        injectSources=None,
        earth_ephem=None,
        sun_ephem=None,
        estimate_covering_band=None,
    ):
        """
        Parameters
        ----------
        label, outdir: str
            A label and directory to read/write data from/to
        sftfilepattern: str
            Pattern to match SFTs using wildcards (*?) and ranges [0-9];
            mutiple patterns can be given separated by colons.
        F0s: array
            Frequency range
        F1, F2, Alpha, Delta: float
            Fixed values to compute twoF(F) over
        tref, minStartTime, maxStartTime: int
            GPS seconds of the reference time, start time and end time

        For all other parameters, see `pyfstat.ComputeFStat` for details
        """

        self._set_init_params_dict(locals())
        self.transientWindowType = "rect"
        self.nsegs = 1
        self.t0Band = None
        self.tauBand = None
        self.tauMin = None

        if os.path.isdir(outdir) is False:
            os.mkdir(outdir)
        self.set_out_file()
        self.F1s = [F1]
        self.F2s = [F2]
        self.Alphas = [Alpha]
        self.Deltas = [Delta]
        self.input_arrays = False
        self.keys = ["minStartTime", "maxStartTime", "F0", "F1", "F2", "Alpha", "Delta"]
        # self.search_keys = [x + "s" for x in self.keys[2:]] # this was different from other classes
        self.search_keys = self.keys[2:]
        self.output_file_header = self.get_output_file_header()

    def inititate_search_object(self):
        logging.info("Setting up search object")
        search_ranges = self._get_search_ranges()
        self.search = ComputeFstat(
            tref=self.tref,
            sftfilepattern=self.sftfilepattern,
            minCoverFreq=self.minCoverFreq,
            maxCoverFreq=self.maxCoverFreq,
            search_ranges=search_ranges,
            detectors=self.detectors,
            transientWindowType=self.transientWindowType,
            minStartTime=self.minStartTime,
            maxStartTime=self.maxStartTime,
            BSGL=self.BSGL,
            SSBprec=self.SSBprec,
            RngMedWindow=self.RngMedWindow,
            injectSources=self.injectSources,
            earth_ephem=self.earth_ephem,
            sun_ephem=self.sun_ephem,
            estimate_covering_band=self.estimate_covering_band,
        )
        self.search.get_det_stat = self.search.get_fullycoherent_twoF

    def get_input_data_array(self):
        coord_arrays = []
        tstarts = [self.minStartTime]
        while tstarts[-1] + self.window_size < self.maxStartTime:
            tstarts.append(tstarts[-1] + self.window_delta)
        coord_arrays = [tstarts]
        for tup in (self.F0s, self.F1s, self.F2s, self.Alphas, self.Deltas):
            coord_arrays.append(self.get_array_from_tuple(tup))

        input_data = []
        for vals in itertools.product(*coord_arrays):
            input_data.append(vals)

        input_data = np.array(input_data)
        input_data = np.insert(
            input_data, 1, input_data[:, 0] + self.window_size, axis=1
        )

        self.coord_arrays = coord_arrays
        self.input_data = np.array(input_data)

    def plot_sliding_window(
        self,
        F0=None,
        ax=None,
        savefig=True,
        colorbar=True,
        timestamps=False,
        F0rescale=1,
        **kwargs
    ):
        data = self.data
        if ax is None:
            ax = plt.subplot()
        tstarts = np.unique(data[:, 0])
        tends = np.unique(data[:, 1])
        frequencies = np.unique(data[:, 2])
        twoF = data[:, -1]
        tmids = (tstarts + tends) / 2.0
        dts = (tmids - self.minStartTime) / 86400.0
        if F0:
            frequencies = frequencies - F0
            ax.set_ylabel(r"Frequency - $f_0$ [Hz] \n $f_0={:0.2f}$".format(F0))
        else:
            ax.set_ylabel(r"Frequency [Hz]")
        twoF = twoF.reshape((len(tmids), len(frequencies)))
        Y, X = np.meshgrid(frequencies, dts)
        pax = ax.pcolormesh(X, Y * F0rescale, twoF, **kwargs)
        if colorbar:
            cb = plt.colorbar(pax, ax=ax)
            cb.set_label(r"$2\mathcal{F}$")
        ax.set_xlabel(
            r"Mid-point (days after $t_\mathrm{{start}}$={})".format(self.minStartTime)
        )
        ax.set_title(
            "Sliding window length = {} days in increments of {} days".format(
                self.window_size / 86400, self.window_delta / 86400
            )
        )
        if timestamps:
            axT = ax.twiny()
            axT.set_xlim(tmids[0] * 1e-9, tmids[-1] * 1e-9)
            axT.set_xlabel(r"Mid-point timestamp [GPS $10^{9}$ s]")
            ax.set_title(ax.get_title(), y=1.18)
        if savefig:
            plt.tight_layout()
            plt.savefig(os.path.join(self.outdir, self.label + "_sliding_window.png"))
        else:
            return ax


class EarthTest(GridSearch):
    """ """

    tex_labels = {
        "deltaRadius": r"$\Delta R$ [m]",
        "phaseOffset": r"phase-offset [rad]",
        "deltaPspin": r"$\Delta P_\mathrm{spin}$ [s]",
    }

    @helper_functions.initializer
    def __init__(
        self,
        label,
        outdir,
        sftfilepattern,
        deltaRadius,
        phaseOffset,
        deltaPspin,
        F0,
        F1,
        F2,
        Alpha,
        Delta,
        tref=None,
        minStartTime=None,
        maxStartTime=None,
        BSGL=False,
        minCoverFreq=None,
        maxCoverFreq=None,
        detectors=None,
        injectSources=None,
        assumeSqrtSX=None,
        earth_ephem=None,
        sun_ephem=None,
    ):
        """
        Parameters
        ----------
        label, outdir: str
            A label and directory to read/write data from/to
        sftfilepattern: str
            Pattern to match SFTs using wildcards (*?) and ranges [0-9];
            mutiple patterns can be given separated by colons.
        F0, F1, F2, Alpha, Delta: float
        tref, minStartTime, maxStartTime: int
            GPS seconds of the reference time, start time and end time

        For all other parameters, see `pyfstat.ComputeFStat` for details
        """
        self._set_init_params_dict(locals())
        self.transientWindowType = None
        self.t0Band = None
        self.tauBand = None
        self.tauMin = None

        if os.path.isdir(outdir) is False:
            os.mkdir(outdir)
        self.nsegs = 1
        self.F0s = [F0]
        self.F1s = [F1]
        self.F2s = [F2]
        self.Alphas = [Alpha]
        self.Deltas = [Delta]
        self.duration = maxStartTime - minStartTime
        self.deltaRadius = np.atleast_1d(deltaRadius)
        self.phaseOffset = np.atleast_1d(phaseOffset)
        self.phaseOffset = (
            self.phaseOffset + 1e-12
        )  # Hack to stop cached data being used
        self.deltaPspin = np.atleast_1d(deltaPspin)
        self.set_out_file()
        self.SSBprec = lalpulsar.SSBPREC_RELATIVISTIC
        self.keys = ["deltaRadius", "phaseOffset", "deltaPspin"]

        self.prior_widths = [
            np.max(self.deltaRadius) - np.min(self.deltaRadius),
            np.max(self.phaseOffset) - np.min(self.phaseOffset),
            np.max(self.deltaPspin) - np.min(self.deltaPspin),
        ]

        if hasattr(self, "search") is False:
            self.inititate_search_object()

    def get_input_data_array(self):
        logging.info("Generating input data array")
        coord_arrays = [self.deltaRadius, self.phaseOffset, self.deltaPspin]
        input_data = []
        for vals in itertools.product(*coord_arrays):
            input_data.append(vals)
        self.input_data = np.array(input_data)
        self.coord_arrays = coord_arrays

    def run_special(self):
        vals = [
            self.minStartTime,
            self.maxStartTime,
            self.F0,
            self.F1,
            self.F2,
            self.Alpha,
            self.Delta,
        ]
        self.special_data = {"zero": [0, 0, 0]}
        for key, (dR, dphi, dP) in self.special_data.items():
            rescaleRadius = 1 + dR / lal.REARTH_SI
            rescalePeriod = 1 + dP / lal.DAYSID_SI
            lalpulsar.BarycenterModifyEarthRotation(
                rescaleRadius, dphi, rescalePeriod, self.tref
            )
            FS = self.search.get_det_stat(*vals)
            self.special_data[key] = list([dR, dphi, dP]) + [FS]

    def run(self):
        self.run_special()
        self.get_input_data_array()
        old_data = self.check_old_data_is_okay_to_use()
        if old_data is not False:
            self.data = old_data
            return

        data = []
        vals = [
            self.minStartTime,
            self.maxStartTime,
            self.F0,
            self.F1,
            self.F2,
            self.Alpha,
            self.Delta,
        ]
        for (dR, dphi, dP) in tqdm(self.input_data):
            rescaleRadius = 1 + dR / lal.REARTH_SI
            rescalePeriod = 1 + dP / lal.DAYSID_SI
            lalpulsar.BarycenterModifyEarthRotation(
                rescaleRadius, dphi, rescalePeriod, self.tref
            )
            FS = self.search.get_det_stat(*vals)
            data.append(list([dR, dphi, dP]) + [FS])

        data = np.array(data, dtype=np.float)
        logging.info("Saving data to {}".format(self.out_file))
        np.savetxt(self.out_file, data, delimiter=" ")
        self.data = data

    def marginalised_bayes_factor(self, prior_widths=None):
        if prior_widths is None:
            prior_widths = self.prior_widths

        ndims = self.data.shape[1] - 1
        params = np.array([np.unique(self.data[:, j]) for j in range(ndims)])
        twoF = self.data[:, -1].reshape(tuple([len(p) for p in params]))
        F = twoF / 2.0
        for i, x in enumerate(params[::-1]):
            if len(x) > 1:
                dx = x[1] - x[0]
                F = logsumexp(F, axis=-1) + np.log(dx) - np.log(prior_widths[-1 - i])
            else:
                F = np.squeeze(F, axis=-1)
        marginalised_F = np.atleast_1d(F)[0]
        F_at_zero = self.special_data["zero"][-1] / 2.0

        max_idx = np.argmax(self.data[:, -1])
        max_F = self.data[max_idx, -1] / 2.0
        max_F_params = self.data[max_idx, :-1]
        logging.info(
            "F at zero = {:.1f}, marginalised_F = {:.1f},"
            " max_F = {:.1f} ({})".format(
                F_at_zero, marginalised_F, max_F, max_F_params
            )
        )
        return F_at_zero - marginalised_F, (F_at_zero - max_F) / F_at_zero

    def plot_corner(
        self, prior_widths=None, fig=None, axes=None, projection="log_mean"
    ):
        Bsa, FmaxMismatch = self.marginalised_bayes_factor(prior_widths)

        data = self.data[:, -1].reshape(
            (len(self.deltaRadius), len(self.phaseOffset), len(self.deltaPspin))
        )
        xyz = [
            self.deltaRadius / lal.REARTH_SI,
            self.phaseOffset / (np.pi),
            self.deltaPspin / 60.0,
        ]
        labels = [
            r"$\frac{\Delta R}{R_\mathrm{Earth}}$",
            r"$\frac{\Delta \phi}{\pi}$",
            r"$\Delta P_\mathrm{spin}$ [min]",
            r"$2\mathcal{F}$",
        ]

        try:
            from gridcorner import gridcorner
        except ImportError:
            raise ImportError(
                "Python module 'gridcorner' not found, please install from "
                "https://gitlab.aei.uni-hannover.de/GregAshton/gridcorner"
            )

        fig, axes = gridcorner(
            data, xyz, projection=projection, factor=1.6, labels=labels
        )
        axes[-1][-1].axvline((lal.DAYJUL_SI - lal.DAYSID_SI) / 60.0, color="C3")
        plt.suptitle(
            r"T={:.1f} days, $f$={:.2f} Hz, $\log\mathcal{{B}}_{{S/A}}$={:.1f},"
            r" $\frac{{\mathcal{{F}}_0-\mathcal{{F}}_\mathrm{{max}}}}"
            r"{{\mathcal{{F}}_0}}={:.1e}$".format(
                self.duration / 86400, self.F0, Bsa, FmaxMismatch
            ),
            y=0.99,
            size=14,
        )
        fig.savefig(os.path.join(self.outdir, self.label + "_projection_matrix.png"))

    def plot(self, key, prior_widths=None):
        Bsa, FmaxMismatch = self.marginalised_bayes_factor(prior_widths)

        rescales_defaults = {
            "deltaRadius": 1 / lal.REARTH_SI,
            "phaseOffset": 1 / np.pi,
            "deltaPspin": 1,
        }
        labels = {
            "deltaRadius": r"$\frac{\Delta R}{R_\mathrm{Earth}}$",
            "phaseOffset": r"$\frac{\Delta \phi}{\pi}$",
            "deltaPspin": r"$\Delta P_\mathrm{spin}$ [s]",
        }

        fig, ax = self.plot_1D(
            key, xrescale=rescales_defaults[key], xlabel=labels[key], savefig=False
        )
        ax.set_title(
            r"T={} days, $f$={} Hz, $\log\mathcal{{B}}_{{S/A}}$={:.1f}".format(
                self.duration / 86400, self.F0, Bsa
            )
        )
        fig.tight_layout()
        fig.savefig(os.path.join(self.outdir, self.label + "_1D.png"))


class DMoff_NO_SPIN(GridSearch):
    """ DMoff test using SSBPREC_NO_SPIN """

    @helper_functions.initializer
    def __init__(
        self,
        par,
        label,
        outdir,
        sftfilepattern,
        minStartTime=None,
        maxStartTime=None,
        minCoverFreq=None,
        maxCoverFreq=None,
        detectors=None,
        injectSources=None,
        assumeSqrtSX=None,
        earth_ephem=None,
        sun_ephem=None,
    ):
        """
        Parameters
        ----------
        par: dict, str
            Either a par dictionary (containing 'F0', 'F1', 'Alpha', 'Delta'
            and 'tref') or a path to a .par file to read in the F0, F1 etc
        label, outdir: str
            A label and directory to read/write data from/to
        sftfilepattern: str
            Pattern to match SFTs using wildcards (*?) and ranges [0-9];
            mutiple patterns can be given separated by colons.
        minStartTime, maxStartTime: int
            GPS seconds of the start time and end time

        For all other parameters, see `pyfstat.ComputeFStat` for details
        """

        self._set_init_params_dict(locals())
        if os.path.isdir(outdir) is False:
            os.mkdir(outdir)

        if type(par) == dict:
            self.par = par
        elif type(par) == str and os.path.isfile(par):
            self.par = read_par(filename=par)
        else:
            raise ValueError("The .par file does not exist")

        self.nsegs = 1
        self.BSGL = False

        self.tref = self.par["tref"]
        self.F1s = [self.par.get("F1", 0)]
        self.F2s = [self.par.get("F2", 0)]
        self.Alphas = [self.par["Alpha"]]
        self.Deltas = [self.par["Delta"]]
        self.Re = 6.371e6
        self.c = 2.998e8
        a0 = self.Re / self.c  # *np.cos(self.par['Delta'])
        self.m0 = np.max([4, int(np.ceil(2 * np.pi * self.par["F0"] * a0))])
        logging.info("Setting up DMoff_NO_SPIN search with m0 = {}".format(self.m0))

    def get_results(self):
        """ Compute the three summed detection statistics

        Returns
        -------
            m0, twoF_SUM, twoFstar_SUM_SIDEREAL, twoFstar_SUM_TERRESTRIAL

        """
        self.SSBprec = lalpulsar.SSBPREC_RELATIVISTIC
        self.set_out_file("SSBPREC_RELATIVISTIC")
        self.F0s = [self.par["F0"] + j / lal.DAYSID_SI for j in range(-4, 5)]
        self.run()
        twoF_SUM = np.sum(self.data[:, -1])

        self.SSBprec = lalpulsar.SSBPREC_NO_SPIN
        self.set_out_file("SSBPREC_NO_SPIN")
        self.F0s = [
            self.par["F0"] + j / lal.DAYSID_SI for j in range(-self.m0, self.m0 + 1)
        ]
        self.run()
        twoFstar_SUM = np.sum(self.data[:, -1])

        self.set_out_file("SSBPREC_NO_SPIN_TERRESTRIAL")
        self.F0s = [
            self.par["F0"] + j / lal.DAYJUL_SI for j in range(-self.m0, self.m0 + 1)
        ]
        self.run()
        twoFstar_SUM_terrestrial = np.sum(self.data[:, -1])

        return self.m0, twoF_SUM, twoFstar_SUM, twoFstar_SUM_terrestrial
