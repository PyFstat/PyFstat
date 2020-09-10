""" Searches using grid-based methods """


import os
import logging
import itertools
from collections import OrderedDict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re

import pyfstat.helper_functions as helper_functions
from pyfstat.core import (
    BaseSearchClass,
    ComputeFstat,
    SemiCoherentGlitchSearch,
    SemiCoherentSearch,
    tqdm,
    args,
    DefunctClass,
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
            )

            def cut_out_tstart_tend(*vals):
                return self.search.get_semicoherent_det_stat(*vals[2:])

            self.search.get_det_stat = cut_out_tstart_tend

    def get_array_from_tuple(self, x):
        if len(x) == 1:
            return np.array(x)
        elif len(x) == 3 and self.input_arrays is False:
            # This used to be
            # return np.arange(x[0], x[1], x[2])
            # but according to the numpy docs:
            # "When using a non-integer step, such as 0.1,
            # the results will often not be consistent.
            # It is better to use numpy.linspace for these cases."
            # and indeed it sometimes included the end point, sometimes didn't
            return np.linspace(
                x[0], x[1], num=int((x[1] - x[0]) / x[2]) + 1, endpoint=True
            )
        else:
            logging.info("Using tuple of length {:d} as is.".format(len(x)))
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
        old_params_dict_str_list = (
            helper_functions.read_parameters_dict_lines_from_file_header(self.out_file)
        )
        new_params_dict_str_list = [
            line.strip(" ") for line in self.pprint_init_params_dict()[1:-1]
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
        if np.shape(old_data)[0] != np.shape(new_data)[0]:
            logging.info(
                "Old data found in '{:s}', but differs"
                " in length ({:d} points in file, {:d} points requested);"
                " continuing with grid search.".format(
                    self.out_file, np.shape(old_data)[0], np.shape(new_data)[0]
                )
            )
            return False
        if np.shape(old_data)[1] < np.shape(new_data)[1]:
            logging.info(
                "Old data found in '{:s}', but has less columns ({:d})"
                " than new input parameters grid ({:d});"
                " continuing with grid search.".format(
                    self.out_file, np.shape(old_data)[1], np.shape(new_data)[1]
                )
            )
            return False
        # not yet explicitly testing the case of
        # np.shape(old_data)[1] >= np.shape(new_data)[1]
        # because output file can have detstat and post-proc quantities
        # added and hence have different number of dimensions
        # (this could in principle be cleverly predicted at this point)
        # and the np.allclose() check should safely catch those situations
        rtol, atol = self._get_tolerance_from_savetxt_fmt()
        column_matches = [
            np.allclose(
                old_data[:, n],
                new_data[:, n],
                rtol=rtol[n],
                atol=atol[n],
            )
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
        logging.info(
            "Running search over a total of {:d} grid points...".format(
                np.shape(iterable)[0]
            )
        )
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
        """Plots a 2D grid of 2F values

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
        """Get the maximum twoF over the grid

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
        if self.outputTransientFstatMap:
            self.tCWfilebase = os.path.splitext(self.out_file)[0] + "_tCW_"
            logging.info(
                "Will save per-Doppler Fstatmap"
                " results to {}*.dat".format(self.tCWfilebase)
            )

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
        self.timingFstatMap = 0.0
        logging.info(
            "Running search over a total of {:d} grid points...".format(
                np.shape(self.input_data)[0]
            )
        )
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
                    tCWfile = self.get_transient_fstat_map_filename(vals)
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

    def get_transient_fstat_map_filename(self, param_point):
        """per-Doppler filename convention: freq alpha delta f1dot f2dot"""
        fmt_keys = ["F0", "Alpha", "Delta", "F1", "F2"]
        fmt = "{:.16g}_{:.16g}_{:.16g}_{:.16g}_{:.16g}"
        if isinstance(param_point, dict):
            vals = [param_point[key] for key in fmt_keys]
        elif isinstance(param_point, list) or isinstance(param_point, np.ndarray):
            vals = [param_point[self.keys.index(key)] for key in fmt_keys]
        else:
            raise ValueError("param_point must be a dict, list or numpy array!")
        f = self.tCWfilebase + fmt.format(*vals) + ".dat"
        return f

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


class SliceGridSearch(DefunctClass):
    last_supported_version = "1.9.0"


class GridUniformPriorSearch(DefunctClass):
    last_supported_version = "1.9.0"


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


class SlidingWindow(DefunctClass):
    last_supported_version = "1.9.0"


class FrequencySlidingWindow(DefunctClass):
    last_supported_version = "1.9.0"


class EarthTest(DefunctClass):
    last_supported_version = "1.9.0"


class DMoff_NO_SPIN(DefunctClass):
    last_supported_version = "1.9.0"
