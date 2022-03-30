"""PyFstat search classes using grid-based methods."""


import itertools
import logging
import os
import re
from collections import OrderedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pyfstat.helper_functions as helper_functions
from pyfstat.core import (
    BaseSearchClass,
    ComputeFstat,
    DefunctClass,
    SemiCoherentGlitchSearch,
    SemiCoherentSearch,
    args,
    tqdm,
)


class GridSearch(BaseSearchClass):
    """A search evaluating the F-statistic over a regular grid in parameter space.

    This implements a simple 'square box' grid
    with fixed spacing and ranges in each dimension,
    i.e. for each parameter there's a simple 1D list of grid points
    and the total grid is just the Cartesian product of these.

    For N parameter space dimensions and a total of M points in the product grid,
    the basic output is a (N+1,M)-dimensional array with the detection statistic
    (twoF or log10BSGL) appended.

    NOTE: if a large number of grid points are used, checks against cached
    data may be slow as the array is loaded into memory. To avoid this, run
    with the `clean` option which uses a generator instead.

    Most parameters are the same as for the `core.ComputeFstat` class,
    only the additional ones are documented here:
    """

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
        label: str
            Output filenames will be constructed using this label.
        outdir: str
            Output directory.
        F0s, F1s, F2s, Alphas, Deltas: tuple
            A length 3 tuple describing the grid for each parameter,
            e.g [F0min, F0max, dF0].
            Alternatively, for a fixed value simply give [F0].
            Unless `input_arrays=True`, then these are the exact arrays to search over.
        nsegs: int
            Number of segments to split the data set into.
            If `nsegs=1`, the basic ComputeFstat class is used.
            If `nsegs>1`, the SemiCoherentSearch class is used.
        input_arrays: bool
            If true, use the F0s, F1s, etc as arrays just as they are given
            (do not interpret as 3-tuples of [min,max,step]).
        """

        self._set_init_params_dict(locals())
        os.makedirs(outdir, exist_ok=True)
        self.set_out_file()
        for k in self.default_search_keys:
            setattr(self, k, np.atleast_1d(getattr(self, k + "s")))
        self.search_keys = self.default_search_keys.copy()
        if self.BSGL:
            self.detstat = "log10BSGL"
        else:
            self.detstat = "twoF"
        self._initiate_search_object()
        self._set_output_keys()
        self.output_file_header = self.get_output_file_header()

    def _set_output_keys(self):
        self.output_keys = self.search_keys.copy()
        self.output_keys.append("twoF")
        if self.search.singleFstats:
            self.output_keys += [f"twoF{IFO}" for IFO in self.search.detector_names]
        if self.BSGL:
            self.output_keys.append(self.detstat)

    def _get_search_ranges(self):
        if (self.minCoverFreq is None) or (self.maxCoverFreq is None):
            return {key: getattr(self, key + "s") for key in self.search_keys}
        else:
            return None

    def _initiate_search_object(self):
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
        # make sure to overwrite the min/max starttime in case the user
        # passed None and they were read from SFTs
        self.minStartTime = self.search.minStartTime
        self.maxStartTime = self.search.maxStartTime

    def _get_array_from_tuple(self, x):
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

    def _get_input_data_array(self):
        """Set up an input data array, i.e. the product array over search dimensions.

        This is a numpy structured array with named columns
        and explicit dtype (cannot have named columns without that).
        (Will also ensure safety when reading/saving data from/to .txt files.)
        """
        logging.info("Generating input data array")
        coord_arrays = []
        for sl in self.search_keys:
            coord_arrays.append(
                self._get_array_from_tuple(np.atleast_1d(getattr(self, sl)))
            )
        self.coord_arrays = coord_arrays
        self.total_iterations = np.prod([len(ca) for ca in coord_arrays])

        if args.clean is False:
            input_data = []
            for vals in itertools.product(*coord_arrays):
                input_data.append(vals)
            input_dtype = np.dtype(
                {
                    "names": self.search_keys,
                    "formats": np.repeat(float, len(self.search_keys)),
                }
            )
            self.input_data = np.array(input_data, dtype=input_dtype)

    def check_old_data_is_okay_to_use(self):
        """Check if an existing output file matches this search and reuse the results.

        Results will be loaded from old output file,
        and no new search run, if all of the following checks pass:

        1. Output file with matching name found in `outdir`.

        2. Output file is not older than SFT files matching `sftfilepattern`.

        3. Parameters string in file header matches current search setup.

        4. Data in old file can be loaded successfully,
           its input parts (i.e. minus the detection statistic columns)
           matches in dimension with current grid,
           and the values in those input columns match with the current grid.

        Through `helper_functions.read_txt_file_with_header()`,
        the existing file is read in with `np.genfromtxt()`.
        """
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
        old_data = helper_functions.read_txt_file_with_header(self.out_file, names=True)
        if len(old_data) != len(self.input_data):
            logging.info(
                "Old data found in '{:s}', but differs"
                " in length ({:d} points in file, {:d} points requested);"
                " continuing with grid search.".format(
                    self.out_file, np.shape(old_data)[0], np.shape(self.input_data)[0]
                )
            )
            return False
        if len(old_data.dtype) < len(self.input_data.dtype):
            logging.info(
                "Old data found in '{:s}', but has less columns ({:d})"
                " than new input parameters grid ({:d});"
                " continuing with grid search.".format(
                    self.out_file, np.shape(old_data)[1], np.shape(self.input_data)[1]
                )
            )
            return False
        # not yet explicitly testing the case of
        # len(old_data.dtype) >= len(self.input_data.dtype)
        # because output file can have detstat and post-proc quantities
        # added and hence have different number of dimensions
        # (this could in principle be cleverly predicted at this point)
        # and the np.allclose() check should safely catch those situations
        rtol, atol = self._get_tolerance_from_savetxt_fmt()
        column_matches = [
            np.allclose(
                old_data[key],
                self.input_data[key],
                rtol=rtol[key],
                atol=atol[key],
            )
            for key in self.search_keys
        ]
        if np.all(column_matches):
            logging.info(
                "Old data found in '{:s}' with matching input parameters grid,"
                " no search performed. Data grid size: {:d}x{:d}".format(
                    self.out_file, len(old_data), len(old_data.dtype)
                )
            )
            return old_data
        else:
            logging.info(
                "Old data found in '{:s}', input parameters grid differs,"
                "  continuing with grid search.".format(self.out_file)
            )
            return False
        return False

    def _get_single_cand_results(self, data, n, vals):
        """Obtain the results at each grid iteration step.

        Re-formats the input params into the format required by self.search.get_det_stat()
        and gets additional detection statistic by-products as needed,
        then stores them all into the right columns of the data array.
        Can be overridden e.g. for legacy get_det_stat() variants.

        Parameters
        ----------
        data: np.ndarray
            The inputs+outputs data set to be updated.
        n: int
            Row index of the data array to update.
        vals: tuple
            Iteration of our iterable.

        Returns
        -------
        data: np.ndarray
            The updated inputs+outputs data set.
        """
        thisCand = {key: val for key, val in zip(self.search_keys, vals)}
        detstat = self.search.get_det_stat(params=thisCand)
        thisCand["twoF"] = self.search.twoF
        if self.search.singleFstats:
            thisCand["twoFX"] = list(self.search.twoFX[: self.search.numDetectors])
        if self.detstat != "twoF":
            thisCand[self.detstat] = detstat
        for key in self.output_keys:
            if key in thisCand.keys():
                data[key][n] = thisCand[key]
            elif key.startswith("twoF"):
                try:
                    X = self.search.detector_names.index(key.lstrip("twoF"))
                    data[key][n] = thisCand["twoFX"][X]
                except (KeyError, IndexError):
                    raise RuntimeError(
                        f"Could not get value for key {key} from candidate dict {thisCand}."
                    )
            else:
                raise RuntimeError(
                    f"Could not get value for key {key} from candidate dict {thisCand}."
                )
        return data

    def run(self, return_data=False):
        """Execute the actual search over the full grid.

        This iterates over all points in the multi-dimensional product grid
        and the end result is either returned as a numpy array or saved to disk.

        Parameters
        ----------
        return_data: boolean
            If true, the final inputs+outputs data set is returned as a numpy array.
            If false, it is saved to disk and nothing is returned.

        Returns
        -------
        data: np.ndarray
            The final inputs+outputs data set.
            Only if `return_data=true`.
        """
        self._get_input_data_array()

        if args.clean:
            iterable = itertools.product(*self.coord_arrays)
        else:
            old_data = self.check_old_data_is_okay_to_use()
            iterable = self.input_data

            if old_data is not False:
                self.data = old_data
                return

        logging.info(
            "Running search over a total of {:d} grid points...".format(
                np.shape(iterable)[0]
            )
        )
        output_dtype = np.dtype(
            {
                "names": self.output_keys,
                "formats": np.repeat(float, len(self.output_keys)),
            }
        )
        data = np.zeros(len(self.input_data), dtype=output_dtype)
        for n, vals in enumerate(
            tqdm(iterable, total=getattr(self, "total_iterations", None))
        ):
            data = self._get_single_cand_results(data, n, vals)

        if return_data:
            return data
        else:
            self.data = data
            self.save_array_to_disk()

    def _get_savetxt_fmt_dict(self):
        """Define the output precision for each parameter and computed quantity."""
        fmt_dict = helper_functions.get_doppler_params_output_format(
            self.output_keys, self.fmt_doppler
        )
        fmt_dict["twoF"] = self.fmt_detstat
        if self.search.singleFstats:
            for IFO in self.search.detector_names:
                fmt_dict[f"twoF{IFO}"] = self.fmt_detstat
        if self.BSGL:
            fmt_dict["log10BSGL"] = self.fmt_detstat
        return fmt_dict

    def _get_savetxt_fmt_list(self):
        """Returns a list of output format specifiers, ordered like the data.

        This is required because the output of _get_savetxt_fmt_dict()
        will depend on the order in which those entries have been coded up.
        """
        fmt_dict = self._get_savetxt_fmt_dict()
        fmt_list = [fmt_dict[key] for key in self.output_keys]
        return fmt_list

    def _get_tolerance_from_savetxt_fmt(self):
        """Decide appropriate input grid comparison tolerances from fprintf formats."""
        fmt = self._get_savetxt_fmt_dict()
        rtol = {}
        atol = {}
        for key, f in fmt.items():
            if f.endswith("d"):
                rtol[key] = 0
                atol[key] = 0
            elif f.endswith("g"):
                precision = int(re.findall(r"\d+", f)[-1])
                rtol[key] = 10 ** (1 - precision)
                atol[key] = 0
            elif f.endswith("f"):
                decimals = int(re.findall(r"\d+", f)[-1])
                rtol[key] = 0
                atol[key] = 10**-decimals
            else:
                raise ValueError(
                    "Cannot parse fprintf format '{:s}' to obtain recommended tolerance.".format(
                        f
                    )
                )
        return rtol, atol

    def save_array_to_disk(self):
        """Save the results array to a txt file.

        This includes a header with version and parameters information.

        It should be flexible enough to be reused by child classes,
        as long as the `_get_savetxt_fmt_dict() method` is suitably overridden
        to account for any additional parameters.
        """
        logging.info("Saving data to {}".format(self.out_file))
        header = "\n".join(self.output_file_header)
        header += "\n" + " ".join(self.output_keys)
        outfmt = self._get_savetxt_fmt_list()
        Ncols = len(self.data.dtype)
        if len(outfmt) != Ncols:
            raise RuntimeError(
                "Lengths of data rows ({:d})"
                " and output format ({:d})"
                " do not match."
                " If your search class uses different"
                " keys than the base GridSearch class,"
                " override the _get_savetxt_fmt_dict"
                " method.".format(Ncols, len(outfmt))
            )
        np.savetxt(
            self.out_file,
            np.nan_to_num(self.data),
            delimiter=" ",
            header=header,
            fmt=outfmt,
        )

    def _convert_F0_to_mismatch(self, F0, F0hat, Tseg):
        DeltaF0 = F0[1] - F0[0]
        m_spacing = (np.pi * Tseg * DeltaF0) ** 2 / 12.0
        N = len(F0)
        return np.arange(-N * m_spacing / 2.0, N * m_spacing / 2.0, m_spacing)

    def _convert_F1_to_mismatch(self, F1, F1hat, Tseg):
        DeltaF1 = F1[1] - F1[0]
        m_spacing = (np.pi * Tseg**2 * DeltaF1) ** 2 / 720.0
        N = len(F1)
        return np.arange(-N * m_spacing / 2.0, N * m_spacing / 2.0, m_spacing)

    def _add_mismatch_to_ax(self, ax, x, y, xkey, ykey, xhat, yhat, Tseg):
        axX = ax.twiny()
        axX.zorder = -10
        axY = ax.twinx()
        axY.zorder = -10
        if xkey == "F0":
            m = self._convert_F0_to_mismatch(x, xhat, Tseg)
            axX.set_xlim(m[0], m[-1])
        if ykey == "F1":
            m = self._convert_F1_to_mismatch(y, yhat, Tseg)
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
        """Make a plot of the detection statistic over a single grid dimension.

        Parameters
        ----------
        xkey: str
            The name of the search parameter to plot against.
        ax: matplotlib.axes._subplots_AxesSubplot or None
            An optional pre-existing axes set to draw into.
        x0: float or None
            Plot x values relative to this central value.
        xrescale: float
            Rescale all x values by this factor.
        savefig : bool
            If true, save the figure in `self.outdir`.
            If false, return an axis object without saving to disk.
        xlabel: str or None
            Override default text label for the x-axis.
        ylabel: str or None
            Override default text label for the y-axis.
        agg_chunksize: int or None
            Set this to some high value to work around
            matplotlib 'Exceeded cell block limit' errors.

        Returns
        -------
        ax: matplotlib.axes._subplots_AxesSubplot, optional
            The axes object containing the plot, only if `savefig=false`.
        """
        if agg_chunksize:
            # FIXME: workaround for matplotlib "Exceeded cell block limit" errors
            plt.rcParams["agg.path.chunksize"] = agg_chunksize
        if ax is None:
            fig, ax = plt.subplots()
        # x = np.unique(self.data[xkey]) # this doesn't work for multi-dim searches!
        x = self.data[xkey]
        if x0:
            x = x - x0
        x = x * xrescale
        z = self.data[self.detstat]
        ax.plot(x, z)
        if xlabel:
            ax.set_xlabel(xlabel)
        elif x0:
            ax.set_xlabel(self.tex_label0(xkey))
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
        savefig=True,
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
        """Plots the detection statistic over a 2D grid.

        FIXME: this will currently fail if the search went over >2 dimensions.

        Parameters
        ----------
        xkey: str
            The name of the first search parameter to plot against.
        ykey: str
            The name of the second search parameter to plot against.
        ax: matplotlib.axes._subplots_AxesSubplot or None
            An optional pre-existing axes set to draw into.
        savefig: bool
            If true, save the figure in `self.outdir`.
            If false, return an axis object without saving to disk.
        vmin, vmax: float or None
            Cutoffs for rescaling the colormap.
        add_mismatch: tuple or None
            If given a tuple `(xhat, yhat, Tseg)`,
            add a secondary axis with the metric mismatch from the
            point `(xhat, yhat)` with duration `Tseg`.
        xN, yN: int or  None
            Number of tick label intervals.
        flat_keys: list
            Keys to be used in flattening higher-dimensional arrays.
        rel_flat_idxs: list
            Indices to be used in flattening higher-dimensional arrays.
        flatten_method: numpy function
            Function to use in flattening the `flat_keys`,
            default: `np.max`.
        title: str or None
            Optional plot title text.
        predicted_twoF: float or None
            Expected/predicted value of twoF,
            used to rescale the z-axis.
        cm: matplotlib.colors.ListedColormap or None
            Override standard (viridis) colormap.
        cbarkwargs: dict
            Additional arguments for colorbar formatting.
        x0: float
            Plot x values relative to this central value.
        y0: float
            Plot y values relative to this central value.
        xrescale: float
            Rescale all x values by this factor.
        yrescale: float
            Rescale all y values by this factor.
        xlabel: str
            Override default text label for the x-axis.
        ylabel: str
            Override default text label for the y-axis.
        zlabel: str
            Override default text label for the z-axis.

        Returns
        -------
        ax: matplotlib.axes._subplots_AxesSubplot, optional
            The axes object containing the plot, only if `savefig=false`.
        """
        if ax is None:
            fig, ax = plt.subplots()
        flat_idxs = [self.search_keys.index(k) for k in flat_keys]

        x = np.unique(self.data[xkey])
        if x0:
            x = x - x0
        y = np.unique(self.data[ykey])
        if y0:
            y = y - y0
        flat_vals = [np.unique(self.data[:, j]) for j in flat_idxs]

        z = self.data[self.detstat]

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
            X * xrescale,
            Y * yrescale,
            Z,
            cmap=cm,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
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
            ax.set_xlabel(self.tex_label0(xkey))
        else:
            ax.set_xlabel(self.tex_labels[xkey])

        if ylabel:
            ax.set_ylabel(ylabel)
        elif y0:
            ax.set_ylabel(self.tex_label0(ykey))
        else:
            ax.set_ylabel(self.tex_labels[ykey])

        if title:
            ax.set_title(title)

        if xN:
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(xN))
        if yN:
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(yN))

        if savefig:
            fig.tight_layout()
            fname = "{}_2D_{}_{}_{}.png".format(self.label, xkey, ykey, self.detstat)
            fig.savefig(os.path.join(self.outdir, fname))
        else:
            return ax

    def get_max_det_stat(self):
        """Get the maximum detection statistic over the grid.

        This requires the `run()` method to have been called before.

        Returns
        -------
        d: dict
            Dictionary containing parameters and detection statistic at the maximum.
        """
        idx = np.argmax(self.data[self.detstat])
        d = OrderedDict([(key, self.data[key][idx]) for key in self.output_keys])
        return d

    def get_max_twoF(self):
        """Get the maximum twoF over the grid.

        This requires the `run()` method to have been called before.

        Returns
        -------
        d: dict
            Dictionary containing parameters and twoF value at the maximum.
        """
        idx = np.argmax(self.data["twoF"])
        d = OrderedDict([(key, self.data[key][idx]) for key in self.output_keys])
        return d

    def print_max_twoF(self):
        """Get and print the maximum twoF point over the grid.

        This prints out the full dictionary from `get_max_twoF()`,
        i.e. the maximum value and its corresponding parameters.
        """
        d = self.get_max_twoF()
        print("Grid point with max(twoF) for {}:".format(self.label))
        for k, v in d.items():
            print("  {}={}".format(k, v))

    def generate_loudest(self):
        """Use lalapps_ComputeFstatistic_v2 to produce a .loudest file"""
        max_params = self.get_max_twoF()
        max_params.pop("twoF")
        max_params = self.translate_keys_to_lal(max_params)
        self.loudest_file = helper_functions.generate_loudest_file(
            max_params=max_params,
            tref=self.tref,
            outdir=self.outdir,
            label=self.label,
            sftfilepattern=self.sftfilepattern,
            minStartTime=self.minStartTime,
            maxStartTime=self.maxStartTime,
            transientWindowType=getattr(self, "transientWindowType", None),
            earth_ephem=self.earth_ephem,
            sun_ephem=self.sun_ephem,
        )

    def set_out_file(self, extra_label=None):
        """Set (or reset) the name of the main output file.

        File will always be stored in `self.outdir`
        and the base of the name be determined from `self.label` and other
        parts of the search setup,
        but this method allows to attach an `extra_label` bit if desired.

        Parameters
        -------
        extra_label: str
            Additional text bit to be attached at the end of the filename
            (but before the extension).
        """
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
    """A search for transient CW-like signals using the F-statistic.

    This is based on the transient signal model and transient-F-stat algorithm
    from Prix, Giampanis & Messenger (PRD 84, 023007, 2011):
    https://arxiv.org/abs/1104.1704

    The frequency evolution parameters are searched over in a grid just like
    in the normal `GridSearch`, then at each point the time-dependent 'atoms'
    are used to evaluate partial sums of the F-statistic over a 2D array
    in transient start times `t0` and duration parameters `tau`.

    The signal templates are modulated by a 'transient window function' which can be

    1. `none` (standard, persistent CW signal)

    2. `rect` (rectangular: constant amplitude within `[t0,t0+tau]`, zero outside)

    3. `exp` (exponential decay over `[t0,t0+3*tau]`, zero outside)

    This class currently only supports fully-coherent searches (`nsegs=1` is hardcoded).

    Also see Keitel & Ashton (CQG 35, 205003, 2018):
    https://arxiv.org/abs/1805.05652
    for a detailed discussion of the GPU implementation.

    Most parameters are the same as for `GridSearch`
    and the `core.ComputeFstat` class,
    only the additional ones are documented here:
    """

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
        transientWindowType: str
            If `rect` or `exp`,
            allow for the Fstat to be computed over a transient range.
            (`none` instead of `None` explicitly calls the transient-window
            function, but with the full range, for debugging.)
        t0Band, tauBand: int
            Search ranges for transient start-time t0 and duration tau.
            If >0, search `t0` in `(minStartTime,minStartTime+t0Band)`
            and tau in `(tauMin,2*Tsft+tauBand)`.
            If =0, only compute the continuous-wave F-stat with `t0=minStartTime`,
            `tau=maxStartTime-minStartTime`.
        tauMin: int
            Minimum transient duration to cover,
            defaults to `2*Tsft`.
        dt0: int
            Grid resolution in transient start-time,
            defaults to `Tsft`.
        dtau: int
            Grid resolution in transient duration,
            defaults to `Tsft`.
        outputTransientFstatMap: bool
            If true, write additional output files for `(t0,tau)` F-stat maps.
            (One file for each grid point!)
        outputAtoms: bool
            If true, write additional output files for the F-stat `atoms`.
            (One file for each grid point!)
        tCWFstatMapVersion: str
            Choose between implementations of the transient F-statistic funcionality:
            standard `lal` implementation,
            `pycuda` for GPU version,
            and some others only for devel/debug.
        cudaDeviceName: str
            GPU name to be matched against drv.Device output,
            only for `tCWFstatMapVersion=pycuda`.
        """

        self._set_init_params_dict(locals())
        self.nsegs = 1
        os.makedirs(outdir, exist_ok=True)
        self.set_out_file()
        for k in self.default_search_keys:
            setattr(self, k, np.atleast_1d(getattr(self, k + "s")))
        self.search_keys = self.default_search_keys.copy()
        if self.BSGL:
            self.detstat = "log10BSGL"
        else:
            self.detstat = "maxTwoF"
        self._initiate_search_object()
        self._set_output_keys()
        self.output_file_header = self.get_output_file_header()
        if self.outputTransientFstatMap:
            self.tCWfilebase = os.path.splitext(self.out_file)[0] + "_tCW_"
            logging.info(
                "Will save per-Doppler Fstatmap"
                " results to {}*.dat".format(self.tCWfilebase)
            )

    def _set_output_keys(self):
        self.output_keys = self.search_keys.copy()
        self.output_keys.append("twoF")
        if self.search.singleFstats:
            self.output_keys += [f"twoF{IFO}" for IFO in self.search.detector_names]
        if self.transientWindowType is not None:
            self.output_keys.append("maxTwoF")
            if hasattr(self.search, "twoFXatMaxTwoF"):
                self.output_keys += [
                    f"twoF{IFO}atMaxTwoF" for IFO in self.search.detector_names
                ]
        if self.detstat not in ["twoF", "maxTwoF"]:
            self.output_keys.append(self.detstat)
        if self.transientWindowType is not None:
            # for consistency, t0/tau must come after detstat
            # they are not included in self.search_keys because the main Fstat
            # code does not loop over them
            self.output_keys += ["t0", "tau"]

    def _initiate_search_object(self):
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
        # make sure to overwrite the min/max starttime in case the user
        # passed None and they were read from SFTs
        self.minStartTime = self.search.minStartTime
        self.maxStartTime = self.search.maxStartTime

    def _get_single_cand_results(self, data, n, vals):
        """Obtain the results at each grid iteration step.

        Re-formats the input params into the format required by self.search.get_det_stat()
        and gets additional detection statistic by-products as needed,
        then stores them all into the right columns of the data array.

        We override the default method here to get the extra transient output values.

        Parameters
        ----------
        data: np.ndarray
            The inputs+outputs data set to be updated.
        n: int
            Row index of the data array to update.
        vals: tuple
            Iteration of our iterable.

        Returns
        -------
        data: np.ndarray
            The updated inputs+outputs data set.
        """
        thisCand = {key: val for key, val in zip(self.search_keys, vals)}
        detstat = self.search.get_det_stat(params=thisCand)
        windowRange = getattr(self.search, "windowRange", None)
        self.timingFstatMap += getattr(self.search, "timingFstatMap", 0.0)
        thisCand["twoF"] = self.search.twoF
        if self.search.singleFstats:
            thisCand += list(self.search.twoFX[: self.search.numDetectors])
        if windowRange is not None:
            thisCand["maxTwoF"] = self.search.maxTwoF
            if hasattr(self.search, "twoFXatMaxTwoF"):
                thisCand += list(self.search.twoFXatMaxTwoF[: self.search.numDetectors])
        if self.detstat not in ["twoF", "maxTwoF"]:
            thisCand[self.detstat] = detstat
        if getattr(self, "transientWindowType", None):
            if not hasattr(self.search, "FstatMap"):
                raise RuntimeError(
                    "Since transientWindowType!=None, we expected to have a FstatMap."
                )
            if self.outputTransientFstatMap:
                tCWfile = self.get_transient_fstat_map_filename(thisCand)
                self.search.FstatMap.write_F_mn_to_file(
                    tCWfile, windowRange, self.output_file_header
                )
            maxidx = self.search.FstatMap.get_maxF_idx()
            thisCand["t0"] = windowRange.t0 + maxidx[0] * windowRange.dt0
            thisCand["tau"] = windowRange.tau + maxidx[1] * windowRange.dtau
        for key in self.output_keys:
            data[key][n] = thisCand[key]
        return data

    def run(self, return_data=False):
        """Execute the actual search over the full grid.

        This iterates over all points in the multi-dimensional product grid
        and the end result is either returned as a numpy array or saved to disk.

        If the `outputTransientFstatMap` or `outputAtoms` options have been set
        when initiating the search,
        additional files are written for each
        frequency-evolution parameter-space point ('Doppler' point).

        Parameters
        ----------
        return_data: boolean
            If true, the final inputs+outputs data set is returned as a numpy array.
            If false, it is saved to disk and nothing is returned.

        Returns
        -------
        data: np.ndarray
            The final inputs+outputs data set.
            Only if `return_data=true`.
        """
        self._get_input_data_array()
        old_data = self.check_old_data_is_okay_to_use()
        if old_data is not False:
            self.data = old_data
            return

        output_dtype = np.dtype(
            {
                "names": self.output_keys,
                "formats": np.repeat(float, len(self.output_keys)),
            }
        )
        data = np.zeros(len(self.input_data), dtype=output_dtype)
        self.timingFstatMap = 0.0
        logging.info(
            "Running search over a total of {:d} grid points...".format(
                np.shape(self.input_data)[0]
            )
        )
        for n, vals in enumerate(tqdm(self.input_data)):
            data = self._get_single_cand_results(data, n, vals)
            if self.outputAtoms:
                self.search.write_atoms_to_file(os.path.splitext(self.out_file)[0])

        logging.info(
            "Total time spent computing transient F-stat maps: {:.2f}s".format(
                self.timingFstatMap
            )
        )

        if return_data:
            return data
        else:
            self.data = data
            self.save_array_to_disk()

    def get_transient_fstat_map_filename(self, param_point):
        """Filename convention for given grid point: freq_alpha_delta_f1dot_f2dot

        FIXME: This does not yet deal properly with spindown terms higher than F2!

        Parameters
        ----------
        param_point: tuple, dict, list, np.void or np.ndarray
            A multi-dimensional parameter point.
            If not a type with named fields (e.g. a plain tuple or list),
            the order must match that of `self.output_keys`.

        Returns
        -------
        f: str
            The constructed filename.
        """
        fmt_keys = ["F0", "Alpha", "Delta", "F1", "F2"]
        fmt = "{:.16g}_{:.16g}_{:.16g}_{:.16g}_{:.16g}"
        if isinstance(param_point, tuple) or isinstance(param_point, np.void):
            param_point = list(param_point)
        if isinstance(param_point, dict):
            vals = [param_point[key] for key in fmt_keys]
        elif isinstance(param_point, list) or isinstance(param_point, np.ndarray):
            vals = [param_point[self.output_keys.index(key)] for key in fmt_keys]
        else:
            raise ValueError("param_point must be a dict, list, tuple or numpy array!")
        f = self.tCWfilebase + fmt.format(*vals) + ".dat"
        return f

    def _get_savetxt_fmt_dict(self):
        """Define the output precision for each parameter and computed quantity."""
        fmt_dict = helper_functions.get_doppler_params_output_format(
            self.output_keys, self.fmt_doppler
        )
        fmt_dict["twoF"] = self.fmt_detstat
        if self.search.singleFstats:
            for IFO in self.search.detector_names:
                fmt_dict[f"twoF{IFO}"] = self.fmt_detstat
        fmt_dict["maxTwoF"] = self.fmt_detstat
        if hasattr(self.search, "twoFXatMaxTwoF"):
            for IFO in self.search.detector_names:
                fmt_dict[f"twoF{IFO}atMaxTwoF"] = self.fmt_detstat
        if self.detstat != "maxTwoF":
            fmt_dict[self.detstat] = self.fmt_detstat
        fmt_dict["t0"] = "%d"
        fmt_dict["tau"] = "%d"
        return fmt_dict

    def __del__(self):
        if hasattr(self, "search"):
            self.search.__del__()


class SliceGridSearch(DefunctClass):
    last_supported_version = "1.9.0"


class GridUniformPriorSearch(DefunctClass):
    last_supported_version = "1.9.0"


class GridGlitchSearch(GridSearch):
    """A grid search using the `SemiCoherentGlitchSearch` class.

    This implements a basic semi-coherent F-stat search in which the data
    is divided into segments either side of the proposed glitch epochs and the
    fully-coherent F-stat in each segment is summed to give the semi-coherent
    F-stat.

    This class currently only works for a single glitch in the observing time.
    """

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
        Most parameters are the same as for `GridSearch`
        and the `core.SemiCoherentGlitchSearch` class,
        only the additional ones are documented here:

        Parameters
        ----------
        delta_F0s: tuple
            A length 3 tuple describing the grid of frequency jumps,
            or just `[delta_F0]` for a fixed value.
        delta_F1s: tuple
            A length 3 tuple describing the grid of spindown parameter jumps,
            or just `[delta_F1]` for a fixed value.
        tglitchs: tuple
            A length 3 tuple describing the grid of glitch epochs,
            or just `[tglitch]` for a fixed value.
            These are relative time offsets, referenced to zero at `minStartTime`.
        """

        self._set_init_params_dict(locals())
        os.makedirs(outdir, exist_ok=True)
        self.set_out_file()
        self.BSGL = False
        self.input_arrays = False
        if tglitchs is None:
            raise ValueError("You must specify `tglitchs`")
        self.search_keys = self.default_search_keys + [
            "delta_F0",
            "delta_F1",
            "tglitch",
        ]
        for k in self.search_keys:
            setattr(self, k, np.atleast_1d(getattr(self, k + "s")))
        self.detstat = "twoF"
        self._initiate_search_object()
        self._set_output_keys()
        self.output_file_header = self.get_output_file_header()

    def _initiate_search_object(self):
        logging.info("Setting up search object")
        search_ranges = self._get_search_ranges()
        self.search = SemiCoherentGlitchSearch(
            label=self.label,
            outdir=self.outdir,
            sftfilepattern=self.sftfilepattern,
            tref=self.tref,
            minStartTime=self.minStartTime,
            maxStartTime=self.maxStartTime,
            minCoverFreq=self.minCoverFreq,
            maxCoverFreq=self.maxCoverFreq,
            search_ranges=search_ranges,
            BSGL=self.BSGL,
            earth_ephem=self.earth_ephem,
            sun_ephem=self.sun_ephem,
        )

    def _get_savetxt_fmt_dict(self):
        """Define the output precision for each parameter and computed quantity."""
        fmt_dict = helper_functions.get_doppler_params_output_format(
            self.output_keys, self.fmt_doppler
        )
        fmt_dict["delta_F0"] = self.fmt_doppler
        fmt_dict["delta_F1"] = self.fmt_doppler
        fmt_dict["tglitch"] = "%d"
        fmt_dict[self.detstat] = self.fmt_detstat
        return fmt_dict

    def _get_single_cand_results(self, data, n, vals):
        """Obtain the results at each grid iteration step.

        FIXME: currently we need to override the default method here
        because the SemiCoherentGlitchSearch.get_semicoherent_nglitch_twoF()
        is not yet ported to the new parameter dictionary style
        of other detection statistics methods.

        Parameters
        ----------
        data: np.ndarray
            The inputs+outputs data set to be updated.
        n: int
            Row index of the data array to update.
        vals: tuple
            Iteration of our iterable.

        Returns
        -------
        data: np.ndarray
            The updated inputs+outputs data set.
        """
        thisCand = list(vals)
        detstat = self.search.get_det_stat(*vals)
        thisCand.append(self.search.twoF)
        if self.search.singleFstats:
            thisCand += list(self.search.twoFX[: self.search.numDetectors])
        if self.detstat != "twoF":
            thisCand.append(detstat)
        for k, key in enumerate(self.output_keys):
            data[key][n] = thisCand[k]
        return data


class SlidingWindow(DefunctClass):
    last_supported_version = "1.9.0"


class FrequencySlidingWindow(DefunctClass):
    last_supported_version = "1.9.0"


class EarthTest(DefunctClass):
    last_supported_version = "1.9.0"


class DMoff_NO_SPIN(DefunctClass):
    last_supported_version = "1.9.0"
