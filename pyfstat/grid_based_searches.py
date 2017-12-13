""" Searches using grid-based methods """
from __future__ import division, absolute_import, print_function

import os
import logging
import itertools
from collections import OrderedDict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.misc import logsumexp

import pyfstat.helper_functions as helper_functions
from pyfstat.core import (BaseSearchClass, ComputeFstat,
                          SemiCoherentGlitchSearch, SemiCoherentSearch, tqdm,
                          args, read_par)
import lalpulsar
import lal


class GridSearch(BaseSearchClass):
    """ Gridded search using ComputeFstat """
    tex_labels = {'F0': '$f$', 'F1': '$\dot{f}$', 'F2': '$\ddot{f}$',
                  'Alpha': r'$\alpha$', 'Delta': r'$\delta$'}
    tex_labels0 = {'F0': '$-f_0$', 'F1': '$-\dot{f}_0$', 'F2': '$-\ddot{f}_0$',
                   'Alpha': r'$-\alpha_0$', 'Delta': r'$-\delta_0$'}

    @helper_functions.initializer
    def __init__(self, label, outdir, sftfilepattern, F0s, F1s, F2s, Alphas,
                 Deltas, tref=None, minStartTime=None, maxStartTime=None,
                 nsegs=1, BSGL=False, minCoverFreq=None, maxCoverFreq=None,
                 detectors=None, SSBprec=None, injectSources=None,
                 input_arrays=False, assumeSqrtSX=None,
                 transientWindowType=None, t0Band=None, tauBand=None,
                 outputTransientFstatMap=False):
        """
        Parameters
        ----------
        label, outdir: str
            A label and directory to read/write data from/to
        sftfilepattern: str
            Pattern to match SFTs using wildcards (*?) and ranges [0-9];
            mutiple patterns can be given separated by colons.
        F0s, F1s, F2s, delta_F0s, delta_F1s, tglitchs, Alphas, Deltas: tuple
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
                   and tau in (2*Tsft,2*Tsft+tauBand).
            if =0, only compute CW Fstat with t0=minStartTime,
                   tau=maxStartTime-minStartTime.
        outputTransientFstatMap: bool
            if true, write output files for (t0,tau) Fstat maps
            (one file for each doppler grid point!)

        For all other parameters, see `pyfstat.ComputeFStat` for details
        """

        if os.path.isdir(outdir) is False:
            os.mkdir(outdir)
        self.set_out_file()
        self.keys = ['_', '_', 'F0', 'F1', 'F2', 'Alpha', 'Delta']
        self.search_keys = [x+'s' for x in self.keys[2:]]
        for k in self.search_keys:
            setattr(self, k, np.atleast_1d(getattr(self, k)))

    def inititate_search_object(self):
        logging.info('Setting up search object')
        if self.nsegs == 1:
            self.search = ComputeFstat(
                tref=self.tref, sftfilepattern=self.sftfilepattern,
                minCoverFreq=self.minCoverFreq, maxCoverFreq=self.maxCoverFreq,
                detectors=self.detectors,
                transientWindowType=self.transientWindowType,
                t0Band=self.t0Band, tauBand=self.tauBand,
                minStartTime=self.minStartTime, maxStartTime=self.maxStartTime,
                BSGL=self.BSGL, SSBprec=self.SSBprec,
                injectSources=self.injectSources,
                assumeSqrtSX=self.assumeSqrtSX)
            self.search.get_det_stat = self.search.get_fullycoherent_twoF
        else:
            self.search = SemiCoherentSearch(
                label=self.label, outdir=self.outdir, tref=self.tref,
                nsegs=self.nsegs, sftfilepattern=self.sftfilepattern,
                BSGL=self.BSGL, minStartTime=self.minStartTime,
                maxStartTime=self.maxStartTime, minCoverFreq=self.minCoverFreq,
                maxCoverFreq=self.maxCoverFreq, detectors=self.detectors,
                injectSources=self.injectSources)

            def cut_out_tstart_tend(*vals):
                return self.search.get_semicoherent_twoF(*vals[2:])
            self.search.get_det_stat = cut_out_tstart_tend

    def get_array_from_tuple(self, x):
        if len(x) == 1:
            return np.array(x)
        elif len(x) == 3 and self.input_arrays is False:
            return np.arange(x[0], x[1], x[2])
        else:
            logging.info('Using tuple as is')
            return np.array(x)

    def get_input_data_array(self):
        logging.info("Generating input data array")
        coord_arrays = []
        for tup in ([self.minStartTime], [self.maxStartTime], self.F0s,
                    self.F1s, self.F2s, self.Alphas, self.Deltas):
            coord_arrays.append(self.get_array_from_tuple(tup))

        input_data = []
        for vals in itertools.product(*coord_arrays):
                input_data.append(vals)
        self.input_data = np.array(input_data)
        self.coord_arrays = coord_arrays

    def check_old_data_is_okay_to_use(self):
        if args.clean:
            return False
        if os.path.isfile(self.out_file) is False:
            logging.info('No old data found, continuing with grid search')
            return False
        if self.sftfilepattern is not None:
            oldest_sft = min([os.path.getmtime(f) for f in
                              self._get_list_of_matching_sfts()])
            if os.path.getmtime(self.out_file) < oldest_sft:
                logging.info('Search output data outdates sft files,'
                             + ' continuing with grid search')
                return False

        data = np.atleast_2d(np.genfromtxt(self.out_file, delimiter=' '))
        if np.all(data[:, 0:-1] == self.input_data):
            logging.info(
                'Old data found with matching input, no search performed')
            return data
        else:
            logging.info(
                'Old data found, input differs, continuing with grid search')
            return False
        return False

    def run(self, return_data=False):
        self.get_input_data_array()
        old_data = self.check_old_data_is_okay_to_use()
        if old_data is not False:
            self.data = old_data
            return

        if hasattr(self, 'search') is False:
            self.inititate_search_object()

        data = []
        for vals in tqdm(self.input_data):
            detstat = self.search.get_det_stat(*vals)
            windowRange = getattr(self.search, 'windowRange', None)
            FstatMap = getattr(self.search, 'FstatMap', None)
            thisCand = list(vals) + [detstat]
            if self.transientWindowType:
                if self.outputTransientFstatMap:
                    tCWfile = os.path.splitext(self.out_file)[0]+'_tCW_%.16f_%.16f_%.16f_%.16g_%.16g.dat' % (vals[2],vals[5],vals[6],vals[3],vals[4]) # freq alpha delta f1dot f2dot
                    fo = lal.FileOpen(tCWfile, 'w')
                    lalpulsar.write_transientFstatMap_to_fp ( fo, FstatMap, windowRange, None )
                    del fo # instead of lal.FileClose() which is not SWIG-exported
                Fmn = FstatMap.F_mn.data
                maxidx = np.unravel_index(Fmn.argmax(), Fmn.shape)
                thisCand += [windowRange.t0+maxidx[0]*windowRange.dt0,
                             windowRange.tau+maxidx[1]*windowRange.dtau]
            data.append(thisCand)

        data = np.array(data, dtype=np.float)
        if return_data:
            return data
        else:
            logging.info('Saving data to {}'.format(self.out_file))
            np.savetxt(self.out_file, data, delimiter=' ')
            self.data = data

    def convert_F0_to_mismatch(self, F0, F0hat, Tseg):
        DeltaF0 = F0[1] - F0[0]
        m_spacing = (np.pi*Tseg*DeltaF0)**2 / 12.
        N = len(F0)
        return np.arange(-N*m_spacing/2., N*m_spacing/2., m_spacing)

    def convert_F1_to_mismatch(self, F1, F1hat, Tseg):
        DeltaF1 = F1[1] - F1[0]
        m_spacing = (np.pi*Tseg**2*DeltaF1)**2 / 720.
        N = len(F1)
        return np.arange(-N*m_spacing/2., N*m_spacing/2., m_spacing)

    def add_mismatch_to_ax(self, ax, x, y, xkey, ykey, xhat, yhat, Tseg):
        axX = ax.twiny()
        axX.zorder = -10
        axY = ax.twinx()
        axY.zorder = -10

        if xkey == 'F0':
            m = self.convert_F0_to_mismatch(x, xhat, Tseg)
            axX.set_xlim(m[0], m[-1])

        if ykey == 'F1':
            m = self.convert_F1_to_mismatch(y, yhat, Tseg)
            axY.set_ylim(m[0], m[-1])

    def plot_1D(self, xkey, ax=None, x0=None, xrescale=1, savefig=True,
                xlabel=None, ylabel='$\widetilde{2\mathcal{F}}$'):
        if ax is None:
            fig, ax = plt.subplots()
        xidx = self.keys.index(xkey)
        x = np.unique(self.data[:, xidx])
        if x0:
            x = x - x0
        x = x * xrescale
        z = self.data[:, -1]
        ax.plot(x, z)
        if x0:
            ax.set_xlabel(self.tex_labels[xkey]+self.tex_labels0[xkey])
        else:
            ax.set_xlabel(self.tex_labels[xkey])

        if xlabel:
            ax.set_xlabel(xlabel)

        ax.set_ylabel(ylabel)
        if savefig:
            fig.tight_layout()
            fig.savefig('{}/{}_1D.png'.format(self.outdir, self.label))
        else:
            return fig, ax

    def plot_2D(self, xkey, ykey, ax=None, save=True, vmin=None, vmax=None,
                add_mismatch=None, xN=None, yN=None, flat_keys=[],
                rel_flat_idxs=[], flatten_method=np.max, title=None,
                predicted_twoF=None, cm=None, cbarkwargs={}, x0=None, y0=None,
                colorbar=False):
        """ Plots a 2D grid of 2F values

        Parameters
        ----------
        add_mismatch: tuple (xhat, yhat, Tseg)
            If not None, add a secondary axis with the metric mismatch from the
            point xhat, yhat with duration Tseg
        flatten_method: np.max
            Function to use in flattening the flat_keys
        """
        if ax is None:
            fig, ax = plt.subplots()
        xidx = self.keys.index(xkey)
        yidx = self.keys.index(ykey)
        flat_idxs = [self.keys.index(k) for k in flat_keys]

        x = np.unique(self.data[:, xidx])
        if x0:
            x = x-x0
        y = np.unique(self.data[:, yidx])
        if y0:
            y = y-y0
        flat_vals = [np.unique(self.data[:, j]) for j in flat_idxs]
        z = self.data[:, -1]

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

        pax = ax.pcolormesh(X, Y, Z, cmap=cm, vmin=vmin, vmax=vmax)
        if colorbar:
            cb = plt.colorbar(pax, ax=ax, **cbarkwargs)
            cb.set_label('$2\mathcal{F}$')

        if add_mismatch:
            self.add_mismatch_to_ax(ax, x, y, xkey, ykey, *add_mismatch)

        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])
        if x0:
            ax.set_xlabel(self.tex_labels[xkey]+self.tex_labels0[xkey])
        else:
            ax.set_xlabel(self.tex_labels[xkey])
        if y0:
            ax.set_ylabel(self.tex_labels[ykey]+self.tex_labels0[ykey])
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
            fig.savefig('{}/{}_2D.png'.format(self.outdir, self.label))
        else:
            return ax

    def get_max_twoF(self):
        twoF = self.data[:, -1]
        idx = np.argmax(twoF)
        v = self.data[idx, :]
        d = OrderedDict(minStartTime=v[0], maxStartTime=v[1], F0=v[2], F1=v[3],
                        F2=v[4], Alpha=v[5], Delta=v[6], twoF=v[7])
        return d

    def print_max_twoF(self):
        d = self.get_max_twoF()
        print('Max twoF values for {}:'.format(self.label))
        for k, v in d.iteritems():
            print('  {}={}'.format(k, v))

    def set_out_file(self, extra_label=None):
        if self.detectors:
            dets = self.detectors.replace(',', '')
        else:
            dets = 'NA'
        if extra_label:
            self.out_file = '{}/{}_{}_{}_{}.txt'.format(
                self.outdir, self.label, dets, type(self).__name__,
                extra_label)
        else:
            self.out_file = '{}/{}_{}_{}.txt'.format(
                self.outdir, self.label, dets, type(self).__name__)


class SliceGridSearch(GridSearch):
    """ Slice gridded search using ComputeFstat """
    @helper_functions.initializer
    def __init__(self, label, outdir, sftfilepattern, F0s, F1s, F2s, Alphas,
                 Deltas, tref=None, minStartTime=None, maxStartTime=None,
                 nsegs=1, BSGL=False, minCoverFreq=None, maxCoverFreq=None,
                 detectors=None, SSBprec=None, injectSources=None,
                 input_arrays=False, assumeSqrtSX=None, Lambda0=None):
        """
        Parameters
        ----------
        label, outdir: str
            A label and directory to read/write data from/to
        sftfilepattern: str
            Pattern to match SFTs using wildcards (*?) and ranges [0-9];
            mutiple patterns can be given separated by colons.
        F0s, F1s, F2s, delta_F0s, delta_F1s, tglitchs, Alphas, Deltas: tuple
            Length 3 tuple describing the grid for each parameter, e.g
            [F0min, F0max, dF0], for a fixed value simply give [F0]. Unless
            input_arrays == True, then these are the values to search at.
        tref, minStartTime, maxStartTime: int
            GPS seconds of the reference time, start time and end time
        input_arrays: bool
            if true, use the F0s, F1s, etc as is

        For all other parameters, see `pyfstat.ComputeFStat` for details
        """

        if os.path.isdir(outdir) is False:
            os.mkdir(outdir)
        self.set_out_file()
        self.keys = ['_', '_', 'F0', 'F1', 'F2', 'Alpha', 'Delta']
        self.ndim = 0
        self.thetas = [F0s, F1s, Alphas, Deltas]
        self.ndim = 4

        self.search_keys = ['F0', 'F1', 'Alpha', 'Delta']
        self.Lambda0 = np.array(Lambda0)
        if len(self.Lambda0) != len(self.search_keys):
            raise ValueError(
                'Lambda0 must be of length {}'.format(len(self.search_keys)))

    def run(self, factor=2, max_n_ticks=4, whspace=0.07, save=True,
            **kwargs):
        lbdim = 0.5 * factor   # size of left/bottom margin
        trdim = 0.4 * factor   # size of top/right margin
        plotdim = factor * self.ndim + factor * (self.ndim - 1.) * whspace
        dim = lbdim + plotdim + trdim

        fig, axes = plt.subplots(self.ndim, self.ndim, figsize=(dim, dim))

        # Format the figure.
        lb = lbdim / dim
        tr = (lbdim + plotdim) / dim
        fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                            wspace=whspace, hspace=whspace)

        search = GridSearch(
            self.label, self.outdir, self.sftfilepattern,
            F0s=self.Lambda0[0], F1s=self.Lambda0[1], F2s=self.F2s[0],
            Alphas=self.Lambda0[2], Deltas=self.Lambda0[3], tref=self.tref,
            minStartTime=self.minStartTime, maxStartTime=self.maxStartTime)

        for i, ikey in enumerate(self.search_keys):
            setattr(search, ikey+'s', self.thetas[i])
            search.label = '{}_{}'.format(self.label, ikey)
            search.set_out_file()
            search.run()
            axes[i, i] = search.plot_1D(ikey, ax=axes[i, i], savefig=False,
                                        x0=self.Lambda0[i]
                                        )
            setattr(search, ikey+'s', [self.Lambda0[i]])
            axes[i, i].yaxis.tick_right()
            axes[i, i].yaxis.set_label_position("right")
            axes[i, i].set_xlabel('')

            for j, jkey in enumerate(self.search_keys):
                ax = axes[i, j]

                if j > i:
                    ax.set_frame_on(False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                ax.get_shared_x_axes().join(axes[self.ndim-1, j], ax)
                if i < self.ndim - 1:
                    ax.set_xticklabels([])
                if j < i:
                    ax.get_shared_y_axes().join(axes[i, i-1], ax)
                    if j > 0:
                        ax.set_yticklabels([])
                if j == i:
                    continue

                ax.xaxis.set_major_locator(
                    matplotlib.ticker.MaxNLocator(max_n_ticks, prune="upper"))
                ax.yaxis.set_major_locator(
                    matplotlib.ticker.MaxNLocator(max_n_ticks, prune="upper"))

                setattr(search, ikey+'s', self.thetas[i])
                setattr(search, jkey+'s', self.thetas[j])
                search.label = '{}_{}'.format(self.label, ikey+jkey)
                search.set_out_file()
                search.run()
                ax = search.plot_2D(jkey, ikey, ax=ax, save=False,
                                    y0=self.Lambda0[i], x0=self.Lambda0[j],
                                    **kwargs)
                setattr(search, ikey+'s', [self.Lambda0[i]])
                setattr(search, jkey+'s', [self.Lambda0[j]])

                ax.grid(lw=0.2, ls='--', zorder=10)
                ax.set_xlabel('')
                ax.set_ylabel('')

        for i, ikey in enumerate(self.search_keys):
            axes[-1, i].set_xlabel(
                self.tex_labels[ikey]+self.tex_labels0[ikey])
            if i > 0:
                axes[i, 0].set_ylabel(
                    self.tex_labels[ikey]+self.tex_labels0[ikey])
            axes[i, i].set_ylabel("$2\mathcal{F}$")

        if save:
            fig.savefig(
                '{}/{}_slice_projection.png'.format(self.outdir, self.label))
        else:
            return fig, axes


class GridUniformPriorSearch():
    @helper_functions.initializer
    def __init__(self, theta_prior, NF0, NF1, label, outdir, sftfilepattern,
                 tref, minStartTime, maxStartTime, minCoverFreq=None,
                 maxCoverFreq=None, BSGL=False, detectors=None, nsegs=1,
                 SSBprec=None, injectSources=None):
        dF0 = (theta_prior['F0']['upper'] - theta_prior['F0']['lower'])/NF0
        dF1 = (theta_prior['F1']['upper'] - theta_prior['F1']['lower'])/NF1
        F0s = [theta_prior['F0']['lower'], theta_prior['F0']['upper'], dF0]
        F1s = [theta_prior['F1']['lower'], theta_prior['F1']['upper'], dF1]
        self.search = GridSearch(
            label, outdir, sftfilepattern, F0s=F0s, F1s=F1s, tref=tref,
            Alphas=[theta_prior['Alpha']], Deltas=[theta_prior['Delta']],
            minStartTime=minStartTime, maxStartTime=maxStartTime, BSGL=BSGL,
            detectors=detectors, minCoverFreq=minCoverFreq,
            injectSources=injectSources, maxCoverFreq=maxCoverFreq,
            nsegs=nsegs, SSBprec=SSBprec)

    def run(self):
        self.search.run()

    def get_2D_plot(self, **kwargs):
        return self.search.plot_2D('F0', 'F1', **kwargs)


class GridGlitchSearch(GridSearch):
    """ Grid search using the SemiCoherentGlitchSearch """
    @helper_functions.initializer
    def __init__(self, label, outdir, sftfilepattern=None, F0s=[0],
                 F1s=[0], F2s=[0], delta_F0s=[0], delta_F1s=[0], tglitchs=None,
                 Alphas=[0], Deltas=[0], tref=None, minStartTime=None,
                 maxStartTime=None, minCoverFreq=None, maxCoverFreq=None,
                 write_after=1000):

        """
        Parameters
        ----------
        label, outdir: str
            A label and directory to read/write data from/to
        sftfilepattern: str
            Pattern to match SFTs using wildcards (*?) and ranges [0-9];
            mutiple patterns can be given separated by colons.
        F0s, F1s, F2s, delta_F0s, delta_F1s, tglitchs, Alphas, Deltas: tuple
            Length 3 tuple describing the grid for each parameter, e.g
            [F0min, F0max, dF0], for a fixed value simply give [F0].
        tref, minStartTime, maxStartTime: int
            GPS seconds of the reference time, start time and end time

        For all other parameters, see pyfstat.ComputeFStat.
        """
        if tglitchs is None:
            self.tglitchs = [self.maxStartTime]

        self.search = SemiCoherentGlitchSearch(
            label=label, outdir=outdir, sftfilepattern=self.sftfilepattern,
            tref=tref, minStartTime=minStartTime, maxStartTime=maxStartTime,
            minCoverFreq=minCoverFreq, maxCoverFreq=maxCoverFreq,
            BSGL=self.BSGL)

        if os.path.isdir(outdir) is False:
            os.mkdir(outdir)
        self.set_out_file()
        self.keys = ['F0', 'F1', 'F2', 'Alpha', 'Delta', 'delta_F0',
                     'delta_F1', 'tglitch']

    def get_input_data_array(self):
        arrays = []
        for tup in (self.F0s, self.F1s, self.F2s, self.Alphas, self.Deltas,
                    self.delta_F0s, self.delta_F1s, self.tglitchs):
            arrays.append(self.get_array_from_tuple(tup))

        input_data = []
        for vals in itertools.product(*arrays):
            input_data.append(vals)

        self.arrays = arrays
        self.input_data = np.array(input_data)


class FrequencySlidingWindow(GridSearch):
    """ A sliding-window search over the Frequency """
    @helper_functions.initializer
    def __init__(self, label, outdir, sftfilepattern, F0s, F1, F2,
                 Alpha, Delta, tref, minStartTime=None,
                 maxStartTime=None, window_size=10*86400, window_delta=86400,
                 BSGL=False, minCoverFreq=None, maxCoverFreq=None,
                 detectors=None, SSBprec=None, injectSources=None):
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

        if os.path.isdir(outdir) is False:
            os.mkdir(outdir)
        self.set_out_file()
        self.nsegs = 1
        self.F1s = [F1]
        self.F2s = [F2]
        self.Alphas = [Alpha]
        self.Deltas = [Delta]
        self.input_arrays = False

    def inititate_search_object(self):
        logging.info('Setting up search object')
        self.search = ComputeFstat(
            tref=self.tref, sftfilepattern=self.sftfilepattern,
            minCoverFreq=self.minCoverFreq, maxCoverFreq=self.maxCoverFreq,
            detectors=self.detectors, transientWindowType=self.transientWindowType,
            minStartTime=self.minStartTime, maxStartTime=self.maxStartTime,
            BSGL=self.BSGL, SSBprec=self.SSBprec,
            injectSources=self.injectSources)
        self.search.get_det_stat = (
            self.search.get_fullycoherent_twoF)

    def get_input_data_array(self):
        arrays = []
        tstarts = [self.minStartTime]
        while tstarts[-1] + self.window_size < self.maxStartTime:
            tstarts.append(tstarts[-1]+self.window_delta)
        arrays = [tstarts]
        for tup in (self.F0s, self.F1s, self.F2s,
                    self.Alphas, self.Deltas):
            arrays.append(self.get_array_from_tuple(tup))

        input_data = []
        for vals in itertools.product(*arrays):
            input_data.append(vals)

        input_data = np.array(input_data)
        input_data = np.insert(
            input_data, 1, input_data[:, 0] + self.window_size, axis=1)

        self.arrays = arrays
        self.input_data = np.array(input_data)

    def plot_sliding_window(self, F0=None, ax=None, savefig=True,
                            colorbar=True, timestamps=False):
        data = self.data
        if ax is None:
            ax = plt.subplot()
        tstarts = np.unique(data[:, 0])
        tends = np.unique(data[:, 1])
        frequencies = np.unique(data[:, 2])
        twoF = data[:, -1]
        tmids = (tstarts + tends) / 2.0
        dts = (tmids - self.minStartTime) / 86400.
        if F0:
            frequencies = frequencies - F0
            ax.set_ylabel('Frequency - $f_0$ [Hz] \n $f_0={:0.2f}$'.format(F0))
        else:
            ax.set_ylabel('Frequency [Hz]')
        twoF = twoF.reshape((len(tmids), len(frequencies)))
        Y, X = np.meshgrid(frequencies, dts)
        pax = ax.pcolormesh(X, Y, twoF)
        if colorbar:
            cb = plt.colorbar(pax, ax=ax)
            cb.set_label('$2\mathcal{F}$')
        ax.set_xlabel(
            r'Mid-point (days after $t_\mathrm{{start}}$={})'.format(
                self.minStartTime))
        ax.set_title(
            'Sliding window length = {} days in increments of {} days'
            .format(self.window_size/86400, self.window_delta/86400),
            )
        if timestamps:
            axT = ax.twiny()
            axT.set_xlim(tmids[0]*1e-9, tmids[-1]*1e-9)
            axT.set_xlabel('Mid-point timestamp [GPS $10^{9}$ s]')
            ax.set_title(ax.get_title(), y=1.18)
        if savefig:
            plt.tight_layout()
            plt.savefig(
                '{}/{}_sliding_window.png'.format(self.outdir, self.label))
        else:
            return ax


class EarthTest(GridSearch):
    """ """
    tex_labels = {'deltaRadius': '$\Delta R$ [m]',
                  'phaseOffset': 'phase-offset [rad]',
                  'deltaPspin': '$\Delta P_\mathrm{spin}$ [s]'}

    @helper_functions.initializer
    def __init__(self, label, outdir, sftfilepattern, deltaRadius,
                 phaseOffset, deltaPspin, F0, F1, F2, Alpha,
                 Delta, tref=None, minStartTime=None, maxStartTime=None,
                 BSGL=False, minCoverFreq=None, maxCoverFreq=None,
                 detectors=None, injectSources=None,
                 assumeSqrtSX=None):
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
        self.phaseOffset = self.phaseOffset + 1e-12  # Hack to stop cached data being used
        self.deltaPspin = np.atleast_1d(deltaPspin)
        self.set_out_file()
        self.SSBprec = lalpulsar.SSBPREC_RELATIVISTIC
        self.keys = ['deltaRadius', 'phaseOffset', 'deltaPspin']

        self.prior_widths = [
            np.max(self.deltaRadius)-np.min(self.deltaRadius),
            np.max(self.phaseOffset)-np.min(self.phaseOffset),
            np.max(self.deltaPspin)-np.min(self.deltaPspin)]

        if hasattr(self, 'search') is False:
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
        vals = [self.minStartTime, self.maxStartTime, self.F0, self.F1,
                self.F2, self.Alpha, self.Delta]
        self.special_data = {'zero': [0, 0, 0]}
        for key, (dR, dphi, dP) in self.special_data.iteritems():
            rescaleRadius = (1 + dR / lal.REARTH_SI)
            rescalePeriod = (1 + dP / lal.DAYSID_SI)
            lalpulsar.BarycenterModifyEarthRotation(
                rescaleRadius, dphi, rescalePeriod, self.tref)
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
        vals = [self.minStartTime, self.maxStartTime, self.F0, self.F1,
                self.F2, self.Alpha, self.Delta]
        for (dR, dphi, dP) in tqdm(self.input_data):
            rescaleRadius = (1 + dR / lal.REARTH_SI)
            rescalePeriod = (1 + dP / lal.DAYSID_SI)
            lalpulsar.BarycenterModifyEarthRotation(
                rescaleRadius, dphi, rescalePeriod, self.tref)
            FS = self.search.get_det_stat(*vals)
            data.append(list([dR, dphi, dP]) + [FS])

        data = np.array(data, dtype=np.float)
        logging.info('Saving data to {}'.format(self.out_file))
        np.savetxt(self.out_file, data, delimiter=' ')
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
                F = logsumexp(F, axis=-1)+np.log(dx)-np.log(prior_widths[-1-i])
            else:
                F = np.squeeze(F, axis=-1)
        marginalised_F = np.atleast_1d(F)[0]
        F_at_zero = self.special_data['zero'][-1]/2.0

        max_idx = np.argmax(self.data[:, -1])
        max_F = self.data[max_idx, -1]/2.0
        max_F_params = self.data[max_idx, :-1]
        logging.info('F at zero = {:.1f}, marginalised_F = {:.1f},'
                     ' max_F = {:.1f} ({})'.format(
                         F_at_zero, marginalised_F, max_F, max_F_params))
        return F_at_zero - marginalised_F, (F_at_zero - max_F) / F_at_zero

    def plot_corner(self, prior_widths=None, fig=None, axes=None,
                    projection='log_mean'):
        Bsa, FmaxMismatch = self.marginalised_bayes_factor(prior_widths)

        data = self.data[:, -1].reshape(
            (len(self.deltaRadius), len(self.phaseOffset),
             len(self.deltaPspin)))
        xyz = [self.deltaRadius/lal.REARTH_SI, self.phaseOffset/(np.pi),
               self.deltaPspin/60.]
        labels = [r'$\frac{\Delta R}{R_\mathrm{Earth}}$',
                  r'$\frac{\Delta \phi}{\pi}$',
                  r'$\Delta P_\mathrm{spin}$ [min]',
                  r'$2\mathcal{F}$']

        try:
            from gridcorner import gridcorner
        except ImportError:
            raise ImportError(
                "Python module 'gridcorner' not found, please install from "
                "https://gitlab.aei.uni-hannover.de/GregAshton/gridcorner")

        fig, axes = gridcorner(data, xyz, projection=projection, factor=1.6,
                               labels=labels)
        axes[-1][-1].axvline((lal.DAYJUL_SI - lal.DAYSID_SI)/60.0, color='C3')
        plt.suptitle(
            'T={:.1f} days, $f$={:.2f} Hz, $\log\mathcal{{B}}_{{S/A}}$={:.1f},'
            r' $\frac{{\mathcal{{F}}_0-\mathcal{{F}}_\mathrm{{max}}}}'
            r'{{\mathcal{{F}}_0}}={:.1e}$'
            .format(self.duration/86400, self.F0, Bsa, FmaxMismatch), y=0.99,
            size=14)
        fig.savefig('{}/{}_projection_matrix.png'.format(
            self.outdir, self.label))

    def plot(self, key, prior_widths=None):
        Bsa, FmaxMismatch = self.marginalised_bayes_factor(prior_widths)

        rescales_defaults = {'deltaRadius': 1/lal.REARTH_SI,
                             'phaseOffset': 1/np.pi,
                             'deltaPspin': 1}
        labels = {'deltaRadius': r'$\frac{\Delta R}{R_\mathrm{Earth}}$',
                  'phaseOffset': r'$\frac{\Delta \phi}{\pi}$',
                  'deltaPspin': r'$\Delta P_\mathrm{spin}$ [s]'
                  }

        fig, ax = self.plot_1D(key, xrescale=rescales_defaults[key],
                               xlabel=labels[key], savefig=False)
        ax.set_title(
            'T={} days, $f$={} Hz, $\log\mathcal{{B}}_{{S/A}}$={:.1f}'
            .format(self.duration/86400, self.F0, Bsa))
        fig.tight_layout()
        fig.savefig('{}/{}_1D.png'.format(self.outdir, self.label))


class DMoff_NO_SPIN(GridSearch):
    """ DMoff test using SSBPREC_NO_SPIN """
    @helper_functions.initializer
    def __init__(self, par, label, outdir, sftfilepattern, minStartTime=None,
                 maxStartTime=None, minCoverFreq=None, maxCoverFreq=None,
                 detectors=None, injectSources=None, assumeSqrtSX=None):
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

        if os.path.isdir(outdir) is False:
            os.mkdir(outdir)

        if type(par) == dict:
            self.par = par
        elif type(par) == str and os.path.isfile(par):
            self.par = read_par(filename=par)
        else:
            raise ValueError('The .par file does not exist')

        self.nsegs = 1
        self.BSGL = False

        self.tref = self.par['tref']
        self.F1s = [self.par.get('F1', 0)]
        self.F2s = [self.par.get('F2', 0)]
        self.Alphas = [self.par['Alpha']]
        self.Deltas = [self.par['Delta']]
        self.Re = 6.371e6
        self.c = 2.998e8
        a0 = self.Re/self.c  # *np.cos(self.par['Delta'])
        self.m0 = np.max([4, int(np.ceil(2*np.pi*self.par['F0']*a0))])
        logging.info(
            'Setting up DMoff_NO_SPIN search with m0 = {}'.format(self.m0))

    def get_results(self):
        """ Compute the three summed detection statistics

        Returns
        -------
            m0, twoF_SUM, twoFstar_SUM_SIDEREAL, twoFstar_SUM_TERRESTRIAL

        """
        self.SSBprec = lalpulsar.SSBPREC_RELATIVISTIC
        self.set_out_file('SSBPREC_RELATIVISTIC')
        self.F0s = [self.par['F0']+j/lal.DAYSID_SI for j in range(-4, 5)]
        self.run()
        twoF_SUM = np.sum(self.data[:, -1])

        self.SSBprec = lalpulsar.SSBPREC_NO_SPIN
        self.set_out_file('SSBPREC_NO_SPIN')
        self.F0s = [self.par['F0']+j/lal.DAYSID_SI
                    for j in range(-self.m0, self.m0+1)]
        self.run()
        twoFstar_SUM = np.sum(self.data[:, -1])

        self.set_out_file('SSBPREC_NO_SPIN_TERRESTRIAL')
        self.F0s = [self.par['F0']+j/lal.DAYJUL_SI
                    for j in range(-self.m0, self.m0+1)]
        self.run()
        twoFstar_SUM_terrestrial = np.sum(self.data[:, -1])

        return self.m0, twoF_SUM, twoFstar_SUM, twoFstar_SUM_terrestrial
