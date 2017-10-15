""" Searches using grid-based methods """
from __future__ import division, absolute_import, print_function

import os
import logging
import itertools
from collections import OrderedDict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import pyfstat.helper_functions as helper_functions
from pyfstat.core import (BaseSearchClass, ComputeFstat,
                          SemiCoherentGlitchSearch, SemiCoherentSearch, tqdm,
                          args, read_par)
import lalpulsar
import lal


class GridSearch(BaseSearchClass):
    """ Gridded search using ComputeFstat """
    @helper_functions.initializer
    def __init__(self, label, outdir, sftfilepattern, F0s=[0], F1s=[0], F2s=[0],
                 Alphas=[0], Deltas=[0], tref=None, minStartTime=None,
                 maxStartTime=None, nsegs=1, BSGL=False, minCoverFreq=None,
                 maxCoverFreq=None, detectors=None, SSBprec=None,
                 injectSources=None, input_arrays=False, assumeSqrtSX=None):
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

    def inititate_search_object(self):
        logging.info('Setting up search object')
        if self.nsegs == 1:
            self.search = ComputeFstat(
                tref=self.tref, sftfilepattern=self.sftfilepattern,
                minCoverFreq=self.minCoverFreq, maxCoverFreq=self.maxCoverFreq,
                detectors=self.detectors, transient=False,
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
        arrays = []
        for tup in ([self.minStartTime], [self.maxStartTime], self.F0s, self.F1s, self.F2s,
                    self.Alphas, self.Deltas):
            arrays.append(self.get_array_from_tuple(tup))

        input_data = []
        for vals in itertools.product(*arrays):
            input_data.append(vals)

        self.arrays = arrays
        self.input_data = np.array(input_data)

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

    def run(self, return_data=False):
        self.get_input_data_array()
        old_data = self.check_old_data_is_okay_to_use()
        if old_data is not False:
            self.data = old_data
            return

        self.inititate_search_object()

        logging.info('Total number of grid points is {}'.format(
            len(self.input_data)))

        data = []
        for vals in tqdm(self.input_data):
            FS = self.search.get_det_stat(*vals)
            data.append(list(vals) + [FS])

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

    def plot_1D(self, xkey):
        fig, ax = plt.subplots()
        xidx = self.keys.index(xkey)
        x = np.unique(self.data[:, xidx])
        z = self.data[:, -1]
        plt.plot(x, z)
        fig.savefig('{}/{}_1D.png'.format(self.outdir, self.label))

    def plot_2D(self, xkey, ykey, ax=None, save=True, vmin=None, vmax=None,
                add_mismatch=None, xN=None, yN=None, flat_keys=[],
                rel_flat_idxs=[], flatten_method=np.max, title=None,
                predicted_twoF=None, cm=None, cbarkwargs={}, x0=None, y0=None):
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
        cb = plt.colorbar(pax, ax=ax, **cbarkwargs)
        cb.set_label('$2\mathcal{F}$')

        if add_mismatch:
            self.add_mismatch_to_ax(ax, x, y, xkey, ykey, *add_mismatch)

        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])
        labels = {'F0': '$f$', 'F1': '$\dot{f}$'}
        labels0 = {'F0': '$-f_0$', 'F1': '$-\dot{f}_0$'}
        if x0:
            ax.set_xlabel(labels[xkey]+labels0[xkey])
        else:
            ax.set_xlabel(labels[xkey])
        if y0:
            ax.set_ylabel(labels[ykey]+labels0[ykey])
        else:
            ax.set_ylabel(labels[ykey])

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
    def __init__(self, label, outdir, sftfilepattern, F0s=[0], F1s=[0], F2s=[0],
                 Alphas=[0], Deltas=[0], tref=None, minStartTime=None,
                 maxStartTime=None, nsegs=1, BSGL=False, minCoverFreq=None,
                 maxCoverFreq=None, detectors=None, SSBprec=None,
                 injectSources=None, input_arrays=False, assumeSqrtSX=None,
                 Lambda0=None):
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

        self.Lambda0 = np.array(Lambda0)
        if len(self.Lambda0) != len(self.keys):
            raise ValueError(
                'Lambda0 must be of length {}'.format(len(self.keys)))

    def run(self, return_data=False):
        self.get_input_data_array()

        self.Lambda0s_grid = []
        for j in range(self.input_data.shape[1]):
            i = np.argmin(np.abs(self.Lambda0[j]-self.input_data[:, j]))
            self.Lambda0s_grid.append(self.input_data[:, j][i])

        old_data = self.check_old_data_is_okay_to_use()
        if old_data is not False:
            self.data = old_data
            return

        self.inititate_search_object()

        logging.info('Total number of grid points is {}'.format(
            len(self.input_data)))

        data = []
        for vals in tqdm(self.input_data):
            if np.sum(vals != self.Lambda0s_grid) < 3:
                FS = self.search.get_det_stat(*vals)
                data.append(list(vals) + [FS])
            else:
                data.append(list(vals) + [0])

        data = np.array(data, dtype=np.float)
        if return_data:
            return data
        else:
            logging.info('Saving data to {}'.format(self.out_file))
            np.savetxt(self.out_file, data, delimiter=' ')
            self.data = data


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

    def inititate_search_object(self):
        logging.info('Setting up search object')
        self.search = ComputeFstat(
            tref=self.tref, sftfilepattern=self.sftfilepattern,
            minCoverFreq=self.minCoverFreq, maxCoverFreq=self.maxCoverFreq,
            detectors=self.detectors, transient=True,
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
