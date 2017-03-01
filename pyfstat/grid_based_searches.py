""" Searches using grid-based methods """

import os
import logging
import itertools
from collections import OrderedDict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import helper_functions
from core import BaseSearchClass, ComputeFstat, SemiCoherentGlitchSearch, SemiCoherentSearch
from core import tqdm, args, earth_ephem, sun_ephem


class GridSearch(BaseSearchClass):
    """ Gridded search using ComputeFstat """
    @helper_functions.initializer
    def __init__(self, label, outdir, sftfilepath, F0s=[0], F1s=[0], F2s=[0],
                 Alphas=[0], Deltas=[0], tref=None, minStartTime=None, nsegs=1,
                 maxStartTime=None, BSGL=False, minCoverFreq=None,
                 maxCoverFreq=None, earth_ephem=None, sun_ephem=None,
                 detectors=None):
        """
        Parameters
        ----------
        label, outdir: str
            A label and directory to read/write data from/to
        sftfilepath: str
            File patern to match SFTs
        F0s, F1s, F2s, delta_F0s, delta_F1s, tglitchs, Alphas, Deltas: tuple
            Length 3 tuple describing the grid for each parameter, e.g
            [F0min, F0max, dF0], for a fixed value simply give [F0].
        tref, minStartTime, maxStartTime: int
            GPS seconds of the reference time, start time and end time

        For all other parameters, see `pyfstat.ComputeFStat` for details
        """

        if earth_ephem is None:
            self.earth_ephem = self.earth_ephem_default
        if sun_ephem is None:
            self.sun_ephem = self.sun_ephem_default

        if os.path.isdir(outdir) is False:
            os.mkdir(outdir)
        self.out_file = '{}/{}_gridFS.txt'.format(self.outdir, self.label)
        self.keys = ['_', '_', 'F0', 'F1', 'F2', 'Alpha', 'Delta']

    def inititate_search_object(self):
        logging.info('Setting up search object')
        if self.nsegs == 1:
            self.search = ComputeFstat(
                tref=self.tref, sftfilepath=self.sftfilepath,
                minCoverFreq=self.minCoverFreq, maxCoverFreq=self.maxCoverFreq,
                earth_ephem=self.earth_ephem, sun_ephem=self.sun_ephem,
                detectors=self.detectors, transient=False,
                minStartTime=self.minStartTime, maxStartTime=self.maxStartTime,
                BSGL=self.BSGL)
            self.search.get_det_stat = self.search.run_computefstatistic_single_point
        else:
            self.search = SemiCoherentSearch(
                label=self.label, outdir=self.outdir, tref=self.tref,
                nsegs=self.nsegs, sftfilepath=self.sftfilepath,
                BSGL=self.BSGL, minStartTime=self.minStartTime,
                maxStartTime=self.maxStartTime, minCoverFreq=self.minCoverFreq,
                maxCoverFreq=self.maxCoverFreq, detectors=self.detectors,
                earth_ephem=self.earth_ephem, sun_ephem=self.sun_ephem)

            def cut_out_tstart_tend(*vals):
                return self.search.run_semi_coherent_computefstatistic_single_point(*vals[2:])
            self.search.get_det_stat = cut_out_tstart_tend

    def get_array_from_tuple(self, x):
        if len(x) == 1:
            return np.array(x)
        else:
            return np.arange(x[0], x[1]*(1+1e-15), x[2])

    def get_input_data_array(self):
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

        data = np.array(data)
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
                predicted_twoF=None, cm=None, cbarkwargs={}):
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
        y = np.unique(self.data[:, yidx])
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
        ax.set_xlabel(labels[xkey])
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


class GridUniformPriorSearch():
    def __init__(self, theta_prior, NF0, NF1, label, outdir, sftfilepath,
                 tref, minStartTime, maxStartTime, minCoverFreq=None,
                 maxCoverFreq=None, BSGL=False, detectors=None, nsegs=1):
        dF0 = (theta_prior['F0']['upper'] - theta_prior['F0']['lower'])/NF0
        dF1 = (theta_prior['F1']['upper'] - theta_prior['F1']['lower'])/NF1
        F0s = [theta_prior['F0']['lower'], theta_prior['F0']['upper'], dF0]
        F1s = [theta_prior['F1']['lower'], theta_prior['F1']['upper'], dF1]
        self.search = GridSearch(
            label, outdir, sftfilepath, F0s=F0s, F1s=F1s, tref=tref,
            Alphas=[theta_prior['Alpha']], Deltas=[theta_prior['Delta']],
            minStartTime=minStartTime, maxStartTime=maxStartTime, BSGL=BSGL,
            detectors=detectors, minCoverFreq=minCoverFreq,
            maxCoverFreq=maxCoverFreq, nsegs=nsegs)

    def run(self, **kwargs):
        self.search.run()
        return self.search.plot_2D('F0', 'F1', **kwargs)


class GridGlitchSearch(GridSearch):
    """ Grid search using the SemiCoherentGlitchSearch """
    @helper_functions.initializer
    def __init__(self, label, outdir, sftfilepath=None, F0s=[0],
                 F1s=[0], F2s=[0], delta_F0s=[0], delta_F1s=[0], tglitchs=None,
                 Alphas=[0], Deltas=[0], tref=None, minStartTime=None,
                 maxStartTime=None, minCoverFreq=None, maxCoverFreq=None,
                 write_after=1000, earth_ephem=None, sun_ephem=None):

        """
        Parameters
        ----------
        label, outdir: str
            A label and directory to read/write data from/to
        sftfilepath: str
            File patern to match SFTs
        F0s, F1s, F2s, delta_F0s, delta_F1s, tglitchs, Alphas, Deltas: tuple
            Length 3 tuple describing the grid for each parameter, e.g
            [F0min, F0max, dF0], for a fixed value simply give [F0].
        tref, minStartTime, maxStartTime: int
            GPS seconds of the reference time, start time and end time

        For all other parameters, see pyfstat.ComputeFStat.
        """
        if tglitchs is None:
            self.tglitchs = [self.maxStartTime]
        if earth_ephem is None:
            self.earth_ephem = self.earth_ephem_default
        if sun_ephem is None:
            self.sun_ephem = self.sun_ephem_default

        self.search = SemiCoherentGlitchSearch(
            label=label, outdir=outdir, sftfilepath=self.sftfilepath,
            tref=tref, minStartTime=minStartTime, maxStartTime=maxStartTime,
            minCoverFreq=minCoverFreq, maxCoverFreq=maxCoverFreq,
            earth_ephem=self.earth_ephem, sun_ephem=self.sun_ephem,
            BSGL=self.BSGL)

        if os.path.isdir(outdir) is False:
            os.mkdir(outdir)
        self.out_file = '{}/{}_gridFS.txt'.format(self.outdir, self.label)
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



