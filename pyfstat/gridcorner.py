""" A corner plotting tool for an array (grid) of dependent values.

Given an N-dimensional set of data (i.e. some function evaluated over a grid
of coordinates), plot all possible 1D and 2D projections in the style of a
'corner' plot.

This code has been copied from Gregory Ashton's repository
https://gitlab.aei.uni-hannover.de/GregAshton/gridcorner
and it uses both the central idea and some specific code from
Daniel Foreman-Mackey's
https://github.com/dfm/corner.py
re-used under the following licence requirements:

Copyright (c) 2013-2020 Daniel Foreman-Mackey

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the FreeBSD Project.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp


def log_mean(loga, axis):
    """Calculate the log(<a>) mean

    Given `N` logged value `log`, calculate the log_mean
    `log(<loga>)=log(sum(np.exp(loga))) - log(N)`. Useful for marginalizing
    over logged likelihoods for example.

    Parameters
    ----------
    loga: array_like
        Input_array.
    axies: None or int or type of ints, optional
        Axis or axes over which the sum is taken. By default axis is None, and
        all elements are summed.
    Returns
    -------
    log_mean: ndarry
        The logged average value (shape loga.shape)
    """
    loga = np.array(loga)
    N = np.prod([loga.shape[i] for i in axis])
    return logsumexp(loga, axis) - np.log(N)


def max_slice(D, axis):
    """Return the slice along the given axis"""
    idxs = [range(D.shape[j]) for j in range(D.ndim)]
    max_idx = list(np.unravel_index(D.argmax(), D.shape))
    for k in np.atleast_1d(axis):
        idxs[k] = [max_idx[k]]
    res = np.squeeze(D[np.ix_(*tuple(idxs))])
    return res


def idx_array_slice(D, axis, slice_idx):
    """Return the slice along the given axis"""
    idxs = [range(D.shape[j]) for j in range(D.ndim)]
    for k in np.atleast_1d(axis):
        idxs[k] = [slice_idx[k]]
    res = np.squeeze(D[np.ix_(*tuple(idxs))])
    return res


def _get_fig_and_axes(ndim, factor, whspace):
    lbdim = 0.5 * factor  # size of left/bottom margin
    trdim = 0.2 * factor  # size of top/right margin
    plotdim = factor * ndim + factor * (ndim - 1.0) * whspace
    dim = lbdim + plotdim + trdim
    fig, axes = plt.subplots(ndim, ndim, figsize=(dim, dim))
    axes = np.atleast_2d(axes)  # allow single-parameter plots

    # Format the figure.
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(
        left=lb, bottom=lb, right=0.98 * tr, top=tr, wspace=whspace, hspace=whspace
    )
    return fig, axes


def gridcorner(
    D,
    xyz,
    labels=None,
    projection="max_slice",
    max_n_ticks=4,
    factor=2,
    whspace=0.05,
    showDvals=True,
    lines=None,
    label_offset=0.4,
    **kwargs
):
    """Generate a grid corner plot

    Parameters
    ----------
    D: array_like
        N-dimensional data to plot, `D.shape` should be  `(n1, n2,..., nN)`,
        where `N`, is the number of grid points along dimension `i`.
    xyz: list
        List of 1-dimensional arrays of coordinates. `xyz[i]` should have
        length `N` (see help for `D`).
    labels: list
        N+1 length list of labels; the first N correspond to the coordinates
        labels, the final label is for the dependent (D) variable.
    projection: str or func
        If a string, one of `{"log_mean", "max_slice"} to use inbuilt functions
        to calculate either the logged mean or maximum slice projection. Else
        a function to use for projection, must take an `axis` argument. Default
        is `gridcorner.max_slice()`, to project out a slice along the
        maximum.
    max_n_ticks: int
        Number of ticks for x and y axis of the `pcolormesh` plots.
    factor: float
        Controls the size of one window.
    showDvals: bool
        If true (default) show the D values on the right-hand-side of the
        1D plots and add a label.
    lines: array_like
        N-dimensional list of values to delineate.

    Returns
    -------
    fig, axes:
        The figure and NxN set of axes

    """

    ndim = D.ndim
    fig, axes = _get_fig_and_axes(ndim, factor, whspace)

    if isinstance(projection, str):
        if projection in ["log_mean"]:
            projection = log_mean
        elif projection in ["max_slice"]:
            projection = max_slice
        else:
            raise ValueError("Projection {} not understood".format(projection))

    for i in range(ndim):
        projection_1D(
            axes[i, i],
            xyz[i],
            D,
            i,
            projection=projection,
            showDvals=showDvals,
            lines=lines,
            **kwargs
        )
        for j in range(ndim):
            ax = axes[i, j]

            if j > i:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            ax.sharex(axes[ndim - 1, j])
            if i < ndim - 1:
                ax.tick_params(labelbottom=False)
            if j < i:
                ax.sharey(axes[i, i - 1])
                if j > 0:
                    ax.tick_params(labelleft=False)
            if j == i:
                continue

            ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="upper"))
            ax.yaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="upper"))

            ax, pax = projection_2D(
                ax,
                xyz[i],
                xyz[j],
                D,
                i,
                j,
                lines=lines,
                projection=projection,
                **kwargs
            )

    if labels:
        for i in range(ndim):
            axes[-1, i].set_xlabel(labels[i])
            if i > 0:
                axes[i, 0].set_ylabel(labels[i])
            if showDvals:
                axes[i, i].set_ylabel(labels[-1])

            for ax in axes[:, 0]:
                ax.yaxis.set_label_coords(-label_offset, 0.5)
                for ax in axes[-1, :]:
                    ax.xaxis.set_label_coords(0.5, -label_offset)
    return fig, axes


def projection_2D(ax, x, y, D, xidx, yidx, projection, lines=None, **kwargs):
    flat_idxs = list(range(D.ndim))
    flat_idxs.remove(xidx)
    flat_idxs.remove(yidx)
    D2D = projection(D, axis=tuple(flat_idxs), **kwargs)
    X, Y = np.meshgrid(x, y, indexing="ij")
    pax = ax.pcolormesh(Y, X, D2D.T, vmin=D.min(), vmax=D.max(), shading="auto")
    if lines:
        ax.axhline(lines[xidx], lw=0.5, color="w")
        ax.axvline(lines[yidx], lw=0.5, color="w")
    return ax, pax


def projection_1D(ax, x, D, xidx, projection, showDvals=True, lines=None, **kwargs):
    flat_idxs = list(range(D.ndim))
    flat_idxs.remove(xidx)
    D1D = projection(D, axis=tuple(flat_idxs), **kwargs)
    ax.plot(x, D1D, color="k")
    if showDvals:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    else:
        ax.yaxis.set_ticklabels([])
    if lines:
        ax.axvline(lines[xidx], lw=0.5, color="C0")
    return ax
