"""
MCMC search v.s. grid search
============================

An example to compare MCMCSearch and GridSearch on the same data.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import pyfstat

# flip this switch for a more expensive 4D (F0,F1,Alpha,Delta) run
# instead of just (F0,F1)
# (still only a few minutes on current laptops)
sky = False

label = "PyFstatExampleSimpleMCMCvsGridComparison"
outdir = os.path.join("PyFstat_example_data", label)
logger = pyfstat.set_up_logger(label=label, outdir=outdir)
if sky:
    outdir += "AlphaDelta"

# parameters for the data set to generate
tstart = 1000000000
duration = 30 * 86400
Tsft = 1800
detectors = "H1,L1"
sqrtSX = 1e-22

# parameters for injected signals
inj = {
    "tref": tstart,
    "F0": 30.0,
    "F1": -1e-10,
    "F2": 0,
    "Alpha": 0.5,
    "Delta": 1,
    "h0": 0.05 * sqrtSX,
    "cosi": 1.0,
}

# latex-formatted plotting labels
labels = {
    "F0": "$f$ [Hz]",
    "F1": "$\\dot{f}$ [Hz/s]",
    "2F": "$2\\mathcal{F}$",
    "Alpha": "$\\alpha$",
    "Delta": "$\\delta$",
}
labels["max2F"] = "$\\max\\,$" + labels["2F"]


def plot_grid_vs_samples(grid_res, mcmc_res, xkey, ykey):
    """local plotting function to avoid code duplication in the 4D case"""
    plt.plot(grid_res[xkey], grid_res[ykey], ".", label="grid")
    plt.plot(mcmc_res[xkey], mcmc_res[ykey], ".", label="mcmc")
    plt.plot(inj[xkey], inj[ykey], "*k", label="injection")
    grid_maxidx = np.argmax(grid_res["twoF"])
    mcmc_maxidx = np.argmax(mcmc_res["twoF"])
    plt.plot(
        grid_res[xkey][grid_maxidx],
        grid_res[ykey][grid_maxidx],
        "+g",
        label=labels["max2F"] + "(grid)",
    )
    plt.plot(
        mcmc_res[xkey][mcmc_maxidx],
        mcmc_res[ykey][mcmc_maxidx],
        "xm",
        label=labels["max2F"] + "(mcmc)",
    )
    plt.xlabel(labels[xkey])
    plt.ylabel(labels[ykey])
    plt.legend()
    plotfilename_base = os.path.join(outdir, "grid_vs_mcmc_{:s}{:s}".format(xkey, ykey))
    plt.savefig(plotfilename_base + ".png")
    if xkey == "F0" and ykey == "F1":
        plt.xlim(zoom[xkey])
        plt.ylim(zoom[ykey])
        plt.savefig(plotfilename_base + "_zoom.png")
    plt.close()


def plot_2F_scatter(res, label, xkey, ykey):
    """local plotting function to avoid code duplication in the 4D case"""
    markersize = 3 if label == "grid" else 1
    sc = plt.scatter(res[xkey], res[ykey], c=res["twoF"], s=markersize)
    cb = plt.colorbar(sc)
    plt.xlabel(labels[xkey])
    plt.ylabel(labels[ykey])
    cb.set_label(labels["2F"])
    plt.title(label)
    plt.plot(inj[xkey], inj[ykey], "*k", label="injection")
    maxidx = np.argmax(res["twoF"])
    plt.plot(
        res[xkey][maxidx],
        res[ykey][maxidx],
        "+r",
        label=labels["max2F"],
    )
    plt.legend()
    plotfilename_base = os.path.join(
        outdir, "{:s}_{:s}{:s}_2F".format(label, xkey, ykey)
    )
    plt.xlim([min(res[xkey]), max(res[xkey])])
    plt.ylim([min(res[ykey]), max(res[ykey])])
    plt.savefig(plotfilename_base + ".png")
    plt.close()


if __name__ == "__main__":
    logger.info("Generating SFTs with injected signal...")
    writer = pyfstat.Writer(
        label=label + "SimulatedSignal",
        outdir=outdir,
        tstart=tstart,
        duration=duration,
        detectors=detectors,
        sqrtSX=sqrtSX,
        Tsft=Tsft,
        **inj,
        Band=1,  # default band estimation would be too narrow for a wide grid/prior
    )
    writer.make_data()

    # set up square search grid with fixed (F0,F1) mismatch
    # and (optionally) some ad-hoc sky coverage
    m = 0.001
    dF0 = np.sqrt(12 * m) / (np.pi * duration)
    dF1 = np.sqrt(180 * m) / (np.pi * duration**2)
    DeltaF0 = 500 * dF0
    DeltaF1 = 200 * dF1
    if sky:
        # cover less range to keep runtime down
        DeltaF0 /= 10
        DeltaF1 /= 10
    F0s = [inj["F0"] - DeltaF0 / 2.0, inj["F0"] + DeltaF0 / 2.0, dF0]
    F1s = [inj["F1"] - DeltaF1 / 2.0, inj["F1"] + DeltaF1 / 2.0, dF1]
    F2s = [inj["F2"]]
    search_keys = ["F0", "F1"]  # only the ones that aren't 0-width
    if sky:
        dSky = 0.01  # rather coarse to keep runtime down
        DeltaSky = 10 * dSky
        Alphas = [inj["Alpha"] - DeltaSky / 2.0, inj["Alpha"] + DeltaSky / 2.0, dSky]
        Deltas = [inj["Delta"] - DeltaSky / 2.0, inj["Delta"] + DeltaSky / 2.0, dSky]
        search_keys += ["Alpha", "Delta"]
    else:
        Alphas = [inj["Alpha"]]
        Deltas = [inj["Delta"]]
    search_keys_label = "".join(search_keys)

    logger.info("Performing GridSearch...")
    gridsearch = pyfstat.GridSearch(
        label="GridSearch" + search_keys_label,
        outdir=outdir,
        sftfilepattern=writer.sftfilepath,
        F0s=F0s,
        F1s=F1s,
        F2s=F2s,
        Alphas=Alphas,
        Deltas=Deltas,
        tref=inj["tref"],
    )
    gridsearch.run()
    gridsearch.print_max_twoF()
    gridsearch.generate_loudest()

    # do some plots of the GridSearch results
    if not sky:  # this plotter can't currently deal with too large result arrays
        logger.info("Plotting 1D 2F distributions...")
        for key in search_keys:
            gridsearch.plot_1D(xkey=key, xlabel=labels[key], ylabel=labels["2F"])

    logger.info("Making GridSearch {:s} corner plot...".format("-".join(search_keys)))
    vals = [np.unique(gridsearch.data[key]) - inj[key] for key in search_keys]
    twoF = gridsearch.data["twoF"].reshape([len(kval) for kval in vals])
    corner_labels = [
        "$f - f_0$ [Hz]",
        "$\\dot{f} - \\dot{f}_0$ [Hz/s]",
    ]
    if sky:
        corner_labels.append("$\\alpha - \\alpha_0$")
        corner_labels.append("$\\delta - \\delta_0$")
    corner_labels.append(labels["2F"])
    gridcorner_fig, gridcorner_axes = pyfstat.gridcorner(
        twoF, vals, projection="log_mean", labels=corner_labels, whspace=0.1, factor=1.8
    )
    gridcorner_fig.savefig(os.path.join(outdir, gridsearch.label + "_corner.png"))
    plt.close(gridcorner_fig)

    logger.info("Performing MCMCSearch...")
    # set up priors in F0 and F1 (over)covering the grid ranges
    if sky:  # MCMC will still be fast in 4D with wider range than grid
        DeltaF0 *= 50
        DeltaF1 *= 50
    theta_prior = {
        "F0": {
            "type": "unif",
            "lower": inj["F0"] - DeltaF0 / 2.0,
            "upper": inj["F0"] + DeltaF0 / 2.0,
        },
        "F1": {
            "type": "unif",
            "lower": inj["F1"] - DeltaF1 / 2.0,
            "upper": inj["F1"] + DeltaF1 / 2.0,
        },
        "F2": inj["F2"],
    }
    if sky:
        # also implicitly covering twice the grid range here
        theta_prior["Alpha"] = {
            "type": "unif",
            "lower": inj["Alpha"] - DeltaSky,
            "upper": inj["Alpha"] + DeltaSky,
        }
        theta_prior["Delta"] = {
            "type": "unif",
            "lower": inj["Delta"] - DeltaSky,
            "upper": inj["Delta"] + DeltaSky,
        }
    else:
        theta_prior["Alpha"] = inj["Alpha"]
        theta_prior["Delta"] = inj["Delta"]
    # ptemcee sampler settings - in a real application we might want higher values
    ntemps = 2
    log10beta_min = -1
    nwalkers = 100
    nsteps = [200, 200]  # [burnin,production]

    mcmcsearch = pyfstat.MCMCSearch(
        label="MCMCSearch" + search_keys_label,
        outdir=outdir,
        sftfilepattern=writer.sftfilepath,
        theta_prior=theta_prior,
        tref=inj["tref"],
        nsteps=nsteps,
        nwalkers=nwalkers,
        ntemps=ntemps,
        log10beta_min=log10beta_min,
    )
    # walker plot is generated during main run of the search class
    mcmcsearch.run(
        walker_plot_args={"plot_det_stat": True, "injection_parameters": inj}
    )
    mcmcsearch.print_summary()

    # call some built-in plotting methods
    # these can all highlight the injection parameters, too
    logger.info("Making MCMCSearch {:s} corner plot...".format("-".join(search_keys)))
    mcmcsearch.plot_corner(truths=inj)
    logger.info("Making MCMCSearch prior-posterior comparison plot...")
    mcmcsearch.plot_prior_posterior(injection_parameters=inj)

    # NOTE: everything below here is just custom commandline output and plotting
    # for this particular example, which uses the PyFstat outputs,
    # but isn't very instructive if you just want to learn the main usage of the package.

    # some informative command-line output comparing search results and injection
    # get max of GridSearch, contains twoF and all Doppler parameters in the dict
    max_dict_grid = gridsearch.get_max_twoF()
    # same for MCMCSearch, here twoF is separate, and non-sampled parameters are not included either
    max_dict_mcmc, max_2F_mcmc = mcmcsearch.get_max_twoF()
    logger.info(
        "max2F={:.4f} from GridSearch, offsets from injection: {:s}.".format(
            max_dict_grid["twoF"],
            ", ".join(
                [
                    "{:.4e} in {:s}".format(max_dict_grid[key] - inj[key], key)
                    for key in search_keys
                ]
            ),
        )
    )
    logger.info(
        "max2F={:.4f} from MCMCSearch, offsets from injection: {:s}.".format(
            max_2F_mcmc,
            ", ".join(
                [
                    "{:.4e} in {:s}".format(max_dict_mcmc[key] - inj[key], key)
                    for key in search_keys
                ]
            ),
        )
    )
    # get additional point and interval estimators
    stats_dict_mcmc = mcmcsearch.get_summary_stats()
    logger.info(
        "mean   from MCMCSearch: offset from injection by      {:s},"
        " or in fractions of 2sigma intervals: {:s}.".format(
            ", ".join(
                [
                    "{:.4e} in {:s}".format(
                        stats_dict_mcmc[key]["mean"] - inj[key], key
                    )
                    for key in search_keys
                ]
            ),
            ", ".join(
                [
                    "{:.2f}% in {:s}".format(
                        100
                        * np.abs(stats_dict_mcmc[key]["mean"] - inj[key])
                        / (2 * stats_dict_mcmc[key]["std"]),
                        key,
                    )
                    for key in search_keys
                ]
            ),
        )
    )
    logger.info(
        "median from MCMCSearch: offset from injection by      {:s},"
        " or in fractions of 90% confidence intervals: {:s}.".format(
            ", ".join(
                [
                    "{:.4e} in {:s}".format(
                        stats_dict_mcmc[key]["median"] - inj[key], key
                    )
                    for key in search_keys
                ]
            ),
            ", ".join(
                [
                    "{:.2f}% in {:s}".format(
                        100
                        * np.abs(stats_dict_mcmc[key]["median"] - inj[key])
                        / (
                            stats_dict_mcmc[key]["upper90"]
                            - stats_dict_mcmc[key]["lower90"]
                        ),
                        key,
                    )
                    for key in search_keys
                ]
            ),
        )
    )

    # do additional custom plotting
    logger.info("Loading grid and MCMC search results for custom comparison plots...")
    gridfile = os.path.join(outdir, gridsearch.label + "_NA_GridSearch.txt")
    if not os.path.isfile(gridfile):
        raise RuntimeError(
            "Failed to load GridSearch results from file '{:s}',"
            " something must have gone wrong!".format(gridfile)
        )
    grid_res = pyfstat.utils.read_txt_file_with_header(gridfile)
    mcmc_file = os.path.join(outdir, mcmcsearch.label + "_samples.dat")
    if not os.path.isfile(mcmc_file):
        raise RuntimeError(
            "Failed to load MCMCSearch results from file '{:s}',"
            " something must have gone wrong!".format(mcmc_file)
        )
    mcmc_res = pyfstat.utils.read_txt_file_with_header(mcmc_file)

    zoom = {
        "F0": [inj["F0"] - 10 * dF0, inj["F0"] + 10 * dF0],
        "F1": [inj["F1"] - 5 * dF1, inj["F1"] + 5 * dF1],
    }

    # we'll use the two local plotting functions defined above
    # to avoid code duplication in the sky case
    logger.info("Creating MCMC-grid comparison plots...")
    plot_grid_vs_samples(grid_res, mcmc_res, "F0", "F1")
    plot_2F_scatter(grid_res, "grid", "F0", "F1")
    plot_2F_scatter(mcmc_res, "mcmc", "F0", "F1")
    if sky:
        plot_grid_vs_samples(grid_res, mcmc_res, "Alpha", "Delta")
        plot_2F_scatter(grid_res, "grid", "Alpha", "Delta")
        plot_2F_scatter(mcmc_res, "mcmc", "Alpha", "Delta")
