import logging
import os
from types import SimpleNamespace

import numpy as np
import pytest
from commons_for_tests import (
    FlakyError,
    default_signal_params,
    default_Writer_params,
    is_flaky,
)

import pyfstat


class _MCMCSearchTestUtils:
    # Plain mixin providing shared utility methods and default parameter values.
    # Subclasses that require SFT data should use BaseForMCMCSearchTests instead.
    label = "TestMCMCSearch"
    Band = 1
    outdir = "TestData"
    # Default Writer params (can be overridden in subclasses)
    sqrtSX = default_Writer_params["sqrtSX"]
    Tsft = default_Writer_params["Tsft"]
    tstart = default_Writer_params["tstart"]
    duration = default_Writer_params["duration"]
    detectors = default_Writer_params["detectors"]
    SFTWindowType = default_Writer_params["SFTWindowType"]
    SFTWindowParam = default_Writer_params["SFTWindowParam"]
    randSeed = default_Writer_params["randSeed"]

    def _run_search_with_interface(
        self,
        interface,
        pyfstat_run_kwargs=None,
        bilby_sampler_kwargs=None,
        bilby_run_kwargs=None,
    ):
        if interface == "pyfstat":
            self.search.run(**(pyfstat_run_kwargs or {}))
            return

        pytest.importorskip("bilby")
        nburn, nprod = self.search.nsteps[-2:]
        sampler_kwargs = {
            "nwalkers": self.search.nwalkers,
            "ntemps": self.search.ntemps,
            "log10beta_min": self.search.log10beta_min,
            "burn_in_fixed_discard": nburn,
            "burn_in_nact": 0,
            "thin_by_nact": 0,
            "nsamples": self.search.nwalkers * nprod,
            "mean_logl_frac": np.inf,
            "autocorr_tol": 0,
            "gradient_tau": np.inf,
            "gradient_mean_log_posterior": np.inf,
            "Q_tol": np.inf,
            "min_tau": 0,
            "niterations_per_check": 1,
            "resume": False,
            "check_point_plot": False,
            "verbose": False,
        }
        sampler_kwargs.update(bilby_sampler_kwargs or {})
        self.search.run_bilby(
            sampler="ptemcee",
            sampler_kwargs=sampler_kwargs,
            **(bilby_run_kwargs or {}),
        )

    def _check_twoF_predicted(self, assertTrue=True):
        self.twoF_predicted = self.Writer.predict_fstat()
        self.max_dict, self.maxTwoF = self.search.get_max_twoF()
        diff = np.abs((self.maxTwoF - self.twoF_predicted)) / self.twoF_predicted
        print(
            (
                "Predicted twoF is {} while recovered is {},"
                " relative difference: {}".format(
                    self.twoF_predicted, self.maxTwoF, diff
                )
            )
        )
        if assertTrue:
            assert diff < 0.3

    def _check_mcmc_quantiles(self, transient=False, assertTrue=True):
        summary_stats = self.search.get_summary_stats()
        nsigmas = 3
        conf = "99"

        if not transient:
            inj = {k: self.signal_params[k] for k in self.max_dict}
        else:
            inj = {
                "transient_tstart": self.Writer.signal_parameters["transientStartTime"],
                "transient_duration": self.Writer.signal_parameters["transientTau"],
            }

        for k in self.max_dict.keys():
            reldiff = np.abs((self.max_dict[k] - inj[k]) / inj[k])
            print("max2F  {:s} reldiff: {:.2e}".format(k, reldiff))
            reldiff = np.abs((summary_stats[k]["mean"] - inj[k]) / inj[k])
            print("mean   {:s} reldiff: {:.2e}".format(k, reldiff))
            reldiff = np.abs((summary_stats[k]["median"] - inj[k]) / inj[k])
            print("median {:s} reldiff: {:.2e}".format(k, reldiff))
        for k in self.max_dict.keys():
            lower = summary_stats[k]["mean"] - nsigmas * summary_stats[k]["std"]
            upper = summary_stats[k]["mean"] + nsigmas * summary_stats[k]["std"]
            within = (inj[k] >= lower) and (inj[k] <= upper)
            print(
                "{:s} in mean+-{:d}std ({} in [{},{}])? {}".format(
                    k, nsigmas, inj[k], lower, upper, within
                )
            )
            if assertTrue:
                try:
                    assert within
                except AssertionError:
                    print("FAIL: Not within tolerances!")
                    raise FlakyError
            within = (inj[k] >= summary_stats[k]["lower" + conf]) and (
                inj[k] <= summary_stats[k]["upper" + conf]
            )
            print(
                "{:s} in {:s}% quantiles ({} in [{},{}])? {}".format(
                    k,
                    conf,
                    inj[k],
                    summary_stats[k]["lower" + conf],
                    summary_stats[k]["upper" + conf],
                    within,
                )
            )
            if assertTrue:
                try:
                    assert within
                except AssertionError:
                    print("FAIL: Not within tolerances!")
                    raise FlakyError

    def _test_plots(self):
        self.search.plot_corner(add_prior=True, truths=self.signal_params)
        self.search.plot_prior_posterior(injection_parameters=self.signal_params)
        self.search.plot_cumulative_max()
        self.search.plot_chainconsumer()


@pytest.mark.flaky(max_runs=3, min_passes=1, rerun_filter=is_flaky)
@pytest.mark.usefixtures("data_fixture")
class BaseForMCMCSearchTests(_MCMCSearchTestUtils):
    # Base class for MCMC search tests that rely on the data_fixture.
    # TestMCMCTransientSearch manages its own Writer and should inherit from
    # _MCMCSearchTestUtils directly to avoid unintended data_fixture parametrisation.
    pass


def test_bilby_likelihood_reuses_cached_periodograms():
    pytest.importorskip("bilby")
    from pyfstat.bilby_based_searches import PyFstatBilbyLikelihood

    likelihood = object.__new__(PyFstatBilbyLikelihood)
    cached_periodograms = {0: np.array([[1.0, 2.0]])}
    likelihood._sft_periodograms_by_detector = cached_periodograms

    assert likelihood._get_running_median_periodograms() is cached_periodograms


@pytest.mark.parametrize(
    "sftfilepattern, include_noise_log_likelihood",
    [
        (None, None),
        ("*.sft", False),
    ],
)
def test_bilby_likelihood_logs_normalization_choice(
    monkeypatch,
    caplog,
    sftfilepattern,
    include_noise_log_likelihood,
):
    pytest.importorskip("bilby")
    from pyfstat.bilby_based_searches import PyFstatBilbyLikelihood

    caplog.set_level(logging.INFO, logger="pyfstat.bilby_based_searches")
    pyfstat_search = SimpleNamespace(theta_keys=["F0"], sftfilepattern=sftfilepattern)
    likelihood = PyFstatBilbyLikelihood(
        pyfstat_search,
        include_noise_log_likelihood=include_noise_log_likelihood,
    )
    monkeypatch.setattr(
        likelihood,
        "_compute_noise_log_likelihood",
        lambda: pytest.fail("Noise likelihood should not be computed"),
    )

    assert likelihood.include_noise_log_likelihood is False
    assert likelihood.noise_log_likelihood() == 0.0
    assert "does not affect posterior parameter estimation" in caplog.text
    assert "include_noise_log_likelihood=False" in caplog.text
    assert not any(record.levelno >= logging.WARNING for record in caplog.records)


def test_bilby_absolute_likelihood_rejects_data_without_file_backed_sfts():
    pytest.importorskip("bilby")
    from pyfstat.bilby_based_searches import PyFstatBilbyLikelihood

    pyfstat_search = SimpleNamespace(theta_keys=["F0"], sftfilepattern=None)
    with pytest.raises(ValueError, match="without file-backed SFT amplitudes"):
        PyFstatBilbyLikelihood(
            pyfstat_search,
            include_noise_log_likelihood=True,
        )


def test_bilby_bsgl_defaults_to_ratio(caplog):
    pytest.importorskip("bilby")
    from pyfstat.bilby_based_searches import PyFstatBilbyLikelihood

    caplog.set_level(logging.INFO, logger="pyfstat.bilby_based_searches")
    pyfstat_search = SimpleNamespace(
        theta_keys=["F0"], sftfilepattern="*.sft", BSGL=True
    )
    likelihood = PyFstatBilbyLikelihood(pyfstat_search)

    assert likelihood.include_noise_log_likelihood is False
    assert likelihood.noise_log_likelihood() == 0.0
    assert "composite Gaussian-or-line-noise denominator" in caplog.text
    assert "does not affect posterior parameter estimation" in caplog.text
    assert not any(record.levelno >= logging.WARNING for record in caplog.records)


def test_bilby_bsgl_overrides_absolute_gaussian_noise_likelihood(caplog):
    pytest.importorskip("bilby")
    from pyfstat.bilby_based_searches import PyFstatBilbyLikelihood

    pyfstat_search = SimpleNamespace(
        theta_keys=["F0"], sftfilepattern="*.sft", BSGL=True
    )
    likelihood = PyFstatBilbyLikelihood(
        pyfstat_search,
        include_noise_log_likelihood=True,
    )

    assert likelihood.include_noise_log_likelihood is False
    assert likelihood.noise_log_likelihood() == 0.0
    assert "Ignoring include_noise_log_likelihood=True" in caplog.text


def test_run_bilby_search_uses_separate_pickle_path(monkeypatch, tmp_path):
    bilby = pytest.importorskip("bilby")
    pandas = pytest.importorskip("pandas")
    from pyfstat.bilby_based_searches import run_bilby_search

    pickle_call = {}
    native_pickle_path = tmp_path / "test_saved_data.p"
    pyfstat_search = SimpleNamespace(
        theta_keys=["F0"],
        theta_prior={"F0": {"type": "unif", "lower": 29.9, "upper": 30.1}},
        sftfilepattern="unused.sft",
        outdir=str(tmp_path),
        label="test",
        ndim=1,
        pickle_path=str(native_pickle_path),
    )
    pyfstat_search._initiate_search_object = lambda: setattr(
        pyfstat_search, "search", object()
    )
    pyfstat_search._logl = lambda sample, search: sample[0]
    pyfstat_search._pickle_data = lambda *args, **kwargs: pickle_call.update(kwargs)
    result = SimpleNamespace(
        posterior=pandas.DataFrame(
            {"F0": [30.0], "log_prior": [0.0], "log_likelihood": [0.0]}
        )
    )
    monkeypatch.setattr(bilby, "run_sampler", lambda **kwargs: result)

    run_bilby_search(
        pyfstat_search,
        save_pickle=True,
        export_samples=False,
        save_loudest=False,
    )

    expected_path = tmp_path / "test_bilby_saved_data.p"
    assert pyfstat_search.bilby_pickle_path == str(expected_path)
    assert pickle_call["pickle_path"] == str(expected_path)
    assert pyfstat_search.pickle_path == str(native_pickle_path)


def test_pickle_data_can_write_to_separate_path(tmp_path):
    search = object.__new__(pyfstat.MCMCSearch)
    search.pickle_path = str(tmp_path / "native_saved_data.p")
    search.chain = np.zeros((1, 1, 1, 1))
    search._get_data_dictionary_to_save = lambda: {}
    bilby_pickle_path = tmp_path / "bilby_saved_data.p"
    samples = np.zeros((1, 1))
    likelihoods = np.zeros(1)

    search._pickle_data(
        samples,
        likelihoods,
        likelihoods,
        likelihoods.reshape((1, 1, 1)),
        pickle_path=str(bilby_pickle_path),
    )

    assert bilby_pickle_path.is_file()
    assert not os.path.isfile(search.pickle_path)


@pytest.mark.parametrize(
    "assume_sqrt_sx, expected_psds",
    [(2.0, {"H1": 4.0, "L1": 4.0}), ("2,3", {"H1": 4.0, "L1": 9.0})],
)
def test_bilby_noise_likelihood_uses_assumed_psd(
    monkeypatch, assume_sqrt_sx, expected_psds
):
    pytest.importorskip("bilby")
    from pyfstat.bilby_based_searches import PyFstatBilbyLikelihood

    data = {
        "H1": np.array([[1.0 + 0.0j], [2.0 + 0.0j]]),
        "L1": np.array([[3.0 + 0.0j], [4.0 + 0.0j]]),
    }
    search = SimpleNamespace(
        minCoverFreq=10.0,
        maxCoverFreq=11.0,
        Tsft=2.0,
        assumeSqrtSX=assume_sqrt_sx,
        detector_names=np.array(["H1", "L1"]),
        numDetectors=2,
    )
    pyfstat_search = SimpleNamespace(
        search=search,
        minStartTime=None,
        maxStartTime=None,
        sftfilepattern="unused",
    )
    likelihood = object.__new__(PyFstatBilbyLikelihood)
    likelihood.pyfstat_search = pyfstat_search
    monkeypatch.setattr(
        pyfstat.utils.sft,
        "get_sft_as_arrays",
        lambda *args, **kwargs: (None, None, data),
    )
    monkeypatch.setattr(
        likelihood,
        "_get_running_median_periodograms",
        lambda: pytest.fail("Running-median PSD should not be computed"),
    )

    expected = 0.0
    for detector, values in data.items():
        psd = expected_psds[detector]
        power = np.abs(values) ** 2
        expected -= np.sum(power / psd + np.log(np.pi * psd))

    assert likelihood._compute_noise_log_likelihood() == pytest.approx(expected)


def test_lognorm_lnprior_returns_log_density_and_rejects_negative_values():
    search = object.__new__(pyfstat.MCMCSearch)
    lnprior = search._generic_lnprior(type="lognorm", loc=np.log(30.0), scale=1e-6)

    assert np.isfinite(lnprior(30.0))
    assert np.isneginf(lnprior(-30.0))


class TestMCMCSearch(BaseForMCMCSearchTests):
    label = "TestMCMCSearch"
    BSGL = False

    @pytest.mark.parametrize(
        "prior_choice",
        [
            "uniformF0-uniformF1-fixedSky",
            "log10uniformF0-uniformF1-fixedSky",
            "normF0-normF1-fixedSky",
            "lognormF0-halfnormF1-fixedSky",
            "normF0-neghalfnormF1-fixedSky",
            "normF0-normF1-uniformSky",
        ],
    )
    @pytest.mark.parametrize("interface", ["pyfstat", "bilby"])
    def test_fully_coherent_MCMC(self, prior_choice, interface):
        thetas = {
            "uniformF0-uniformF1-fixedSky": {
                "F0": {
                    "type": "unif",
                    "lower": self.signal_params["F0"] - 1e-6,
                    "upper": self.signal_params["F0"] + 1e-6,
                },
                "F1": {
                    "type": "unif",
                    "lower": self.signal_params["F1"] - 1e-10,
                    "upper": self.signal_params["F1"] + 1e-10,
                },
                "F2": self.signal_params["F2"],
                "Alpha": self.signal_params["Alpha"],
                "Delta": self.signal_params["Delta"],
            },
            "log10uniformF0-uniformF1-fixedSky": {
                "F0": {
                    "type": "log10unif",
                    "log10lower": np.log10(self.signal_params["F0"] - 1e-6),
                    "log10upper": np.log10(self.signal_params["F0"] + 1e-6),
                },
                "F1": {
                    "type": "unif",
                    "lower": self.signal_params["F1"] - 1e-10,
                    "upper": self.signal_params["F1"] + 1e-10,
                },
                "F2": self.signal_params["F2"],
                "Alpha": self.signal_params["Alpha"],
                "Delta": self.signal_params["Delta"],
            },
            "normF0-normF1-fixedSky": {
                "F0": {"type": "norm", "loc": self.signal_params["F0"], "scale": 1e-6},
                "F1": {"type": "norm", "loc": self.signal_params["F1"], "scale": 1e-10},
                "F2": self.signal_params["F2"],
                "Alpha": self.signal_params["Alpha"],
                "Delta": self.signal_params["Delta"],
            },
            "lognormF0-halfnormF1-fixedSky": {
                # lognorm parametrization is weird, from the scipy docs:
                # "A common parametrization for a lognormal random variable Y
                # is in terms of the mean, mu, and standard deviation, sigma,
                # of the unique normally distributed random variable X
                # such that exp(X) = Y.
                # This parametrization corresponds to setting s = sigma
                # and scale = exp(mu)."
                # Hence, to set up a "lognorm" prior, we need
                # to give "loc" in log scale but "scale" in linear scale
                # Also, "lognorm" makes no sense for negative F1,
                # hence combining this with "halfnorm" into a single case.
                "F0": {
                    "type": "lognorm",
                    "loc": np.log(self.signal_params["F0"]),
                    "scale": 1e-6,
                },
                "F1": {
                    "type": "halfnorm",
                    "loc": self.signal_params["F1"] - 1e-10,
                    "scale": 1e-10,
                },
                "F2": self.signal_params["F2"],
                "Alpha": self.signal_params["Alpha"],
                "Delta": self.signal_params["Delta"],
            },
            "normF0-neghalfnormF1-fixedSky": {
                "F0": {"type": "norm", "loc": self.signal_params["F0"], "scale": 1e-6},
                "F1": {
                    "type": "neghalfnorm",
                    "loc": self.signal_params["F1"] + 1e-10,
                    "scale": 1e-10,
                },
                "F2": self.signal_params["F2"],
                "Alpha": self.signal_params["Alpha"],
                "Delta": self.signal_params["Delta"],
            },
            "normF0-normF1-uniformSky": {
                # norm in sky is too dangerous, can easily jump out of range
                "F0": {"type": "norm", "loc": self.signal_params["F0"], "scale": 1e-6},
                "F1": {"type": "norm", "loc": self.signal_params["F1"], "scale": 1e-10},
                "F2": self.signal_params["F2"],
                "Alpha": {
                    "type": "unif",
                    "lower": self.signal_params["Alpha"] - 0.01,
                    "upper": self.signal_params["Alpha"] + 0.01,
                },
                "Delta": {
                    "type": "unif",
                    "lower": self.signal_params["Delta"] - 0.01,
                    "upper": self.signal_params["Delta"] + 0.01,
                },
            },
        }
        theta = thetas[prior_choice]
        nsteps = (
            [50, 50]
            if prior_choice
            in {
                "lognormF0-halfnormF1-fixedSky",
                "normF0-neghalfnormF1-fixedSky",
            }
            else [20, 20]
        )
        cover_frequency_kwargs = (
            {
                "minCoverFreq": self.signal_params["F0"] - 0.0035,
                "maxCoverFreq": self.signal_params["F0"] + 0.0035,
            }
            if prior_choice == "lognormF0-halfnormF1-fixedSky"
            else {}
        )
        self.search = pyfstat.MCMCSearch(
            label=self.label + "-" + prior_choice + "-" + interface,
            outdir=self.outdir,
            theta_prior=theta,
            tref=self.signal_params["tref"],
            sftfilepattern=self.Writer.sftfilepath,
            nsteps=nsteps,
            nwalkers=20,
            ntemps=2,
            log10beta_min=-1,
            BSGL=self.BSGL,
            **cover_frequency_kwargs,
        )
        bilby_run_kwargs = None
        test_direct_bilby_options = (
            interface == "bilby"
            and not self.BSGL
            and prior_choice == "uniformF0-uniformF1-fixedSky"
        )
        if test_direct_bilby_options:
            bilby = pytest.importorskip("bilby")

            bilby_priors = bilby.core.prior.PriorDict()
            bilby_priors["F0"] = bilby.core.prior.Uniform(
                theta["F0"]["lower"], theta["F0"]["upper"], name="F0"
            )
            bilby_priors["F1"] = bilby.core.prior.Uniform(
                theta["F1"]["lower"], theta["F1"]["upper"], name="F1"
            )
            bilby_run_kwargs = {
                "bilby_priors": bilby_priors,
                "save_bilby_logs": True,
                "save_bilby_progress": True,
            }
        self._run_search_with_interface(
            interface,
            pyfstat_run_kwargs={"plot_walkers": False},
            bilby_run_kwargs=bilby_run_kwargs,
        )
        if test_direct_bilby_options:
            assert os.path.isfile(os.path.join(self.outdir, f"{self.search.label}.log"))
            assert os.path.isfile(
                os.path.join(self.outdir, f"{self.search.label}_bilby_progress.log")
            )
        self.search.print_summary()
        self.search.write_prior_table()
        self._check_twoF_predicted()
        self._check_mcmc_quantiles()
        self._test_plots()


class TestMCMCSearchBSGL(TestMCMCSearch):
    label = "TestMCMCSearch"
    detectors = "H1,L1"
    BSGL = True

    @pytest.mark.parametrize("interface", ["pyfstat", "bilby"])
    def test_MCMC_search_on_data_with_line(self, interface):
        # We reuse the default multi-IFO SFTs
        # but add an additional single-detector artifact to H1 only.
        # For simplicity, this is modelled here as a fully modulated CW-like signal,
        # just restricted to the single detector.
        SFTs_H1 = self.Writer.sftfilepath.split(";")[0]
        SFTs_L1 = self.Writer.sftfilepath.split(";")[1]
        extra_writer = pyfstat.Writer(
            label=self.label + "WithLine",
            outdir=self.outdir,
            tref=self.signal_params["tref"],
            F0=self.Writer.F0 + 0.5e-2,
            F1=0,
            F2=0,
            Alpha=self.Writer.signal_parameters["Alpha"],
            Delta=self.Writer.signal_parameters["Delta"],
            h0=10 * self.Writer.signal_parameters["h0"],
            cosi=self.Writer.signal_parameters["cosi"],
            sqrtSX=0,  # don't add yet another set of Gaussian noise
            noiseSFTs=SFTs_H1,
            SFTWindowType=self.Writer.SFTWindowType,
            SFTWindowParam=self.Writer.SFTWindowParam,
        )
        extra_writer.make_data()
        data_with_line = ";".join([SFTs_L1, extra_writer.sftfilepath])
        # use a single fixed prior and search F0 only for speed
        thetas = {
            "F0": {
                "type": "unif",
                "lower": self.signal_params["F0"] - 1e-2,
                "upper": self.signal_params["F0"] + 1e-2,
            },
            "F1": self.signal_params["F1"],
            "F2": self.signal_params["F2"],
            "Alpha": self.signal_params["Alpha"],
            "Delta": self.signal_params["Delta"],
        }
        # now run a standard F-stat search over this data
        self.search = pyfstat.MCMCSearch(
            label=self.label + "F-" + interface,
            outdir=self.outdir,
            theta_prior=thetas,
            tref=self.signal_params["tref"],
            sftfilepattern=data_with_line,
            nsteps=[20, 20],
            nwalkers=20,
            ntemps=2,
            log10beta_min=-1,
            BSGL=False,
        )
        self._run_search_with_interface(
            interface, pyfstat_run_kwargs={"plot_walkers": True}
        )
        self.search.print_summary()
        # The standard checks here are expected to fail,
        # as the F-search will get confused by the line
        # and recover a much higher maxTwoF than predicted.
        self._check_twoF_predicted(assertTrue=False)
        mode_F0_Fsearch = self.max_dict["F0"]
        maxTwoF_Fsearch = self.maxTwoF
        self._check_mcmc_quantiles(assertTrue=False)
        assert maxTwoF_Fsearch > self.twoF_predicted
        self._test_plots()
        # also run a BSGL search over the same data
        self.search = pyfstat.MCMCSearch(
            label=self.label + "BSGL-" + interface,
            outdir=self.outdir,
            theta_prior=thetas,
            tref=self.signal_params["tref"],
            sftfilepattern=data_with_line,
            nsteps=[20, 20],
            nwalkers=20,
            ntemps=2,
            log10beta_min=-1,
            BSGL=True,
        )
        self._run_search_with_interface(
            interface, pyfstat_run_kwargs={"plot_walkers": True}
        )
        self.search.print_summary()
        # Still skipping the standard checks,
        # as we're using too cheap a MCMC setup here for them to be robust.
        self._check_twoF_predicted(assertTrue=False)
        mode_F0_BSGLsearch = self.max_dict["F0"]
        maxTwoF_BSGLsearch = self.maxTwoF
        self._check_mcmc_quantiles(assertTrue=False)
        # But for sure, the BSGL search should find a lower-F mode
        # closer to the true multi-IFO signal.
        assert maxTwoF_BSGLsearch < maxTwoF_Fsearch
        assert mode_F0_BSGLsearch < mode_F0_Fsearch
        assert np.abs(mode_F0_BSGLsearch - self.signal_params["F0"]) < np.abs(
            mode_F0_Fsearch - self.signal_params["F0"]
        )
        assert maxTwoF_BSGLsearch < self.twoF_predicted
        self._test_plots()


class TestMCMCSemiCoherentSearch(BaseForMCMCSearchTests):
    label = "TestMCMCSemiCoherentSearch"

    @pytest.mark.parametrize("interface", ["pyfstat", "bilby"])
    def test_semi_coherent_MCMC(self, interface):
        theta = {
            "F0": {
                "type": "unif",
                "lower": self.signal_params["F0"] - 1e-6,
                "upper": self.signal_params["F0"] + 1e-6,
            },
            "F1": {
                "type": "unif",
                "lower": self.signal_params["F1"] - 1e-10,
                "upper": self.signal_params["F1"] + 1e-10,
            },
            "F2": self.signal_params["F2"],
            "Alpha": self.signal_params["Alpha"],
            "Delta": self.signal_params["Delta"],
        }
        nsegs = 2
        self.search = pyfstat.MCMCSemiCoherentSearch(
            label=self.label + "-" + interface,
            outdir=self.outdir,
            theta_prior=theta,
            tref=self.signal_params["tref"],
            sftfilepattern=self.Writer.sftfilepath,
            nsteps=[100, 100],
            nwalkers=100,
            ntemps=2,
            log10beta_min=-1,
            nsegs=nsegs,
        )
        self._run_search_with_interface(
            interface, pyfstat_run_kwargs={"plot_walkers": False}
        )
        self.search.print_summary()

        self._check_twoF_predicted()

        # recover per-segment twoF values at max point
        twoF_sc = self.search.search.get_semicoherent_det_stat(
            self.max_dict["F0"],
            self.max_dict["F1"],
            self.signal_params["F2"],
            self.signal_params["Alpha"],
            self.signal_params["Delta"],
            record_segments=True,
        )
        assert np.abs(twoF_sc - self.maxTwoF) / self.maxTwoF < 0.01
        twoF_per_seg = np.array(self.search.search.twoF_per_segment)
        assert len(twoF_per_seg) == nsegs
        twoF_summed = twoF_per_seg.sum()
        assert np.abs(twoF_summed - twoF_sc) / twoF_sc < 0.01

        self._check_mcmc_quantiles()
        self._test_plots()


class TestMCMCFollowUpSearch(BaseForMCMCSearchTests):
    label = "TestMCMCFollowUpSearch"
    # Supersky metric cannot be computed for segment lengths <= ~24 hours
    duration = 10 * 86400
    # FIXME: if h0 too high for given duration, offsets to PFS become too large
    h0 = 0.1

    def test_MCMC_followup_search(self):
        theta = {
            "F0": {
                "type": "unif",
                "lower": self.signal_params["F0"] - 1e-6,
                "upper": self.signal_params["F0"] + 1e-6,
            },
            "F1": {
                "type": "unif",
                "lower": self.signal_params["F1"] - 1e-10,
                "upper": self.signal_params["F1"] + 1e-10,
            },
            "F2": self.signal_params["F2"],
            "Alpha": self.signal_params["Alpha"],
            "Delta": self.signal_params["Delta"],
        }
        nsegs = 10
        NstarMax = 1000
        self.search = pyfstat.MCMCFollowUpSearch(
            label=self.label,
            outdir=self.outdir,
            theta_prior=theta,
            tref=self.signal_params["tref"],
            sftfilepattern=self.Writer.sftfilepath,
            nsteps=[100, 100],
            nwalkers=100,
            ntemps=2,
            log10beta_min=-1,
        )
        self.search.run(
            plot_walkers=False,
            NstarMax=NstarMax,
            Nsegs0=nsegs,
        )
        self.search.print_summary()
        self._check_twoF_predicted()
        self._check_mcmc_quantiles()
        self._test_plots()


def test_mcmc_followup_search_rejects_bilby_interface():
    search = object.__new__(pyfstat.MCMCFollowUpSearch)

    with pytest.raises(NotImplementedError, match="multiple sampler stages"):
        search.run_bilby()


@pytest.mark.flaky(max_runs=3, min_passes=1, rerun_filter=is_flaky)
@pytest.mark.usefixtures("outdir")
class TestMCMCTransientSearch(_MCMCSearchTestUtils):
    label = "TestMCMCTransientSearch"
    duration = 86400

    def setup_method(self, method):
        # Allow overwriting parameters from child classes
        self.signal_params = {}
        for key, val in default_signal_params.items():
            self.signal_params[key] = getattr(self, key, val)
        self.signal_params["transientWindowType"] = "rect"
        self.signal_params["transientStartTime"] = int(
            self.tstart + 0.25 * self.duration
        )
        self.signal_params["transientTau"] = int(0.5 * self.duration)
        self.Writer = pyfstat.Writer(
            label=self.label,
            tstart=self.tstart,
            duration=self.duration,
            **{
                k: v
                for k, v in self.signal_params.items()
                if not (k.startswith("F") and int(k[-1]) > 2)
            },
            outdir=self.outdir,
            sqrtSX=self.sqrtSX,
            Band=self.Band,
            detectors=self.detectors,
            SFTWindowType=self.SFTWindowType,
            SFTWindowParam=self.SFTWindowParam,
            randSeed=self.randSeed,
        )
        self.Writer.make_data(verbose=True)
        self.basic_theta = {
            "F0": self.signal_params["F0"],
            "F1": self.signal_params["F1"],
            "F2": self.signal_params["F2"],
            "Alpha": self.signal_params["Alpha"],
            "Delta": self.signal_params["Delta"],
        }
        self.MCMC_params = {
            "nsteps": [50, 50],
            "nwalkers": 50,
            "ntemps": 2,
            "log10beta_min": -1,
        }

    def _get_valid_bilby_pos0(self):
        """Draw PTMCMC initial points whose transient ends within the data."""
        rng = np.random.default_rng(0)
        shape = (self.search.ntemps, self.search.nwalkers)
        draws = {
            key: rng.uniform(
                self.search.theta_prior[key]["lower"],
                self.search.theta_prior[key]["upper"],
                size=shape,
            )
            for key in self.search.theta_keys
        }
        tstart = draws.get(
            "transient_tstart",
            np.full(shape, self.signal_params["transientStartTime"]),
        )
        duration = draws.get(
            "transient_duration",
            np.full(shape, self.signal_params["transientTau"]),
        )
        invalid = tstart + duration > self.Writer.tend
        while np.any(invalid):
            for key in draws:
                prior = self.search.theta_prior[key]
                draws[key][invalid] = rng.uniform(
                    prior["lower"], prior["upper"], size=np.sum(invalid)
                )
            tstart = draws.get(
                "transient_tstart",
                np.full(shape, self.signal_params["transientStartTime"]),
            )
            duration = draws.get(
                "transient_duration",
                np.full(shape, self.signal_params["transientTau"]),
            )
            invalid = tstart + duration > self.Writer.tend
        return np.stack([draws[key] for key in self.search.theta_keys], axis=-1)

    def _run_transient_search(self, interface):
        bilby_sampler_kwargs = None
        if interface == "bilby":
            bilby_sampler_kwargs = {"pos0": self._get_valid_bilby_pos0()}
        self._run_search_with_interface(
            interface,
            pyfstat_run_kwargs={"plot_walkers": False},
            bilby_sampler_kwargs=bilby_sampler_kwargs,
        )

    @pytest.mark.parametrize("interface", ["pyfstat", "bilby"])
    def test_transient_MCMC_t0only(self, interface):
        theta = {
            **self.basic_theta,
            "transient_tstart": {
                "type": "unif",
                "lower": self.Writer.tstart,
                "upper": self.Writer.tend - 2 * self.Writer.Tsft,
            },
            "transient_duration": self.signal_params["transientTau"],
        }
        self.search = pyfstat.MCMCTransientSearch(
            label=self.label + "-t0only-" + interface,
            outdir=self.outdir,
            theta_prior=theta,
            tref=self.signal_params["tref"],
            sftfilepattern=self.Writer.sftfilepath,
            **self.MCMC_params,
            transientWindowType=self.signal_params["transientWindowType"],
        )
        self._run_transient_search(interface)
        self.search.print_summary()
        self._check_twoF_predicted()
        self._check_mcmc_quantiles(transient=True)
        self._test_plots()

    @pytest.mark.parametrize("interface", ["pyfstat", "bilby"])
    def test_transient_MCMC_tauonly(self, interface):
        theta = {
            **self.basic_theta,
            "transient_tstart": self.signal_params["transientStartTime"],
            "transient_duration": {
                "type": "unif",
                "lower": 2 * self.Writer.Tsft,
                "upper": self.Writer.duration - 2 * self.Writer.Tsft,
            },
        }
        self.search = pyfstat.MCMCTransientSearch(
            label=self.label + "-tauonly-" + interface,
            outdir=self.outdir,
            theta_prior=theta,
            tref=self.signal_params["tref"],
            sftfilepattern=self.Writer.sftfilepath,
            **self.MCMC_params,
            transientWindowType=self.signal_params["transientWindowType"],
        )
        self._run_transient_search(interface)
        self.search.print_summary()
        self._check_twoF_predicted()
        self._check_mcmc_quantiles(transient=True)
        self._test_plots()

    def _test_transient_MCMC_t0_tau(self, interface, BtSG):
        theta = {
            **self.basic_theta,
            "transient_tstart": {
                "type": "unif",
                "lower": self.Writer.tstart,
                "upper": self.Writer.tend - 2 * self.Writer.Tsft,
            },
            "transient_duration": {
                "type": "unif",
                "lower": 2 * self.Writer.Tsft,
                "upper": self.Writer.duration - 2 * self.Writer.Tsft,
            },
        }
        self.search = pyfstat.MCMCTransientSearch(
            label=(self.label + ("-BtSG" if BtSG else "-t0-tau") + "-" + interface),
            outdir=self.outdir,
            theta_prior=theta,
            tref=self.signal_params["tref"],
            sftfilepattern=self.Writer.sftfilepath,
            **self.MCMC_params,
            transientWindowType=self.signal_params["transientWindowType"],
            BtSG=BtSG,
        )
        self._run_transient_search(interface)
        self.search.print_summary()
        self._check_twoF_predicted(assertTrue=not BtSG)
        self._check_mcmc_quantiles(transient=True)
        self._test_plots()

    @pytest.mark.parametrize("interface", ["pyfstat", "bilby"])
    def test_transient_MCMC_t0_tau(self, interface):
        self._test_transient_MCMC_t0_tau(interface=interface, BtSG=False)

    @pytest.mark.parametrize("interface", ["pyfstat", "bilby"])
    def test_transient_MCMC_t0_tau_BtSG(self, interface):
        self._test_transient_MCMC_t0_tau(interface=interface, BtSG=True)
