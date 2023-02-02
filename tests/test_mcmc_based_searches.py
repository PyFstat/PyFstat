import numpy as np
import pytest

# FIXME this should be made cleaner with fixtures
from commons_for_tests import (
    BaseForTestsWithData,
    FlakyError,
    default_signal_params,
    is_flaky,
)

import pyfstat


@pytest.mark.flaky(max_runs=3, min_passes=1, rerun_filter=is_flaky)
class BaseForMCMCSearchTests(BaseForTestsWithData):
    # this class is only used for common utilities for MCMCSearch-based classes
    # and doesn't run any tests itself
    label = "TestMCMCSearch"
    Band = 1

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
            self.assertTrue(diff < 0.3)

    def _check_mcmc_quantiles(self, transient=False, assertTrue=True):
        summary_stats = self.search.get_summary_stats()
        nsigmas = 3
        conf = "99"

        if not transient:
            inj = {k: getattr(self.Writer, k) for k in self.max_dict}
        else:
            inj = {
                "transient_tstart": self.Writer.transientStartTime,
                "transient_duration": self.Writer.transientTau,
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
                    self.assertTrue(within)
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
                    self.assertTrue(within)
                except AssertionError:
                    print("FAIL: Not within tolerances!")
                    raise FlakyError

    def _test_plots(self):
        self.search.plot_corner(add_prior=True)
        self.search.plot_prior_posterior()
        self.search.plot_cumulative_max()
        self.search.plot_chainconsumer()


class TestMCMCSearch(BaseForMCMCSearchTests):
    label = "TestMCMCSearch"
    BSGL = False

    def test_fully_coherent_MCMC(self):
        # use a single test case with loop over multiple prior choices
        # this could be much more elegantly done with @pytest.mark.parametrize
        # but that cannot be mixed with unittest classes
        thetas = {
            "uniformF0-uniformF1-fixedSky": {
                "F0": {
                    "type": "unif",
                    "lower": self.F0 - 1e-6,
                    "upper": self.F0 + 1e-6,
                },
                "F1": {
                    "type": "unif",
                    "lower": self.F1 - 1e-10,
                    "upper": self.F1 + 1e-10,
                },
                "F2": self.F2,
                "Alpha": self.Alpha,
                "Delta": self.Delta,
            },
            "log10uniformF0-uniformF1-fixedSky": {
                "F0": {
                    "type": "log10unif",
                    "log10lower": np.log10(self.F0 - 1e-6),
                    "log10upper": np.log10(self.F0 + 1e-6),
                },
                "F1": {
                    "type": "unif",
                    "lower": self.F1 - 1e-10,
                    "upper": self.F1 + 1e-10,
                },
                "F2": self.F2,
                "Alpha": self.Alpha,
                "Delta": self.Delta,
            },
            "normF0-normF1-fixedSky": {
                "F0": {"type": "norm", "loc": self.F0, "scale": 1e-6},
                "F1": {"type": "norm", "loc": self.F1, "scale": 1e-10},
                "F2": self.F2,
                "Alpha": self.Alpha,
                "Delta": self.Delta,
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
                "F0": {"type": "lognorm", "loc": np.log(self.F0), "scale": 1e-6},
                "F1": {"type": "halfnorm", "loc": self.F1 - 1e-10, "scale": 1e-10},
                "F2": self.F2,
                "Alpha": self.Alpha,
                "Delta": self.Delta,
            },
            "normF0-normF1-uniformSky": {
                # norm in sky is too dangerous, can easily jump out of range
                "F0": {"type": "norm", "loc": self.F0, "scale": 1e-6},
                "F1": {"type": "norm", "loc": self.F1, "scale": 1e-10},
                "F2": self.F2,
                "Alpha": {
                    "type": "unif",
                    "lower": self.Alpha - 0.01,
                    "upper": self.Alpha + 0.01,
                },
                "Delta": {
                    "type": "unif",
                    "lower": self.Delta - 0.01,
                    "upper": self.Delta + 0.01,
                },
            },
        }

        for prior_choice in thetas:
            self.search = pyfstat.MCMCSearch(
                label=self.label + "-" + prior_choice,
                outdir=self.outdir,
                theta_prior=thetas[prior_choice],
                tref=self.tref,
                sftfilepattern=self.Writer.sftfilepath,
                nsteps=[20, 20],
                nwalkers=20,
                ntemps=2,
                log10beta_min=-1,
                BSGL=self.BSGL,
            )
            self.search.run(plot_walkers=False)
            self.search.print_summary()
            self._check_twoF_predicted()
            self._check_mcmc_quantiles()
            self._test_plots()


class TestMCMCSearchBSGL(TestMCMCSearch):
    label = "TestMCMCSearch"
    detectors = "H1,L1"
    BSGL = True

    def test_MCMC_search_on_data_with_line(self):
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
            F0=self.Writer.F0 + 0.5e-2,
            F1=0,
            F2=0,
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
        # use a single fixed prior and search F0 only for speed
        thetas = {
            "F0": {
                "type": "unif",
                "lower": self.F0 - 1e-2,
                "upper": self.F0 + 1e-2,
            },
            "F1": self.F1,
            "F2": self.F2,
            "Alpha": self.Alpha,
            "Delta": self.Delta,
        }
        # now run a standard F-stat search over this data
        self.search = pyfstat.MCMCSearch(
            label=self.label + "F",
            outdir=self.outdir,
            theta_prior=thetas,
            tref=self.tref,
            sftfilepattern=data_with_line,
            nsteps=[20, 20],
            nwalkers=20,
            ntemps=2,
            log10beta_min=-1,
            BSGL=False,
        )
        self.search.run(plot_walkers=True)
        self.search.print_summary()
        # The standard checks here are expected to fail,
        # as the F-search will get confused by the line
        # and recover a much higher maxTwoF than predicted.
        self._check_twoF_predicted(assertTrue=False)
        mode_F0_Fsearch = self.max_dict["F0"]
        maxTwoF_Fsearch = self.maxTwoF
        self._check_mcmc_quantiles(assertTrue=False)
        self.assertTrue(maxTwoF_Fsearch > self.twoF_predicted)
        self._test_plots()
        # also run a BSGL search over the same data
        self.search = pyfstat.MCMCSearch(
            label=self.label + "BSGL",
            outdir=self.outdir,
            theta_prior=thetas,
            tref=self.tref,
            sftfilepattern=data_with_line,
            nsteps=[20, 20],
            nwalkers=20,
            ntemps=2,
            log10beta_min=-1,
            BSGL=True,
        )
        self.search.run(plot_walkers=True)
        self.search.print_summary()
        # Still skipping the standard checks,
        # as we're using too cheap a MCMC setup here for them to be robust.
        self._check_twoF_predicted(assertTrue=False)
        mode_F0_BSGLsearch = self.max_dict["F0"]
        maxTwoF_BSGLsearch = self.maxTwoF
        self._check_mcmc_quantiles(assertTrue=False)
        # But for sure, the BSGL search should find a lower-F mode
        # closer to the true multi-IFO signal.
        self.assertTrue(maxTwoF_BSGLsearch < maxTwoF_Fsearch)
        self.assertTrue(mode_F0_BSGLsearch < mode_F0_Fsearch)
        self.assertTrue(
            np.abs(mode_F0_BSGLsearch - self.F0) < np.abs(mode_F0_Fsearch - self.F0)
        )
        self.assertTrue(maxTwoF_BSGLsearch < self.twoF_predicted)
        self._test_plots()


class TestMCMCSemiCoherentSearch(BaseForMCMCSearchTests):
    label = "TestMCMCSemiCoherentSearch"

    def test_semi_coherent_MCMC(self):
        theta = {
            "F0": {
                "type": "unif",
                "lower": self.F0 - 1e-6,
                "upper": self.F0 + 1e-6,
            },
            "F1": {
                "type": "unif",
                "lower": self.F1 - 1e-10,
                "upper": self.F1 + 1e-10,
            },
            "F2": self.F2,
            "Alpha": self.Alpha,
            "Delta": self.Delta,
        }
        nsegs = 2
        self.search = pyfstat.MCMCSemiCoherentSearch(
            label=self.label,
            outdir=self.outdir,
            theta_prior=theta,
            tref=self.tref,
            sftfilepattern=self.Writer.sftfilepath,
            nsteps=[100, 100],
            nwalkers=100,
            ntemps=2,
            log10beta_min=-1,
            nsegs=nsegs,
        )
        self.search.run(plot_walkers=False)
        self.search.print_summary()

        self._check_twoF_predicted()

        # recover per-segment twoF values at max point
        twoF_sc = self.search.search.get_semicoherent_det_stat(
            self.max_dict["F0"],
            self.max_dict["F1"],
            self.F2,
            self.Alpha,
            self.Delta,
            record_segments=True,
        )
        self.assertTrue(np.abs(twoF_sc - self.maxTwoF) / self.maxTwoF < 0.01)
        twoF_per_seg = np.array(self.search.search.twoF_per_segment)
        self.assertTrue(len(twoF_per_seg) == nsegs)
        twoF_summed = twoF_per_seg.sum()
        self.assertTrue(np.abs(twoF_summed - twoF_sc) / twoF_sc < 0.01)

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
                "lower": self.F0 - 1e-6,
                "upper": self.F0 + 1e-6,
            },
            "F1": {
                "type": "unif",
                "lower": self.F1 - 1e-10,
                "upper": self.F1 + 1e-10,
            },
            "F2": self.F2,
            "Alpha": self.Alpha,
            "Delta": self.Delta,
        }
        nsegs = 10
        NstarMax = 1000
        self.search = pyfstat.MCMCFollowUpSearch(
            label=self.label,
            outdir=self.outdir,
            theta_prior=theta,
            tref=self.tref,
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


class TestMCMCTransientSearch(BaseForMCMCSearchTests):
    label = "TestMCMCTransientSearch"
    duration = 86400

    def setup_method(self, method):
        self.transientWindowType = "rect"
        self.transientStartTime = int(self.tstart + 0.25 * self.duration)
        self.transientTau = int(0.5 * self.duration)
        self.Writer = pyfstat.Writer(
            label=self.label,
            tstart=self.tstart,
            duration=self.duration,
            tref=self.tref,
            **default_signal_params,
            outdir=self.outdir,
            sqrtSX=self.sqrtSX,
            Band=self.Band,
            detectors=self.detectors,
            SFTWindowType=self.SFTWindowType,
            SFTWindowParam=self.SFTWindowParam,
            randSeed=self.randSeed,
            transientWindowType=self.transientWindowType,
            transientStartTime=self.transientStartTime,
            transientTau=self.transientTau,
        )
        self.Writer.make_data(verbose=True)
        self.basic_theta = {
            "F0": self.F0,
            "F1": self.F1,
            "F2": self.F2,
            "Alpha": self.Alpha,
            "Delta": self.Delta,
        }
        self.MCMC_params = {
            "nsteps": [50, 50],
            "nwalkers": 50,
            "ntemps": 2,
            "log10beta_min": -1,
        }

    def test_transient_MCMC_t0only(self):
        theta = {
            **self.basic_theta,
            "transient_tstart": {
                "type": "unif",
                "lower": self.Writer.tstart,
                "upper": self.Writer.tend - 2 * self.Writer.Tsft,
            },
            "transient_duration": self.transientTau,
        }
        self.search = pyfstat.MCMCTransientSearch(
            label=self.label,
            outdir=self.outdir,
            theta_prior=theta,
            tref=self.tref,
            sftfilepattern=self.Writer.sftfilepath,
            **self.MCMC_params,
            transientWindowType=self.transientWindowType,
        )
        self.search.run(plot_walkers=False)
        self.search.print_summary()
        self._check_twoF_predicted()
        self._check_mcmc_quantiles(transient=True)
        self._test_plots()

    def test_transient_MCMC_tauonly(self):
        theta = {
            **self.basic_theta,
            "transient_tstart": self.transientStartTime,
            "transient_duration": {
                "type": "unif",
                "lower": 2 * self.Writer.Tsft,
                "upper": self.Writer.duration - 2 * self.Writer.Tsft,
            },
        }
        self.search = pyfstat.MCMCTransientSearch(
            label=self.label,
            outdir=self.outdir,
            theta_prior=theta,
            tref=self.tref,
            sftfilepattern=self.Writer.sftfilepath,
            **self.MCMC_params,
            transientWindowType=self.transientWindowType,
        )
        self.search.run(plot_walkers=False)
        self.search.print_summary()
        self._check_twoF_predicted()
        self._check_mcmc_quantiles(transient=True)
        self._test_plots()

    def test_transient_MCMC_t0_tau(self, BtSG=False):
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
            label=self.label,
            outdir=self.outdir,
            theta_prior=theta,
            tref=self.tref,
            sftfilepattern=self.Writer.sftfilepath,
            **self.MCMC_params,
            transientWindowType=self.transientWindowType,
            BtSG=BtSG,
        )
        self.search.run(plot_walkers=False)
        self.search.print_summary()
        self._check_twoF_predicted()
        self._check_mcmc_quantiles(transient=True)
        self._test_plots()

    def test_transient_MCMC_t0_tau_BtSG(self, BtSG=False):
        self.test_transient_MCMC_t0_tau(BtSG=True)
