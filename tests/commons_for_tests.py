import logging
import os
import shutil
import unittest

import pyfstat


# custom class to allow flaky filtering only on specific excepted exceptions
class FlakyError(Exception):
    pass


# flaky filter function
def is_flaky(err, *args):
    return issubclass(err[0], FlakyError)


class BaseForTestsWithOutdir(unittest.TestCase):
    outdir = "TestData"

    @classmethod
    def setUpClass(self):
        # ensure a clean working directory
        if os.path.isdir(self.outdir):
            try:
                shutil.rmtree(self.outdir)
            except OSError:
                logging.warning("{} not removed prior to tests".format(self.outdir))
        os.makedirs(self.outdir, exist_ok=True)

    @classmethod
    def tearDownClass(self):
        if os.path.isdir(self.outdir):
            try:
                shutil.rmtree(self.outdir)
            except OSError:
                logging.warning("{} not removed after tests".format(self.outdir))


default_Writer_params = {
    "label": "test",
    "sqrtSX": 1,
    "Tsft": 1800,
    "tstart": 700000000,
    "duration": 4 * 1800,
    "detectors": "H1",
    "SFTWindowType": "tukey",
    "SFTWindowParam": 0.001,
    "randSeed": 42,
    "Band": None,
}


default_signal_params_no_sky = {
    "F0": 30.0,
    "F1": -1e-10,
    "F2": 0,
    "h0": 5.0,
    "cosi": 0,
    "psi": 0,
    "phi": 0,
}


default_signal_params = {
    **default_signal_params_no_sky,
    **{"Alpha": 5e-1, "Delta": 1.2},
}


default_binary_params = {
    "period": 45 * 24 * 3600.0,
    "asini": 10.0,
    "tp": default_Writer_params["tstart"] + 0.25 * default_Writer_params["duration"],
    "ecc": 0.5,
    "argp": 0.3,
}


default_transient_params = {
    "transientWindowType": "rect",
    "transientStartTime": default_Writer_params["Tsft"]
    + default_Writer_params["tstart"],
    "transientTau": 2 * default_Writer_params["Tsft"],
}


class BaseForTestsWithData(BaseForTestsWithOutdir):
    outdir = "TestData"

    @classmethod
    def setUpClass(self):
        # ensure a clean working directory
        if os.path.isdir(self.outdir):
            try:
                shutil.rmtree(self.outdir)
            except OSError:
                logging.warning("{} not removed prior to tests".format(self.outdir))
        # skip making outdir, since Writer should do so on first call
        # os.makedirs(self.outdir, exist_ok=True)

        # create fake data SFTs
        # if we directly set any options as self.xy = 1 here,
        # then values set for derived classes may get overwritten,
        # so use a default dict and only insert if no value previous set
        for key, val in {**default_Writer_params, **default_signal_params}.items():
            if not hasattr(self, key):
                setattr(self, key, val)
        self.tref = self.tstart
        self.Writer = pyfstat.Writer(
            label=self.label,
            tstart=self.tstart,
            duration=self.duration,
            tref=self.tref,
            F0=self.F0,
            F1=self.F1,
            F2=self.F2,
            Alpha=self.Alpha,
            Delta=self.Delta,
            h0=self.h0,
            cosi=self.cosi,
            Tsft=self.Tsft,
            outdir=self.outdir,
            sqrtSX=self.sqrtSX,
            Band=self.Band,
            detectors=self.detectors,
            SFTWindowType=self.SFTWindowType,
            SFTWindowParam=self.SFTWindowParam,
            randSeed=self.randSeed,
        )
        self.Writer.make_data(verbose=True)
        self.search_keys = ["F0", "F1", "F2", "Alpha", "Delta"]
        self.search_ranges = {key: [getattr(self, key)] for key in self.search_keys}
