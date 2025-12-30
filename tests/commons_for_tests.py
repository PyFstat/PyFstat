import logging
import os
import shutil
import unittest

import pytest

import pyfstat


# custom class to allow flaky filtering only on specific excepted exceptions
class FlakyError(Exception):
    pass


# flaky filter function
def is_flaky(err, *args):
    return issubclass(err[0], FlakyError)


# ============================================================================
# Pytest fixtures (recommended approach)
# ============================================================================


@pytest.fixture(scope="class")
def outdir(request):
    """Pytest fixture that provides a clean output directory for tests.

    This fixture creates a test data directory before tests run and cleans it up
    afterwards. The directory name can be customized by setting an 'outdir'
    attribute on the test class.

    Yields:
        str: Path to the output directory
    """
    # Get outdir from test class if it exists, otherwise use default
    test_outdir = getattr(request.cls, "outdir", "TestData")

    # Ensure a clean working directory
    if os.path.isdir(test_outdir):
        try:
            shutil.rmtree(test_outdir)
        except OSError:
            logging.warning("{} not removed prior to tests".format(test_outdir))
    os.makedirs(test_outdir, exist_ok=True)

    # Store outdir on the test class instance if it exists
    if request.cls is not None:
        request.cls.outdir = test_outdir

    yield test_outdir

    # Cleanup after tests
    if os.path.isdir(test_outdir):
        try:
            shutil.rmtree(test_outdir)
        except OSError:
            logging.warning("{} not removed after tests".format(test_outdir))


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


# ============================================================================
# Pytest fixtures for default parameters (recommended for new tests)
# ============================================================================


@pytest.fixture
def default_Writer_parameters():
    """Fixture providing default Writer parameters.

    Returns a copy of the default_Writer_params dictionary.
    """
    return default_Writer_params.copy()


@pytest.fixture
def default_signal_parameters_no_sky():
    """Fixture providing default signal parameters without sky location.

    Returns a copy of the default_signal_params_no_sky dictionary.
    """
    return default_signal_params_no_sky.copy()


@pytest.fixture
def default_signal_parameters():
    """Fixture providing default signal parameters.

    Returns a copy of the default_signal_params dictionary.
    """
    return default_signal_params.copy()


@pytest.fixture
def default_binary_parameters():
    """Fixture providing default binary parameters.

    Returns a copy of the default_binary_params dictionary.
    """
    return default_binary_params.copy()


@pytest.fixture
def default_transient_parameters():
    """Fixture providing default transient parameters.

    Returns a copy of the default_transient_params dictionary.
    """
    return default_transient_params.copy()


# ============================================================================
# Pytest fixtures for test setup
# ============================================================================


@pytest.fixture(scope="class")
def data_fixture(request, outdir):
    """Pytest fixture that provides test data with a Writer object and SFTs.

    This fixture creates fake data SFTs using the pyfstat.Writer class. It reads
    parameters from the test class attributes (with defaults) and creates a Writer
    object that generates synthetic signal data.

    The fixture makes the Writer object and related parameters available as class
    attributes so they can be accessed in test methods.

    This fixture is designed for use with class-based tests only.

    Args:
        request: pytest request object
        outdir: Output directory from the outdir fixture

    Yields:
        The test class instance with Writer and related attributes set
    """
    # Skip making outdir, since Writer should do so on first call
    # Note: outdir fixture already handles directory creation

    # Get test class - this fixture requires a class-based test
    test_cls = request.cls
    if test_cls is None:
        raise ValueError(
            "data_fixture is designed for class-based tests. "
            "Use @pytest.mark.usefixtures('data_fixture') on a test class."
        )

    # Create fake data SFTs
    # If we directly set any options as self.xy = 1 here,
    # then values set for derived classes may get overwritten,
    # so use a default dict and only insert if no value previously set
    params = {**default_Writer_params, **default_signal_params}
    for key, val in params.items():
        if not hasattr(test_cls, key):
            setattr(test_cls, key, val)

    test_cls.tref = test_cls.tstart
    test_cls.Writer = pyfstat.Writer(
        label=test_cls.label,
        tstart=test_cls.tstart,
        duration=test_cls.duration,
        tref=test_cls.tref,
        F0=test_cls.F0,
        F1=test_cls.F1,
        F2=test_cls.F2,
        Alpha=test_cls.Alpha,
        Delta=test_cls.Delta,
        h0=test_cls.h0,
        cosi=test_cls.cosi,
        Tsft=test_cls.Tsft,
        outdir=test_cls.outdir,
        sqrtSX=test_cls.sqrtSX,
        Band=test_cls.Band,
        detectors=test_cls.detectors,
        SFTWindowType=test_cls.SFTWindowType,
        SFTWindowParam=test_cls.SFTWindowParam,
        randSeed=test_cls.randSeed,
    )
    test_cls.Writer.make_data(verbose=True)
    test_cls.search_keys = ["F0", "F1", "F2", "Alpha", "Delta"]
    test_cls.search_ranges = {
        key: [getattr(test_cls, key)] for key in test_cls.search_keys
    }

    yield test_cls


# ============================================================================
# Legacy unittest.TestCase base classes (deprecated, use fixtures instead)
# ============================================================================


class BaseForTestsWithOutdir(unittest.TestCase):
    """Legacy base class for tests requiring an output directory.

    .. deprecated::
        Use the `outdir` pytest fixture instead.
    """

    outdir = "TestData"

    @classmethod
    def setUpClass(cls):
        logging.warning(
            "BaseForTestsWithOutdir is deprecated. "
            "Use the 'outdir' pytest fixture instead."
        )
        # ensure a clean working directory
        if os.path.isdir(cls.outdir):
            try:
                shutil.rmtree(cls.outdir)
            except OSError:
                logging.warning("{} not removed prior to tests".format(cls.outdir))
        os.makedirs(cls.outdir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls.outdir):
            try:
                shutil.rmtree(cls.outdir)
            except OSError:
                logging.warning("{} not removed after tests".format(cls.outdir))


class BaseForTestsWithData(BaseForTestsWithOutdir):
    """Legacy base class for tests requiring test data with SFTs.

    .. deprecated::
        Use the `data_fixture` pytest fixture instead.
    """

    outdir = "TestData"

    @classmethod
    def setUpClass(cls):
        logging.warning(
            "BaseForTestsWithData is deprecated. "
            "Use the 'data_fixture' pytest fixture instead."
        )
        # ensure a clean working directory
        if os.path.isdir(cls.outdir):
            try:
                shutil.rmtree(cls.outdir)
            except OSError:
                logging.warning("{} not removed prior to tests".format(cls.outdir))
        # skip making outdir, since Writer should do so on first call
        # os.makedirs(cls.outdir, exist_ok=True)

        # create fake data SFTs
        # if we directly set any options as self.xy = 1 here,
        # then values set for derived classes may get overwritten,
        # so use a default dict and only insert if no value previous set
        for key, val in {**default_Writer_params, **default_signal_params}.items():
            if not hasattr(cls, key):
                setattr(cls, key, val)
        cls.tref = cls.tstart
        cls.Writer = pyfstat.Writer(
            label=cls.label,
            tstart=cls.tstart,
            duration=cls.duration,
            tref=cls.tref,
            F0=cls.F0,
            F1=cls.F1,
            F2=cls.F2,
            Alpha=cls.Alpha,
            Delta=cls.Delta,
            h0=cls.h0,
            cosi=cls.cosi,
            Tsft=cls.Tsft,
            outdir=cls.outdir,
            sqrtSX=cls.sqrtSX,
            Band=cls.Band,
            detectors=cls.detectors,
            SFTWindowType=cls.SFTWindowType,
            SFTWindowParam=cls.SFTWindowParam,
            randSeed=cls.randSeed,
        )
        cls.Writer.make_data(verbose=True)
        cls.search_keys = ["F0", "F1", "F2", "Alpha", "Delta"]
        cls.search_ranges = {key: [getattr(cls, key)] for key in cls.search_keys}
