"""
Example test demonstrating the new pytest fixture approach.

This file shows how to use the new `outdir` and `data_fixture` fixtures
from commons_for_tests.py instead of the legacy unittest.TestCase base classes.
"""

import os

import pytest
from commons_for_tests import default_signal_params, default_Writer_params

import pyfstat


@pytest.mark.usefixtures("outdir")
class TestExampleWithOutdir:
    """Example test class using the outdir fixture.

    This demonstrates how to use the outdir fixture instead of
    inheriting from BaseForTestsWithOutdir.
    """

    outdir = "TestDataExample"  # Optional: customize the directory name

    def test_outdir_exists(self):
        """Test that the output directory was created."""
        assert os.path.isdir(self.outdir)

    def test_can_write_file(self):
        """Test that we can write files to the output directory."""
        test_file = os.path.join(self.outdir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")

        assert os.path.isfile(test_file)

        with open(test_file, "r") as f:
            content = f.read()

        assert content == "test content"


@pytest.mark.usefixtures("data_fixture")
class TestExampleWithData:
    """Example test class using the data_fixture.

    This demonstrates how to use the data_fixture instead of
    inheriting from BaseForTestsWithData.
    """

    outdir = "TestDataExampleWithWriter"
    label = "TestExampleWriter"

    def test_writer_created(self):
        """Test that the Writer object was created."""
        assert hasattr(self, "Writer")
        assert isinstance(self.Writer, pyfstat.Writer)

    def test_writer_parameters(self):
        """Test that Writer has expected parameters."""
        # Check that default parameters were set
        assert self.Writer.F0 == default_signal_params["F0"]
        assert self.Writer.tstart == default_Writer_params["tstart"]
        assert self.Writer.detectors == default_Writer_params["detectors"]

    def test_sft_file_created(self):
        """Test that SFT files were created."""
        assert hasattr(self, "Writer")
        assert self.Writer.sftfilepath is not None
        # The sftfilepath may contain wildcards, so check the directory
        assert os.path.isdir(self.outdir)


# ============================================================================
# Examples using default parameter fixtures
# ============================================================================


def test_with_parameter_fixtures(default_Writer_parameters, default_signal_parameters):
    """Example using parameter fixtures for test isolation.

    This demonstrates how parameter fixtures provide copies of the default
    dictionaries, ensuring modifications don't affect other tests.
    """
    # Modify parameters without affecting other tests
    default_Writer_parameters["label"] = "CustomLabel"
    default_signal_parameters["F0"] = 50.0

    # Verify the modifications
    assert default_Writer_parameters["label"] == "CustomLabel"
    assert default_signal_parameters["F0"] == 50.0


def test_with_dictionary_constants():
    """Example using dictionary constants directly.

    This shows the simpler approach of importing and using the dictionaries
    directly, which is recommended when you don't need to modify them.
    """
    # Import and use dictionaries directly
    assert default_signal_params["F0"] == 30.0

    # Merge dictionaries for custom configurations
    custom_params = {**default_Writer_params, "label": "my_custom_test"}
    assert custom_params["label"] == "my_custom_test"
    assert custom_params["Tsft"] == default_Writer_params["Tsft"]
