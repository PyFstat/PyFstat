"""
Example test demonstrating the new pytest fixture approach.

This file shows how to use the new `outdir` and `data_fixture` fixtures
from commons_for_tests.py instead of the legacy unittest.TestCase base classes.
"""

import os

import pytest

from commons_for_tests import (
    data_fixture,  # noqa: F401 - imported for fixture usage
    default_signal_params,
    default_Writer_params,
    outdir,  # noqa: F401 - imported for fixture usage
)

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
    
    def test_search_keys_available(self):
        """Test that search keys and ranges are available."""
        assert hasattr(self, "search_keys")
        assert hasattr(self, "search_ranges")
        assert "F0" in self.search_keys
        assert "F0" in self.search_ranges


# You can also use fixtures directly in test functions (not just class methods)
def test_with_outdir_function(outdir):
    """Example function test using the outdir fixture directly."""
    # Note: For function-scoped fixtures, you may want to define
    # a separate fixture with scope="function" instead of "class"
    assert os.path.isdir(outdir)
