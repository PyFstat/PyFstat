# Pytest Fixtures for PyFstat Tests

This module provides pytest fixtures for test setup and teardown, replacing
the legacy unittest.TestCase base classes.

## Fixtures

### outdir
**Scope:** class

Provides a clean output directory for tests. The directory is created
before tests run and cleaned up afterwards. The directory name can be
customized by setting an 'outdir' attribute on the test class.

**Example usage:**
```python
@pytest.mark.usefixtures("outdir")
class TestExample:
    outdir = "MyTestData"  # Optional: customize directory name
    
    def test_something(self):
        # self.outdir is available
        filepath = os.path.join(self.outdir, "test.txt")
```

### data_fixture
**Scope:** class

Provides test data with a pyfstat.Writer object and SFTs. This fixture
automatically sets up Writer parameters from class attributes (with sensible
defaults) and generates synthetic signal data.

**Example usage:**
```python
@pytest.mark.usefixtures("data_fixture")
class TestExample:
    outdir = "TestData"
    label = "TestExample"
    Band = 0.5  # Optional: override default parameters
    
    def test_something(self):
        # self.Writer, self.outdir, self.search_keys, etc. are available
        assert self.Writer.F0 == 30.0
```

## Legacy Classes (Deprecated)

### BaseForTestsWithOutdir
**Type:** unittest.TestCase (deprecated)

Legacy base class for tests requiring an output directory.
Use the `outdir` fixture instead.

### BaseForTestsWithData
**Type:** unittest.TestCase (deprecated)

Legacy base class for tests requiring test data with SFTs.
Use the `data_fixture` fixture instead.

The legacy classes are maintained for backward compatibility but will emit
deprecation warnings. New tests should use the pytest fixtures.

## Migration

See `FIXTURE_MIGRATION_GUIDE.md` for detailed migration instructions and examples.

## Default Parameters

The module also exports several default parameter dictionaries:
- `default_Writer_params`: Default parameters for pyfstat.Writer
- `default_signal_params`: Default signal parameters
- `default_signal_params_no_sky`: Signal parameters without sky location
- `default_binary_params`: Binary system parameters
- `default_transient_params`: Transient signal parameters

## Utilities

### FlakyError
**Type:** Exception

Custom exception class for flaky test filtering.

### is_flaky
**Type:** function

Filter function for identifying flaky test errors.
