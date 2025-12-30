# Migration Guide: Using pytest fixtures instead of unittest.TestCase

This guide demonstrates how to migrate from the legacy `BaseForTestsWithOutdir` and `BaseForTestsWithData` classes to the new pytest fixtures.

## Overview

The `commons_for_tests.py` module now provides pytest fixtures as the recommended approach for test setup:
- `outdir` fixture: Provides a clean output directory for tests
- `data_fixture` fixture: Provides test data with a Writer object and SFTs

The legacy `BaseForTestsWithOutdir` and `BaseForTestsWithData` classes are still available for backward compatibility but are deprecated.

## Example: Using outdir fixture

### Old approach (unittest.TestCase):

```python
from commons_for_tests import BaseForTestsWithOutdir

class TestExample(BaseForTestsWithOutdir):
    label = "TestExample"
    
    def test_something(self):
        # self.outdir is available
        filepath = os.path.join(self.outdir, "test.txt")
        # ... test code ...
```

### New approach (pytest fixtures):

```python
import pytest

@pytest.mark.usefixtures("outdir")
class TestExample:
    outdir = "TestData"  # Optional: override default directory
    label = "TestExample"
    
    def test_something(self):
        # self.outdir is available
        filepath = os.path.join(self.outdir, "test.txt")
        # ... test code ...
```

## Example: Using data_fixture

### Old approach (unittest.TestCase):

```python
from commons_for_tests import BaseForTestsWithData

class TestGridSearch(BaseForTestsWithData):
    label = "TestGridSearch"
    F0s = [29.999, 30.001, 1e-4]
    Band = 0.5
    
    def test_grid_search(self):
        # self.Writer, self.outdir, etc. are available
        search = pyfstat.GridSearch(
            "grid_search",
            self.outdir,
            self.Writer.sftfilepath,
            # ... more params ...
        )
        # ... test code ...
```

### New approach (pytest fixtures):

```python
import pytest

@pytest.mark.usefixtures("data_fixture")
class TestGridSearch:
    outdir = "TestData"
    label = "TestGridSearch"
    F0s = [29.999, 30.001, 1e-4]
    Band = 0.5
    
    def test_grid_search(self):
        # self.Writer, self.outdir, etc. are available
        search = pyfstat.GridSearch(
            "grid_search",
            self.outdir,
            self.Writer.sftfilepath,
            # ... more params ...
        )
        # ... test code ...
```

## Key differences

1. **No inheritance required**: Test classes no longer need to inherit from `unittest.TestCase`
2. **Use `@pytest.mark.usefixtures`**: This decorator tells pytest which fixtures to use
3. **No need for assertions helpers**: Use plain `assert` statements instead of `self.assertTrue()`, `self.assertEqual()`, etc.

## Converting assertions

When migrating, also update unittest-style assertions to pytest-style:

| unittest style | pytest style |
|----------------|--------------|
| `self.assertTrue(x)` | `assert x` |
| `self.assertFalse(x)` | `assert not x` |
| `self.assertEqual(a, b)` | `assert a == b` |
| `self.assertIn(a, b)` | `assert a in b` |
| `self.assertRaises(Exception)` | `pytest.raises(Exception)` |

## Benefits of pytest fixtures

- **Better separation of concerns**: Setup logic is decoupled from test classes
- **More flexible**: Fixtures can be easily composed and reused
- **Better debugging**: pytest provides more informative output
- **Modern standard**: pytest is the modern testing standard for Python
- **No inheritance required**: Simpler test class hierarchies

## Backward compatibility

The legacy `BaseForTestsWithOutdir` and `BaseForTestsWithData` classes will continue to work but will emit deprecation warnings. This allows for gradual migration of existing tests.
