import importlib.util

import numpy as np
import pytest

SEED = 42
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "sklearn: needs the optional scikit-learn extra (skipped when it is missing)"
    )


def pytest_collection_modifyitems(config, items):
    if HAS_SKLEARN:
        return
    skip = pytest.mark.skip(reason="optional scikit-learn is not installed")
    for item in items:
        if "sklearn" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


def gaussian_blocks(rng, n, loc=0.0, sd=1.0):
    return rng.normal(loc, sd, size=(n, 32))
