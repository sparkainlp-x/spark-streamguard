import numpy as np
import pytest

SEED = 42


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


def gaussian_blocks(rng, n, loc=0.0, sd=1.0):
    return rng.normal(loc, sd, size=(n, 32))
