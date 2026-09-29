"""Scoring correctness on hand-computed examples."""

import math

import numpy as np
import pytest

from streamguard.score import (
    BLOCK,
    W_MEAN_ABS,
    W_PEAK,
    W_RMS,
    as_block,
    block_score,
    residual,
    score_components,
)


def test_weights_sum_to_one():
    assert (W_PEAK, W_RMS, W_MEAN_ABS) == (0.45, 0.35, 0.20)
    assert W_PEAK + W_RMS + W_MEAN_ABS == pytest.approx(1.0)
    assert BLOCK == 32


def test_zero_block_scores_zero():
    bs = score_components(np.zeros(32))
    assert (bs.score, bs.peak, bs.rms, bs.mean_abs) == (0.0, 0.0, 0.0, 0.0)


def test_constant_block_score_equals_value():
    # |r_j| = 0.01 everywhere -> Peak = RMS = MeanAbs = 0.01 -> S = 0.01
    assert block_score([0.01] * 32) == pytest.approx(0.01)


def test_plus_minus_one_block():
    # 16 x +1, 16 x -1: Peak = 1, RMS = 1, MeanAbs = 1 -> S = 1
    bs = score_components([1.0] * 16 + [-1.0] * 16)
    assert (bs.peak, bs.rms, bs.mean_abs) == (1.0, 1.0, 1.0)
    assert bs.score == pytest.approx(1.0)


def test_single_unit_spike():
    # Peak = 1, RMS = sqrt(1/32), MeanAbs = 1/32
    # S = 0.45 + 0.35*0.1767767 + 0.20*0.03125 = 0.5181218
    x = np.zeros(32)
    x[7] = -1.0
    bs = score_components(x)
    assert bs.peak == 1.0
    assert bs.rms == pytest.approx(math.sqrt(1 / 32))
    assert bs.mean_abs == pytest.approx(1 / 32)
    assert bs.score == pytest.approx(0.45 + 0.35 * math.sqrt(1 / 32) + 0.20 / 32)
    assert bs.score == pytest.approx(0.5181218, abs=1e-7)


def test_burst_example_from_oes512q_latch_demo():
    # oes512q-latch demo "burst": 31 x 0.01 and one 2.0.
    # Peak = 2, MeanAbs = 2.31/32 = 0.0721875, RMS = sqrt(4.0031/32) = 0.3536903
    # S = 0.9 + 0.35*0.3536903 + 0.2*0.0721875 = 1.0382291
    x = [0.01] * 32
    x[0] = 2.0
    assert block_score(x) == pytest.approx(1.0382291, abs=1e-7)


def test_residual_with_vector_reference_and_scale():
    x = np.full(32, 2.0)
    ref = np.ones(32)
    r = residual(x, ref, scale=2.0)
    np.testing.assert_allclose(r, 0.5)
    assert block_score(x, ref, 2.0) == pytest.approx(0.5)


def test_peak_is_max_abs_residual():
    x = np.arange(32, dtype=float)
    bs = score_components(residual(x, 10.0))
    assert bs.peak == 21.0  # max |x - 10| = |31 - 10|


def test_score_between_meanabs_and_peak(rng):
    for _ in range(50):
        bs = score_components(rng.normal(size=32) * rng.uniform(0.1, 10))
        assert bs.mean_abs <= bs.rms <= bs.peak
        assert bs.mean_abs - 1e-12 <= bs.score <= bs.peak + 1e-12


def test_score_sign_invariant(rng):
    x = rng.normal(size=32)
    assert block_score(x) == pytest.approx(block_score(-x))


def test_integer_input_accepted():
    assert block_score(list(range(32)), 0) == block_score(np.arange(32.0))


@pytest.mark.parametrize(
    "bad, exc",
    [
        ([0.0] * 31, ValueError),
        ([0.0] * 33, ValueError),
        ([], ValueError),
        (np.zeros((4, 8)), ValueError),
        ([float("nan")] + [0.0] * 31, ValueError),
        ([float("inf")] + [0.0] * 31, ValueError),
        ([True] * 32, TypeError),
        (["a"] * 32, TypeError),
    ],
)
def test_as_block_rejects(bad, exc):
    with pytest.raises(exc):
        as_block(bad)


@pytest.mark.parametrize(
    "ref, scale",
    [
        (np.zeros(31), 1.0),
        (float("nan"), 1.0),
        (0.0, 0.0),
        (0.0, -1.0),
        (0.0, float("inf")),
    ],
)
def test_residual_rejects_bad_ref_or_scale(ref, scale):
    with pytest.raises(ValueError):
        residual(np.zeros(32), ref, scale)


def test_residual_overflow_raises():
    x = np.full(32, 1e308)
    with pytest.raises(ValueError, match="overflow"):
        residual(x, -1e308)
