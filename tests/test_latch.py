"""Fail-closed admit/latch decisions and reason codes."""

import numpy as np
import pytest

from streamguard.latch import Reason, check_block, decide, latch_bit, tau_is_valid

NAN = float("nan")
INF = float("inf")


def test_reason_values_are_strings():
    assert str(Reason.OK) == "OK"
    assert Reason.SCORE_OVER_TAU.value == "SCORE_OVER_TAU"
    assert {r.value for r in Reason} >= {
        "OK",
        "SCORE_OVER_TAU",
        "NAN_INPUT",
        "INF_INPUT",
        "BAD_SHAPE",
        "BAD_TYPE",
        "NOT_CALIBRATED",
        "NO_REFERENCE",
        "CALIBRATION_FAILED",
        "NONFINITE_SCORE",
        "BAD_TAU",
    }


def test_admit_below_tau_latch_at_and_above():
    assert decide(0.49, 0.5) == (True, Reason.OK)
    assert decide(0.5, 0.5) == (False, Reason.SCORE_OVER_TAU)  # equality latches
    assert decide(3.0, 0.5) == (False, Reason.SCORE_OVER_TAU)


@pytest.mark.parametrize("score", [NAN, INF, -INF, None, "0.1"])
def test_bad_score_latches(score):
    assert decide(score, 0.5) == (False, Reason.NONFINITE_SCORE)


@pytest.mark.parametrize("tau", [NAN, INF, -0.1, None, True, "x"])
def test_bad_tau_latches(tau):
    assert decide(0.0, tau) == (False, Reason.BAD_TAU)
    assert not tau_is_valid(tau)


def test_tau_zero_latches_everything():
    assert decide(0.0, 0.0) == (False, Reason.SCORE_OVER_TAU)


@pytest.mark.parametrize(
    "block, reason",
    [
        ([NAN] + [0.0] * 31, Reason.NAN_INPUT),
        ([0.0] * 31 + [INF], Reason.INF_INPUT),
        ([-INF] * 32, Reason.INF_INPUT),
        ([NAN, INF] + [0.0] * 30, Reason.NAN_INPUT),
        ([], Reason.BAD_SHAPE),
        ([0.0] * 31, Reason.BAD_SHAPE),
        ([0.0] * 33, Reason.BAD_SHAPE),
        (np.zeros((2, 16)), Reason.BAD_SHAPE),
        (np.float64(1.0), Reason.BAD_SHAPE),
        ([[1.0], [1.0, 2.0]], Reason.BAD_SHAPE),  # ragged
        (None, Reason.BAD_TYPE),
        ("0" * 32, Reason.BAD_TYPE),
        (b"x" * 32, Reason.BAD_TYPE),
        ({"a": 1}, Reason.BAD_TYPE),
        ([True] * 32, Reason.BAD_TYPE),
        (["1"] * 32, Reason.BAD_TYPE),
        ([None] * 32, Reason.BAD_TYPE),
        ([1.0] * 31 + [None], Reason.BAD_TYPE),
        ([1j] * 32, Reason.BAD_TYPE),
    ],
)
def test_invalid_blocks_fail_closed(block, reason):
    arr, got = check_block(block)
    assert arr is None
    assert got is reason
    bit, got2 = latch_bit(block)
    assert bit == 1 and got2 is reason


def test_valid_block_passes_check():
    arr, reason = check_block(list(range(32)))
    assert reason is Reason.OK
    assert arr.dtype == np.float64 and arr.shape == (32,)


def test_latch_bit_default_tau():
    assert latch_bit([0.01] * 32) == (0, Reason.OK)
    spike = [0.0] * 32
    spike[0] = 1.0  # S = 0.518 >= 0.50
    assert latch_bit(spike) == (1, Reason.SCORE_OVER_TAU)
    assert latch_bit(spike, tau=0.6) == (0, Reason.OK)


def test_latch_bit_overflow_and_bad_scale_fail_closed():
    assert latch_bit([1e308] * 32, x_ref=-1e308) == (1, Reason.NONFINITE_SCORE)
    assert latch_bit([0.0] * 32, scale=0.0) == (1, Reason.NONFINITE_SCORE)


def test_latch_bit_bad_tau():
    assert latch_bit([0.0] * 32, tau=NAN) == (1, Reason.BAD_TAU)
