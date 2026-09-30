"""Fail-closed admit/latch decision with a reason code.

Rule (ported from ``oes512q-latch``): **latch iff S >= tau** (equality latches). On top of that
rule this module is *fail-closed*: any input that cannot be scored with confidence latches, with
a reason code saying why. Nothing here raises for bad data; exceptions are reserved for
programming errors in the guard configuration.
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike

from streamguard.score import BLOCK, block_score


class Reason(str, Enum):
    """Why a block was admitted or latched. ``OK`` is the only admitting reason."""

    OK = "OK"  # S < tau: admitted
    SCORE_OVER_TAU = "SCORE_OVER_TAU"  # S >= tau: latched
    NAN_INPUT = "NAN_INPUT"  # block contains NaN
    INF_INPUT = "INF_INPUT"  # block contains +inf or -inf
    BAD_SHAPE = "BAD_SHAPE"  # not exactly 32 values in one dimension (includes empty)
    BAD_TYPE = "BAD_TYPE"  # non-numeric, boolean or unparseable values
    NONFINITE_SCORE = "NONFINITE_SCORE"  # finite input but residual/score overflowed
    NO_REFERENCE = "NO_REFERENCE"  # first valid block: seeds the trailing reference
    NOT_CALIBRATED = "NOT_CALIBRATED"  # warm-up in progress, no tau yet
    CALIBRATION_FAILED = "CALIBRATION_FAILED"  # warm-up could not support a tau (e.g. constant)
    BAD_TAU = "BAD_TAU"  # tau is not a finite non-negative number
    SEQUENCE_GAP = "SEQUENCE_GAP"  # seq given and not previous seq + 1 (dropped/reordered)

    def __str__(self) -> str:
        return self.value


def check_block(block: object) -> tuple[np.ndarray | None, Reason]:
    """Validate one block without raising.

    Returns ``(float64 array of shape (32,), Reason.OK)`` or ``(None, <failure reason>)``.
    Check order: type, shape, NaN, inf.
    """
    if block is None or isinstance(block, (str, bytes, dict, set)):
        return None, Reason.BAD_TYPE
    try:
        arr = np.asarray(block)
    except (TypeError, ValueError):  # ragged nested sequences, etc.
        return None, Reason.BAD_SHAPE
    if arr.dtype == np.bool_ or not (
        np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)
    ):
        return None, Reason.BAD_TYPE
    if arr.shape != (BLOCK,):
        return None, Reason.BAD_SHAPE
    arr = arr.astype(np.float64, copy=False)
    if np.isnan(arr).any():
        return None, Reason.NAN_INPUT
    if np.isinf(arr).any():
        return None, Reason.INF_INPUT
    return arr, Reason.OK


def tau_is_valid(tau: object) -> bool:
    """True iff ``tau`` is a finite, non-negative real (not bool)."""
    if isinstance(tau, bool):
        return False
    try:
        t = float(tau)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return math.isfinite(t) and t >= 0.0


def decide(score: float, tau: float) -> tuple[bool, Reason]:
    """Turn a score into ``(admit, reason)``. Fail-closed on a bad score or tau."""
    if not tau_is_valid(tau):
        return False, Reason.BAD_TAU
    if not isinstance(score, (int, float, np.floating, np.integer)) or not math.isfinite(score):
        return False, Reason.NONFINITE_SCORE
    if score >= tau:
        return False, Reason.SCORE_OVER_TAU
    return True, Reason.OK


def latch_bit(
    block: ArrayLike, tau: float = 0.50, x_ref: ArrayLike = 0.0, scale: float = 1.0
) -> tuple[int, Reason]:
    """Stateless helper: ``(1, reason)`` if the block latches, ``(0, Reason.OK)`` if admitted."""
    arr, reason = check_block(block)
    if arr is None:
        return 1, reason
    try:
        s = block_score(arr, x_ref, scale)
    except ValueError:
        return 1, Reason.NONFINITE_SCORE
    admit, reason = decide(s, tau)
    return int(not admit), reason
