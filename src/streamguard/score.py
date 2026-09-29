"""Block residual and weighted score (ported from ``oes512q-latch``, OES-32 scorer).

Definitions (identical to ``oes32_hilbert_hook.block_score`` in oes512q-latch):

* A block ``x`` is a vector of exactly ``BLOCK = 32`` finite values.
* ``x_ref`` is a reference: one scalar for the whole block or a 32-vector.
* ``scale`` is a positive divisor (``1.0`` in fixed-tau mode; the robust warm-up scale in
  calibrated mode, see :mod:`streamguard.calibrate`).
* Element-wise residual ``r_j = (x_j - x_ref_j) / scale`` for ``j = 0..31``.

All three statistics are computed over the **32 elements of that one residual block**:

* ``Peak    = max_j |r_j|``          (this is the block residual ``R = max|x - x_ref|``, scaled)
* ``RMS     = sqrt(mean_j r_j**2)``  (divides by 32)
* ``MeanAbs = mean_j |r_j|``         (divides by 32)

Score ``S = 0.45*Peak + 0.35*RMS + 0.20*MeanAbs``. Because ``MeanAbs <= RMS <= Peak`` and the
weights sum to 1, ``MeanAbs <= S <= Peak``.

These functions *raise* on invalid input. The fail-closed decision layer that turns invalid
input into a latch with a reason code lives in :mod:`streamguard.latch`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

BLOCK = 32
W_PEAK = 0.45
W_RMS = 0.35
W_MEAN_ABS = 0.20


@dataclass(frozen=True)
class BlockScore:
    """Score of one block and its three components (all over the same 32 residuals)."""

    score: float
    peak: float
    rms: float
    mean_abs: float


def as_block(x: ArrayLike, label: str = "block") -> np.ndarray:
    """Return ``x`` as a float64 vector of length 32, or raise.

    Raises ``TypeError`` for non-numeric / boolean data and ``ValueError`` for a wrong shape
    or a non-finite value.
    """
    arr = np.asarray(x)
    if arr.dtype == np.bool_ or not (
        np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)
    ):
        raise TypeError(f"{label} must contain real numbers, got dtype {arr.dtype}")
    if arr.shape != (BLOCK,):
        raise ValueError(f"{label} must have shape ({BLOCK},), got {arr.shape}")
    arr = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} contains NaN or inf")
    return arr


def residual(x: ArrayLike, x_ref: ArrayLike = 0.0, scale: float = 1.0) -> np.ndarray:
    """Scaled element-wise residual ``(x - x_ref) / scale`` of one 32-value block."""
    xb = as_block(x)
    ref = np.asarray(x_ref, dtype=np.float64)
    if ref.shape not in ((), (BLOCK,)):
        raise ValueError(f"x_ref must be a scalar or shape ({BLOCK},), got {ref.shape}")
    if not np.all(np.isfinite(ref)):
        raise ValueError("x_ref contains NaN or inf")
    s = float(scale)
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError(f"scale must be finite and > 0, got {scale!r}")
    with np.errstate(over="ignore", invalid="ignore"):
        r = (xb - ref) / s
    if not np.all(np.isfinite(r)):
        raise ValueError("residual overflowed to a non-finite value")
    return r


def score_components(r: ArrayLike) -> BlockScore:
    """Weighted score of an already-computed 32-value residual block ``r``."""
    rb = as_block(r, "residual")
    a = np.abs(rb)
    peak = float(a.max())
    with np.errstate(over="ignore"):
        rms = float(np.sqrt(np.mean(rb * rb)))
    mean_abs = float(a.mean())
    score = W_PEAK * peak + W_RMS * rms + W_MEAN_ABS * mean_abs
    return BlockScore(score=score, peak=peak, rms=rms, mean_abs=mean_abs)


def block_score(x: ArrayLike, x_ref: ArrayLike = 0.0, scale: float = 1.0) -> float:
    """``S = 0.45*Peak + 0.35*RMS + 0.20*MeanAbs`` of ``(x - x_ref) / scale``."""
    return score_components(residual(x, x_ref, scale)).score
