"""Threshold (tau) calibration: fixed default, warm-up quantile, optional bounded drift.

Ported from ``oes_calibration`` in oes512q-latch ("OES latch v2") and rewritten on numpy:

1. Warm-up residuals ``x - x_ref`` (unscaled) are collected for ``n`` blocks of 32.
2. Robust scale over all ``32*n`` warm-up residual values: ``1.4826 * MAD``; if the MAD is 0
   (mostly constant or count data) fall back to ``sqrt(pi/2) * mean|v - median|``; if that is 0
   as well, calibration fails closed with :class:`CalibrationError`.
3. Each warm-up block is scored on ``(x - x_ref) / scale`` with :func:`streamguard.score`.
4. ``tau`` = the ``quantile`` (default 0.99) of the warm-up block scores, Hyndman-Fan type 7
   (``np.quantile(..., method="linear")``).

Everything is deterministic: no randomness, no labels, independent of block order.

The optional drift rule (:func:`drift_update`) is a bounded stochastic quantile tracker. On
every scored block, ``tau += rate * tau0 * (hit - (1 - quantile))`` where ``hit = S >= tau``,
then ``tau`` is clamped to ``[tau0 * (1 - max_rel), tau0 * (1 + max_rel)]``. At equilibrium the
latch rate tends to ``1 - quantile``; each step moves tau by at most ``rate * tau0`` and tau can
never leave the band around the warm-up value. ``rate = 0`` (the default) disables drift.

The fixed ``tau = 0.50`` of the original contract stays available (``StreamGuard(tau=0.5)``);
it is known to behave poorly on real data, so calibrated mode is the default in StreamGuard.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike

from streamguard.score import BLOCK, score_components

DEFAULT_TAU = 0.50
DEFAULT_QUANTILE = 0.99
MIN_WARMUP_BLOCKS = 4
MAD_TO_SIGMA = 1.4826
MEANAD_TO_SIGMA = math.sqrt(math.pi / 2)  # 1.2533...
METHOD = "warmup-quantile-type7"


class CalibrationError(ValueError):
    """The warm-up data cannot support a calibration (too few blocks, zero spread, ...)."""


def validate_quantile(q: float) -> float:
    """Return ``q`` as float; require a real strictly inside (0, 1)."""
    if isinstance(q, bool) or not isinstance(q, (int, float, np.floating, np.integer)):
        raise TypeError(f"quantile must be a real number, got {type(q).__name__}")
    q = float(q)
    if not (0.0 < q < 1.0):
        raise ValueError(f"quantile must be in the open interval (0, 1), got {q!r}")
    return q


def robust_scale(values: ArrayLike) -> tuple[float, str]:
    """Robust sigma of ``values``: ``(scale, "mad" | "mean_abs_dev")``.

    Raises :class:`CalibrationError` on empty or zero-spread input and ``ValueError`` on
    non-finite input.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        raise CalibrationError("robust scale of an empty sequence")
    if not np.all(np.isfinite(v)):
        raise ValueError("robust scale input contains NaN or inf")
    dev = np.abs(v - np.median(v))
    mad = MAD_TO_SIGMA * float(np.median(dev))
    if math.isfinite(mad) and mad > 0:
        return mad, "mad"
    meanad = MEANAD_TO_SIGMA * math.fsum(dev.tolist()) / dev.size
    if math.isfinite(meanad) and meanad > 0:
        return meanad, "mean_abs_dev"
    raise CalibrationError("zero spread in warm-up residuals (constant data); cannot calibrate")


@dataclass(frozen=True)
class TauCalibration:
    """Result of :func:`calibrate_tau` (immutable)."""

    tau: float
    scale: float
    quantile: float
    n_blocks: int
    method: str = METHOD
    scale_method: str = "mad"

    def as_dict(self) -> dict:
        return asdict(self)


def calibrate_tau(
    warmup_residuals: ArrayLike,
    quantile: float = DEFAULT_QUANTILE,
    scale: str | float = "robust",
    min_blocks: int = MIN_WARMUP_BLOCKS,
) -> TauCalibration:
    """Self-calibrate tau from warm-up residual blocks (shape ``(n, 32)``, unscaled ``x - x_ref``).

    ``scale`` is ``"robust"`` (see module docstring) or a positive finite real used as-is
    (``scale_method = "fixed"``). Fewer than ``min_blocks`` blocks raises
    :class:`CalibrationError`.
    """
    q = validate_quantile(quantile)
    if isinstance(min_blocks, bool) or not isinstance(min_blocks, int) or min_blocks < 1:
        raise ValueError(f"min_blocks must be an int >= 1, got {min_blocks!r}")
    raw = np.asarray(warmup_residuals, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != BLOCK:
        raise ValueError(f"warmup_residuals must have shape (n, {BLOCK}), got {raw.shape}")
    n = raw.shape[0]
    if n < min_blocks:
        raise CalibrationError(f"need at least {min_blocks} warm-up blocks, got {n}")
    if not np.all(np.isfinite(raw)):
        raise ValueError("warm-up residuals contain NaN or inf")

    if isinstance(scale, str):
        if scale != "robust":
            raise ValueError(f"scale must be 'robust' or a positive real, got {scale!r}")
        sigma, scale_method = robust_scale(raw)
    else:
        sigma = float(scale)
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"scale must be finite and > 0, got {scale!r}")
        scale_method = "fixed"

    scores = np.array([score_components(row / sigma).score for row in raw])
    tau = float(np.quantile(scores, q, method="linear"))
    if not math.isfinite(tau) or tau < 0:  # pragma: no cover - defensive
        raise CalibrationError(f"calibrated tau is not a finite non-negative number: {tau!r}")
    return TauCalibration(
        tau=tau, scale=sigma, quantile=q, n_blocks=n, method=METHOD, scale_method=scale_method
    )


@dataclass(frozen=True)
class DriftConfig:
    """Bounded slow drift of tau after calibration. ``rate = 0`` disables drift."""

    rate: float = 0.0
    max_rel: float = 0.25

    def __post_init__(self) -> None:
        for name in ("rate", "max_rel"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{name} must be a finite real, got {v!r}")
        if not (0.0 <= self.rate <= 0.1):
            raise ValueError(f"rate must be in [0, 0.1], got {self.rate!r}")
        if not (0.0 <= self.max_rel < 1.0):
            raise ValueError(f"max_rel must be in [0, 1), got {self.max_rel!r}")

    @property
    def enabled(self) -> bool:
        return self.rate > 0.0 and self.max_rel > 0.0


def drift_update(tau: float, tau0: float, score: float, quantile: float, cfg: DriftConfig) -> float:
    """One bounded drift step (see module docstring). Returns the new tau."""
    if not cfg.enabled or not math.isfinite(score):
        return tau
    hit = 1.0 if score >= tau else 0.0
    new = tau + cfg.rate * tau0 * (hit - (1.0 - quantile))
    lo, hi = tau0 * (1.0 - cfg.max_rel), tau0 * (1.0 + cfg.max_rel)
    return min(max(new, lo), hi)
