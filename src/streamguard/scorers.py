"""Pluggable block scorers. The gate (validation, reason codes, calibration, audit log) is the
same for every scorer; a scorer only turns valid 32-value blocks into one number.

Scorer protocol (duck-typed; see :class:`Scorer`):

``start(warmup_blocks, scale)``
    Allocate bounded warm-up state (calibrated mode).
``warmup(x) -> Reason``
    Consume one valid warm-up block. Returns ``Reason.NOT_CALIBRATED`` if the block counted
    towards the warm-up, ``Reason.NO_REFERENCE`` if it only seeded history, or
    ``Reason.NONFINITE_SCORE`` if its residual overflowed.
``fit(quantile) -> TauCalibration``
    Finish the warm-up: fix the scale/model, compute the warm-up block scores
    (``warmup_scores``, kept for audit) and ``tau`` = their ``quantile`` (numpy ``linear``).
    Raises :class:`CalibrationError` if the warm-up cannot support a threshold; the scorer then
    clears its warm-up buffer. Releases the warm-up buffer on success.
``fit_fixed(scale)``
    Fixed-tau mode: no warm-up; ``scale`` (default 1.0) is used as-is.
``score(x) -> (score, peak) | Reason``
    Score one valid block after fitting and update the state. Returns a ``Reason`` instead when
    the block cannot be scored (``NO_REFERENCE`` after a reset, ``NONFINITE_SCORE`` on overflow).
``on_invalid()``
    Called when a block fails validation or a sequence gap is detected.
``params() -> dict``
    JSON-safe parameters, written to the audit log's config record.

Built-ins, all with O(1) or bounded state per stream:

``spark``      the original residual score ``0.45*Peak + 0.35*RMS + 0.20*MeanAbs`` against a
               reference model (``diff``/``detrend``/``ewma``/``median``/fixed).
``ewma``       EWMA control chart on the trailing-median residual: ``z_t = lam*r_t +
               (1-lam)*z_{t-1}``, block score ``max |z_t| / sqrt(lam/(2-lam))``. Numpy only.
``isoforest``  scikit-learn IsolationForest fitted on the warm-up residual blocks only
               (optional ``bench``/``isoforest`` extra). Model size is fixed by the number of
               trees and ``min(256, warmup_blocks)`` samples per tree.

``ewma`` and ``isoforest`` use the residual of the M4 benchmark baselines: the block minus the
median of the trailing ``ref_window`` samples, divided by the robust warm-up scale.
"""

from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

import numpy as np

from streamguard.calibrate import (
    METHOD,
    CalibrationError,
    TauCalibration,
    calibrate_tau,
    robust_scale,
)
from streamguard.latch import Reason
from streamguard.reference import DEFAULT_REFERENCE, make_reference
from streamguard.score import BLOCK, score_components

ScoreResult = tuple[float, float | None] | Reason
_OFFSETS = np.arange(BLOCK)


@runtime_checkable
class Scorer(Protocol):
    name: str

    def start(self, warmup_blocks: int, scale: float | None) -> None: ...
    def warmup(self, x: np.ndarray) -> Reason: ...
    def fit(self, quantile: float) -> TauCalibration: ...
    def fit_fixed(self, scale: float | None) -> None: ...
    def score(self, x: np.ndarray) -> ScoreResult: ...
    def on_invalid(self) -> None: ...
    def params(self) -> dict[str, Any]: ...


def quantile_tau(scores: np.ndarray, quantile: float) -> float:
    with np.errstate(invalid="ignore", over="ignore"):
        tau = float(np.quantile(scores, quantile, method="linear"))
    if not math.isfinite(tau) or tau < 0:
        raise CalibrationError(f"warm-up scores give no finite non-negative tau: {tau!r}")
    return tau


class SparkScorer:
    """Residual score against a reference model (the original StreamGuard score)."""

    name = "spark"
    __slots__ = ("_ref", "_scale_fixed", "_wu", "_n", "scale", "warmup_scores", "_cfg")

    def __init__(
        self, reference: Any = DEFAULT_REFERENCE, ref_blocks: int = 4, ewma_alpha: float = 0.05
    ) -> None:
        self._ref = make_reference(reference, ref_blocks, ewma_alpha)
        cfg: dict[str, Any] = {"reference": self._ref.name, **self._ref.config()}
        self._cfg = cfg
        self._scale_fixed: float | None = None
        self._wu: np.ndarray | None = None
        self._n = 0
        self.scale: float | None = None
        self.warmup_scores: np.ndarray | None = None

    def start(self, warmup_blocks: int, scale: float | None) -> None:
        self._wu = np.empty((warmup_blocks, BLOCK), dtype=np.float64)
        self._n = 0
        self._scale_fixed = scale

    def _raw(self, x: np.ndarray) -> np.ndarray | Reason:
        ref = self._ref.ref(x)
        if ref is None:
            self._ref.push(x)
            return Reason.NO_REFERENCE
        with np.errstate(over="ignore", invalid="ignore"):
            raw = x - ref
        self._ref.push(x)
        if not np.all(np.isfinite(raw)):
            return Reason.NONFINITE_SCORE
        return raw

    def warmup(self, x: np.ndarray) -> Reason:
        raw = self._raw(x)
        if isinstance(raw, Reason):
            return raw
        self._wu[self._n] = raw
        self._n += 1
        return Reason.NOT_CALIBRATED

    def fit(self, quantile: float) -> TauCalibration:
        wu = self._wu[: self._n]
        scale_arg: str | float = "robust" if self._scale_fixed is None else self._scale_fixed
        try:
            cal = calibrate_tau(wu, quantile=quantile, scale=scale_arg, min_blocks=self._n)
        except (CalibrationError, ValueError) as exc:
            self._n = 0
            raise CalibrationError(str(exc)) from exc
        self.scale = cal.scale
        self.warmup_scores = np.array([score_components(r / cal.scale).score for r in wu])
        self._wu = None
        return cal

    def fit_fixed(self, scale: float | None) -> None:
        self.scale = 1.0 if scale is None else scale

    def score(self, x: np.ndarray) -> ScoreResult:
        raw = self._raw(x)
        if isinstance(raw, Reason):
            return raw
        with np.errstate(over="ignore", invalid="ignore"):
            r = raw / self.scale
        if not np.all(np.isfinite(r)):
            return Reason.NONFINITE_SCORE
        bs = score_components(r)
        return bs.score, bs.peak

    def on_invalid(self) -> None:
        self._ref.on_invalid()

    def params(self) -> dict[str, Any]:
        return dict(self._cfg)


class _TrailingMedianScorer:
    """Shared state for scorers on the trailing-median residual (bounded ring buffer)."""

    __slots__ = (
        "_buf",
        "_count",
        "_pos",
        "_wu",
        "_n",
        "_scale_fixed",
        "scale",
        "warmup_scores",
        "scale_method",
        "ref_window",
    )

    def __init__(self, ref_window: int) -> None:
        if isinstance(ref_window, bool) or not isinstance(ref_window, int) or ref_window < BLOCK:
            raise ValueError(f"ref_window must be an int >= {BLOCK}, got {ref_window!r}")
        self.ref_window = ref_window
        self._buf = np.empty(ref_window, dtype=np.float64)
        self._count = 0
        self._pos = 0
        self._wu: np.ndarray | None = None
        self._n = 0
        self._scale_fixed: float | None = None
        self.scale: float | None = None
        self.warmup_scores: np.ndarray | None = None
        self.scale_method = "fixed"

    def _push(self, x: np.ndarray) -> None:
        idx = (self._pos + _OFFSETS) % self.ref_window
        self._buf[idx] = x
        self._pos = (self._pos + BLOCK) % self.ref_window
        self._count = min(self._count + BLOCK, self.ref_window)

    def _raw(self, x: np.ndarray) -> np.ndarray | Reason:
        if self._count == 0:
            self._push(x)
            return Reason.NO_REFERENCE
        with np.errstate(over="ignore", invalid="ignore"):
            ref = float(np.median(self._buf[: self._count]))
        with np.errstate(over="ignore", invalid="ignore"):
            raw = x - ref
        self._push(x)
        if not np.all(np.isfinite(raw)):
            return Reason.NONFINITE_SCORE
        return raw

    def start(self, warmup_blocks: int, scale: float | None) -> None:
        self._wu = np.empty((warmup_blocks, BLOCK), dtype=np.float64)
        self._n = 0
        self._scale_fixed = scale

    def warmup(self, x: np.ndarray) -> Reason:
        raw = self._raw(x)
        if isinstance(raw, Reason):
            return raw
        self._wu[self._n] = raw
        self._n += 1
        return Reason.NOT_CALIBRATED

    def _fit_scale(self) -> np.ndarray:
        wu = self._wu[: self._n]
        if self._scale_fixed is None:
            try:
                self.scale, self.scale_method = robust_scale(wu)
            except CalibrationError:
                self._n = 0
                raise
        else:
            self.scale, self.scale_method = self._scale_fixed, "fixed"
        with np.errstate(over="ignore", invalid="ignore"):
            r = wu / self.scale
        if not np.all(np.isfinite(r)):
            self._n = 0
            raise CalibrationError("scaled warm-up residuals are not finite")
        return r

    def _calibration(self, quantile: float) -> TauCalibration:
        try:
            tau = quantile_tau(self.warmup_scores, quantile)
        except CalibrationError:
            self._n = 0
            raise
        n = self._n
        self._wu = None
        return TauCalibration(
            tau=tau,
            scale=self.scale,
            quantile=quantile,
            n_blocks=n,
            method=METHOD,
            scale_method=self.scale_method,
        )

    def on_invalid(self) -> None:
        pass  # trailing history and state continue across the gap (documented)

    def _scaled(self, x: np.ndarray) -> np.ndarray | Reason:
        raw = self._raw(x)
        if isinstance(raw, Reason):
            return raw
        with np.errstate(over="ignore", invalid="ignore"):
            r = raw / self.scale
        if not np.all(np.isfinite(r)):
            return Reason.NONFINITE_SCORE
        return r


class EwmaScorer(_TrailingMedianScorer):
    """Streaming EWMA control chart on the trailing-median residual (numpy only)."""

    name = "ewma"
    __slots__ = ("lam", "_z", "_W", "_d", "_c")

    def __init__(self, lam: float = 0.2, ref_window: int = 288) -> None:
        super().__init__(ref_window)
        lam = float(lam)
        if not (0.0 < lam <= 1.0):
            raise ValueError(f"lam must be in (0, 1], got {lam!r}")
        self.lam = lam
        j = np.arange(BLOCK)
        # z_j = d_j * z_prev + sum_k W[j, k] * r_k  (closed form of the recursion in a block)
        diff = j[:, None] - j[None, :]
        self._W = np.where(diff >= 0, lam * (1.0 - lam) ** np.maximum(diff, 0), 0.0)
        self._d = (1.0 - lam) ** (j + 1)
        self._c = math.sqrt(lam / (2.0 - lam))
        self._z = 0.0

    def _block(self, r: np.ndarray) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            z = self._W @ r + self._d * self._z
            self._z = float(z[-1])
            return float(np.abs(z).max() / self._c)

    def fit(self, quantile: float) -> TauCalibration:
        r = self._fit_scale()
        self._z = 0.0
        self.warmup_scores = np.array([self._block(row) for row in r])
        return self._calibration(quantile)

    def fit_fixed(self, scale: float | None) -> None:
        self.scale = 1.0 if scale is None else scale
        self._z = 0.0

    def score(self, x: np.ndarray) -> ScoreResult:
        r = self._scaled(x)
        if isinstance(r, Reason):
            return r
        s = self._block(r)
        if not math.isfinite(s):
            self._z = 0.0  # do not let an overflow poison later blocks
            return Reason.NONFINITE_SCORE
        return s, float(np.abs(r).max())

    def params(self) -> dict[str, Any]:
        return {"lam": self.lam, "ref_window": self.ref_window}


class IsoForestScorer(_TrailingMedianScorer):
    """IsolationForest fitted on the warm-up residual blocks only (needs scikit-learn)."""

    name = "isoforest"
    __slots__ = ("n_estimators", "seed", "_model")

    def __init__(self, n_estimators: int = 100, seed: int = 42, ref_window: int = 288) -> None:
        super().__init__(ref_window)
        if isinstance(n_estimators, bool) or not isinstance(n_estimators, int) or n_estimators < 1:
            raise ValueError("n_estimators must be an int >= 1")
        try:
            import sklearn.ensemble  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "the isoforest scorer needs scikit-learn: "
                "pip install 'spark-streamguard[isoforest]'"
            ) from exc
        self.n_estimators = n_estimators
        self.seed = int(seed)
        self._model = None

    def fit(self, quantile: float) -> TauCalibration:
        from sklearn.ensemble import IsolationForest

        r = self._fit_scale()
        self._model = IsolationForest(n_estimators=self.n_estimators, random_state=self.seed)
        self._model.fit(r)
        self.warmup_scores = -self._model.score_samples(r)
        return self._calibration(quantile)

    def fit_fixed(self, scale: float | None) -> None:
        raise ValueError("the isoforest scorer needs a warm-up (use tau=None)")

    def score(self, x: np.ndarray) -> ScoreResult:
        r = self._scaled(x)
        if isinstance(r, Reason):
            return r
        return float(-self._model.score_samples(r[None, :])[0]), float(np.abs(r).max())

    def params(self) -> dict[str, Any]:
        return {"n_estimators": self.n_estimators, "seed": self.seed, "ref_window": self.ref_window}


SCORERS = {"spark": SparkScorer, "ewma": EwmaScorer, "isoforest": IsoForestScorer}


def make_scorer(name: str, **params: Any) -> Scorer:
    """Build a built-in scorer by name."""
    if name not in SCORERS:
        raise ValueError(f"unknown scorer {name!r}; expected one of {sorted(SCORERS)}")
    return SCORERS[name](**params)
