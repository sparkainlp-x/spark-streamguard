"""StreamGuard: per-stream state machine with O(1) memory, plus a JSONL audit log writer.

Per stream, the guard keeps only fixed-size state allocated in ``__init__``:

* the reference model (:mod:`streamguard.reference`): one float for ``diff`` (default) and
  ``ewma``, nothing for ``detrend``, a ``ref_blocks * 32`` ring buffer for ``median``, and
* a ``(warmup_blocks, 32)`` warm-up buffer, released after calibration succeeds.

Nothing grows with the length of the stream, so the guard can run indefinitely.

Lifecycle of one stream in calibrated mode (the default):

1. For ``diff``, ``median`` and ``ewma`` the first valid block only seeds the reference ->
   latched, ``NO_REFERENCE`` (also in fixed-tau mode). ``detrend`` and a fixed scalar/vector
   reference skip this step.
2. The next ``warmup_blocks`` valid blocks fill the warm-up buffer -> latched,
   ``NOT_CALIBRATED``. When it is full, tau and the scale are calibrated (``event =
   "CALIBRATED"``); if calibration fails the buffer is cleared and warm-up restarts
   (``CALIBRATION_FAILED``). Fail-closed: nothing is admitted before a tau exists.
   Warm-up residuals use the same reference mode as scoring.
3. After that, every valid block is scored and admitted iff ``S < tau``.

Invalid blocks (NaN, inf, wrong shape, non-numeric) always latch with their reason code and
never touch the warm-up buffer or drift state. They do not update the reference either; the
``diff`` reference is reset by them, so the next valid block latches with ``NO_REFERENCE``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from os import PathLike
from typing import IO, Any

import numpy as np
from numpy.typing import ArrayLike

from streamguard.calibrate import (
    MIN_WARMUP_BLOCKS,
    CalibrationError,
    DriftConfig,
    drift_update,
    validate_quantile,
)
from streamguard.latch import Reason, check_block, decide, tau_is_valid
from streamguard.score import BLOCK
from streamguard.scorers import Scorer, make_scorer

# v2: the config record names the scorer ("scorer", "scorer_params") instead of flat
# reference fields; decisions may carry the SEQUENCE_GAP reason.
SCHEMA_VERSION = 2
DEFAULT_SCORER = "spark"

# Stream defaults chosen by the SYNTHETIC warm-up x quantile sweep (benchmarks/warmup_sweep.py):
# rule 2 (no setting reached a 1% false-latch rate; lowest rate with recall >= 90%).
# calibrate_tau() keeps quantile 0.99 as its own default (parity with oes512q-latch).
STREAM_WARMUP_BLOCKS = 64
STREAM_QUANTILE = 0.999


@dataclass(frozen=True)
class Decision:
    """Outcome for one block.

    ``score`` and ``peak`` are ``None`` when the block could not be scored. ``tau`` is the
    threshold used for this block (``None`` before calibration). ``event`` is ``"CALIBRATED"``,
    ``"CALIBRATION_FAILED"`` or ``"TAU_DRIFT"`` when this block changed the guard's tau, and
    ``tau_next`` is then the new threshold.
    """

    index: int
    admit: bool
    score: float | None
    tau: float | None
    reason: Reason
    peak: float | None = None
    event: str | None = None
    tau_next: float | None = None

    @property
    def latch(self) -> int:
        """Latch bit: 1 if the block was blocked, 0 if admitted."""
        return 0 if self.admit else 1

    def to_dict(self) -> dict[str, Any]:
        def num(v: float | None) -> float | None:
            return None if v is None or not math.isfinite(v) else float(v)

        return {
            "type": "decision",
            "v": SCHEMA_VERSION,
            "i": self.index,
            "admit": self.admit,
            "latch": self.latch,
            "score": num(self.score),
            "tau": num(self.tau),
            "reason": self.reason.value,
            "peak": num(self.peak),
            "event": self.event,
            "tau_next": num(self.tau_next),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, ensure_ascii=True)


class StreamGuard:
    """Fail-closed, audited gate for one stream of 32-value blocks, around any scorer.

    Parameters
    ----------
    tau:
        ``None`` (default) for self-calibrated mode, or a fixed threshold such as the
        original contract's ``0.50``.
    warmup_blocks:
        Number of scored warm-up blocks used for calibration (calibrated mode only).
        Default 64 (SYNTHETIC sweep, see README).
    quantile:
        Quantile of warm-up scores used as tau. Default 0.999; with 64 warm-up blocks this is
        close to the largest warm-up score.
    scale:
        ``None``: robust warm-up scale in calibrated mode, ``1.0`` in fixed mode.
        A positive real: use it as the residual divisor in both modes.
    scorer:
        ``"spark"`` (default), ``"ewma"``, ``"isoforest"`` (needs scikit-learn), or any object
        implementing the :class:`streamguard.scorers.Scorer` protocol. The gate (validation,
        reason codes, calibration, drift, audit log) is identical for every scorer.
    scorer_params:
        Keyword arguments for a built-in scorer given by name, e.g. ``{"lam": 0.2}`` for
        ``ewma``.
    reference, ref_blocks, ewma_alpha:
        Shortcuts for the ``spark`` scorer: reference mode (``"diff"`` default, ``"detrend"``,
        ``"ewma"``, ``"median"``, a scalar or a 32-vector), trailing-median window in blocks,
        per-sample smoothing factor. See :mod:`streamguard.reference`. Passing them with
        another scorer raises ``ValueError``.
    drift:
        Optional :class:`DriftConfig` for a bounded slow tau update after calibration.
    audit:
        Optional :class:`AuditLogWriter`; every decision is written to it.
    """

    __slots__ = (
        "_audit",
        "_drift",
        "_expected_seq",
        "_index",
        "_quantile",
        "_scale_fixed",
        "_tau",
        "_tau0",
        "_warmup_blocks",
        "_wu_n",
        "calibration",
        "mode",
        "scorer",
    )

    def __init__(
        self,
        tau: float | None = None,
        *,
        warmup_blocks: int = STREAM_WARMUP_BLOCKS,
        quantile: float = STREAM_QUANTILE,
        scale: float | None = None,
        scorer: str | Scorer = DEFAULT_SCORER,
        scorer_params: dict[str, Any] | None = None,
        reference: str | float | ArrayLike | None = None,
        ref_blocks: int | None = None,
        ewma_alpha: float | None = None,
        drift: DriftConfig | None = None,
        audit: AuditLogWriter | None = None,
    ) -> None:
        if tau is not None and not tau_is_valid(tau):
            raise ValueError(f"tau must be a finite non-negative real or None, got {tau!r}")
        if isinstance(warmup_blocks, bool) or not isinstance(warmup_blocks, int):
            raise TypeError("warmup_blocks must be an int")
        if warmup_blocks < MIN_WARMUP_BLOCKS:
            raise ValueError(f"warmup_blocks must be >= {MIN_WARMUP_BLOCKS}")
        if scale is not None:
            scale = float(scale)
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"scale must be finite and > 0, got {scale!r}")

        self._quantile = validate_quantile(quantile)
        self._warmup_blocks = warmup_blocks
        self._drift = drift if drift is not None else DriftConfig()
        self._audit = audit
        self._scale_fixed = scale
        self._index = 0
        self._expected_seq: int | None = None
        self.calibration = None
        self.scorer = _build_scorer(scorer, scorer_params, reference, ref_blocks, ewma_alpha)

        if tau is None:
            self.mode = "calibrated"
            self._tau = None
            self._tau0 = None
            self.scorer.start(warmup_blocks, scale)
        else:
            self.mode = "fixed"
            self._tau = float(tau)
            self._tau0 = float(tau)
            self.scorer.fit_fixed(scale)
        self._wu_n = 0

    # ------------------------------------------------------------------ properties
    @property
    def tau(self) -> float | None:
        return self._tau

    @property
    def scale(self) -> float | None:
        return getattr(self.scorer, "scale", None)

    @property
    def blocks_seen(self) -> int:
        return self._index

    @property
    def calibrated(self) -> bool:
        return self._tau is not None

    def config(self) -> dict[str, Any]:
        """JSON-safe description of the guard configuration (for the audit log header)."""
        return {
            "type": "config",
            "v": SCHEMA_VERSION,
            "mode": self.mode,
            "tau": self._tau0 if self.mode == "fixed" else None,
            "warmup_blocks": self._warmup_blocks if self.mode == "calibrated" else None,
            "quantile": self._quantile,
            "scale": self._scale_fixed,
            "scorer": self.scorer.name,
            "scorer_params": self.scorer.params(),
            "drift_rate": self._drift.rate,
            "drift_max_rel": self._drift.max_rel,
            "block": BLOCK,
        }

    # ------------------------------------------------------------------ internals
    def _emit(self, d: Decision) -> Decision:
        if self._audit is not None:
            self._audit.write(d)
        return d

    def _gap(self, seq: int | None) -> bool:
        if seq is None:
            return False
        if isinstance(seq, bool) or not isinstance(seq, (int, np.integer)):
            raise TypeError(f"seq must be an int or None, got {type(seq).__name__}")
        seq = int(seq)
        gap = self._expected_seq is not None and seq != self._expected_seq
        self._expected_seq = seq + 1
        return gap

    # ------------------------------------------------------------------ main entry
    def process(self, block: object, seq: int | None = None) -> Decision:
        """Score one block and return the admit/latch :class:`Decision`. Never raises on data.

        ``seq`` is an optional producer sequence number. When given, a block whose ``seq`` is
        not the previous ``seq + 1`` (dropped, repeated or reordered blocks) latches with
        ``SEQUENCE_GAP``, is not scored, and resets the scorer like an invalid block.
        """
        i = self._index
        self._index += 1
        tau = self._tau

        gap = self._gap(seq)
        arr, reason = check_block(block)
        if gap:  # reported even if the block is also malformed
            reason = Reason.SEQUENCE_GAP
        if arr is None or gap:
            self.scorer.on_invalid()
            return self._emit(Decision(i, False, None, tau, reason))

        if tau is None:  # warm-up (calibrated mode)
            r = self.scorer.warmup(arr)
            if r is not Reason.NOT_CALIBRATED:
                return self._emit(Decision(i, False, None, None, r))
            self._wu_n += 1
            if self._wu_n < self._warmup_blocks:
                return self._emit(Decision(i, False, None, None, Reason.NOT_CALIBRATED))
            try:
                cal = self.scorer.fit(self._quantile)
            except (CalibrationError, ValueError):
                self._wu_n = 0
                return self._emit(
                    Decision(
                        i, False, None, None, Reason.CALIBRATION_FAILED, event="CALIBRATION_FAILED"
                    )
                )
            self.calibration = cal
            self._tau = self._tau0 = cal.tau
            return self._emit(
                Decision(
                    i,
                    False,
                    None,
                    None,
                    Reason.NOT_CALIBRATED,
                    event="CALIBRATED",
                    tau_next=cal.tau,
                )
            )

        res = self.scorer.score(arr)
        if isinstance(res, Reason):
            return self._emit(Decision(i, False, None, tau, res))
        score, peak = res
        admit, reason = decide(score, tau)
        if reason is Reason.NONFINITE_SCORE:
            return self._emit(Decision(i, False, None, tau, reason))

        event = tau_next = None
        new_tau = drift_update(tau, self._tau0, score, self._quantile, self._drift)
        if new_tau != tau:
            self._tau = new_tau
            event, tau_next = "TAU_DRIFT", new_tau
        return self._emit(Decision(i, admit, score, tau, reason, peak, event, tau_next))


def _build_scorer(
    scorer: str | Scorer,
    scorer_params: dict[str, Any] | None,
    reference: Any,
    ref_blocks: int | None,
    ewma_alpha: float | None,
) -> Scorer:
    shortcuts = {"reference": reference, "ref_blocks": ref_blocks, "ewma_alpha": ewma_alpha}
    shortcuts = {k: v for k, v in shortcuts.items() if v is not None}
    params = dict(scorer_params or {})
    if isinstance(scorer, str):
        if shortcuts and scorer != "spark":
            raise ValueError(f"{sorted(shortcuts)} only apply to the 'spark' scorer")
        return make_scorer(scorer, **params, **shortcuts)
    if shortcuts or params:
        raise ValueError("scorer_params/reference options cannot be combined with a scorer object")
    missing = [m for m in _PROTOCOL if not hasattr(scorer, m)]
    if missing:
        raise TypeError(f"scorer object is missing {missing}")
    return scorer


_PROTOCOL = ("name", "start", "warmup", "fit", "fit_fixed", "score", "on_invalid", "params")


class AuditLogWriter:
    """Append-only JSONL writer: one JSON object per line, UTF-8/ASCII, no NaN tokens.

    ``target`` is a path (opened in append mode, closed by :meth:`close`) or an open text
    stream (left open). Keeps only a line counter, never the records.
    """

    __slots__ = ("_fh", "_owns", "_flush", "lines")

    def __init__(self, target: str | PathLike[str] | IO[str], *, flush: bool = True) -> None:
        if isinstance(target, (str, PathLike)):
            self._fh: IO[str] = open(target, "a", encoding="utf-8")  # noqa: SIM115
            self._owns = True
        else:
            self._fh = target
            self._owns = False
        self._flush = flush
        self.lines = 0

    def write_record(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, allow_nan=False, ensure_ascii=True) + "\n")
        self.lines += 1
        if self._flush:
            self._fh.flush()

    def write(self, decision: Decision) -> None:
        self.write_record(decision.to_dict())

    def close(self) -> None:
        if self._owns:
            self._fh.close()
        elif self._flush:
            self._fh.flush()

    def __enter__(self) -> AuditLogWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
