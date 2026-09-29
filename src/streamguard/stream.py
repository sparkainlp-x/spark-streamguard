"""StreamGuard: per-stream state machine with O(1) memory, plus a JSONL audit log writer.

Per stream, the guard keeps only fixed-size numpy buffers allocated in ``__init__``:

* a ring buffer of the last ``ref_blocks * 32`` valid samples for the trailing-median
  reference (the causal reference used by the oes512q-latch benchmark), and
* a ``(warmup_blocks, 32)`` warm-up buffer, released after calibration succeeds.

Nothing grows with the length of the stream, so the guard can run indefinitely.

Lifecycle of one stream in calibrated mode (the default):

1. The first valid block only seeds the trailing reference -> latched, ``NO_REFERENCE``
   (also in fixed-tau mode; not applicable with a fixed scalar/vector reference).
2. The next ``warmup_blocks`` valid blocks fill the warm-up buffer -> latched,
   ``NOT_CALIBRATED``. When it is full, tau and the scale are calibrated (``event =
   "CALIBRATED"``); if calibration fails the buffer is cleared and warm-up restarts
   (``CALIBRATION_FAILED``). Fail-closed: nothing is admitted before a tau exists.
3. After that, every valid block is scored and admitted iff ``S < tau``.

Invalid blocks (NaN, inf, wrong shape, non-numeric) always latch with their reason code and
never touch the reference, warm-up buffer or drift state.
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
    DEFAULT_QUANTILE,
    MIN_WARMUP_BLOCKS,
    CalibrationError,
    DriftConfig,
    calibrate_tau,
    drift_update,
    validate_quantile,
)
from streamguard.latch import Reason, check_block, decide, tau_is_valid
from streamguard.score import BLOCK, score_components

SCHEMA_VERSION = 1


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
    """Fail-closed streaming guard for one stream of 32-value blocks.

    Parameters
    ----------
    tau:
        ``None`` (default) for self-calibrated mode, or a fixed threshold such as the
        original contract's ``0.50``.
    warmup_blocks:
        Number of scored warm-up blocks used for calibration (calibrated mode only).
    quantile:
        Quantile of warm-up scores used as tau.
    scale:
        ``None``: robust warm-up scale in calibrated mode, ``1.0`` in fixed mode.
        A positive real: use it as the residual divisor in both modes.
    reference:
        ``"trailing_median"`` (median of the last ``ref_blocks * 32`` valid samples), a scalar,
        or a 32-vector.
    ref_blocks:
        Size of the trailing-median window, in blocks.
    drift:
        Optional :class:`DriftConfig` for a bounded slow tau update after calibration.
    audit:
        Optional :class:`AuditLogWriter`; every decision is written to it.
    """

    __slots__ = (
        "_audit",
        "_drift",
        "_index",
        "_quantile",
        "_ref_buf",
        "_ref_count",
        "_ref_fixed",
        "_ref_pos",
        "_scale",
        "_scale_fixed",
        "_tau",
        "_tau0",
        "_warmup_blocks",
        "_wu_buf",
        "_wu_n",
        "calibration",
        "mode",
    )

    def __init__(
        self,
        tau: float | None = None,
        *,
        warmup_blocks: int = 32,
        quantile: float = DEFAULT_QUANTILE,
        scale: float | None = None,
        reference: str | float | ArrayLike = "trailing_median",
        ref_blocks: int = 4,
        drift: DriftConfig | None = None,
        audit: AuditLogWriter | None = None,
    ) -> None:
        if tau is not None and not tau_is_valid(tau):
            raise ValueError(f"tau must be a finite non-negative real or None, got {tau!r}")
        if isinstance(warmup_blocks, bool) or not isinstance(warmup_blocks, int):
            raise TypeError("warmup_blocks must be an int")
        if warmup_blocks < MIN_WARMUP_BLOCKS:
            raise ValueError(f"warmup_blocks must be >= {MIN_WARMUP_BLOCKS}")
        if isinstance(ref_blocks, bool) or not isinstance(ref_blocks, int) or ref_blocks < 1:
            raise ValueError("ref_blocks must be an int >= 1")
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
        self.calibration = None

        if isinstance(reference, str):
            if reference != "trailing_median":
                raise ValueError(f"unknown reference {reference!r}")
            self._ref_fixed = None
            self._ref_buf = np.empty(ref_blocks * BLOCK, dtype=np.float64)
        else:
            ref = np.asarray(reference, dtype=np.float64)
            if ref.shape not in ((), (BLOCK,)) or not np.all(np.isfinite(ref)):
                raise ValueError(f"reference must be finite, scalar or shape ({BLOCK},)")
            self._ref_fixed = ref.copy()
            self._ref_buf = None
        self._ref_count = 0
        self._ref_pos = 0

        if tau is None:
            self.mode = "calibrated"
            self._tau = None
            self._tau0 = None
            self._scale = None
            self._wu_buf = np.empty((warmup_blocks, BLOCK), dtype=np.float64)
        else:
            self.mode = "fixed"
            self._tau = float(tau)
            self._tau0 = float(tau)
            self._scale = scale if scale is not None else 1.0
            self._wu_buf = None
        self._wu_n = 0

    # ------------------------------------------------------------------ properties
    @property
    def tau(self) -> float | None:
        return self._tau

    @property
    def scale(self) -> float | None:
        return self._scale

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
            "reference": "trailing_median" if self._ref_fixed is None else self._ref_fixed.tolist(),
            "ref_window": None if self._ref_buf is None else int(self._ref_buf.size),
            "drift_rate": self._drift.rate,
            "drift_max_rel": self._drift.max_rel,
            "block": BLOCK,
        }

    # ------------------------------------------------------------------ internals
    def _reference(self) -> np.ndarray | None:
        if self._ref_fixed is not None:
            return self._ref_fixed
        if self._ref_count == 0:
            return None
        return np.asarray(np.median(self._ref_buf[: self._ref_count]))

    def _push_reference(self, arr: np.ndarray) -> None:
        buf = self._ref_buf
        if buf is None:
            return
        buf[self._ref_pos : self._ref_pos + BLOCK] = arr
        self._ref_pos = (self._ref_pos + BLOCK) % buf.size
        self._ref_count = min(self._ref_count + BLOCK, buf.size)

    def _emit(self, d: Decision) -> Decision:
        if self._audit is not None:
            self._audit.write(d)
        return d

    # ------------------------------------------------------------------ main entry
    def process(self, block: object) -> Decision:
        """Score one block and return the admit/latch :class:`Decision`. Never raises on data."""
        i = self._index
        self._index += 1
        tau = self._tau

        arr, reason = check_block(block)
        if arr is None:
            return self._emit(Decision(i, False, None, tau, reason))

        ref = self._reference()
        if ref is None:  # first valid block only seeds the trailing reference
            self._push_reference(arr)
            return self._emit(Decision(i, False, None, tau, Reason.NO_REFERENCE))

        with np.errstate(over="ignore", invalid="ignore"):
            raw = arr - ref
        if not np.all(np.isfinite(raw)):
            self._push_reference(arr)
            return self._emit(Decision(i, False, None, tau, Reason.NONFINITE_SCORE))

        if tau is None:  # warm-up (calibrated mode)
            self._wu_buf[self._wu_n] = raw
            self._wu_n += 1
            self._push_reference(arr)
            if self._wu_n < self._warmup_blocks:
                return self._emit(Decision(i, False, None, None, Reason.NOT_CALIBRATED))
            scale_arg: str | float = "robust" if self._scale_fixed is None else self._scale_fixed
            try:
                cal = calibrate_tau(
                    self._wu_buf,
                    quantile=self._quantile,
                    scale=scale_arg,
                    min_blocks=self._warmup_blocks,
                )
            except (CalibrationError, ValueError):
                self._wu_n = 0
                return self._emit(
                    Decision(
                        i, False, None, None, Reason.CALIBRATION_FAILED, event="CALIBRATION_FAILED"
                    )
                )
            self.calibration = cal
            self._tau = self._tau0 = cal.tau
            self._scale = cal.scale
            self._wu_buf = None  # release the warm-up buffer
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

        with np.errstate(over="ignore", invalid="ignore"):
            r = raw / self._scale
        self._push_reference(arr)
        if not np.all(np.isfinite(r)):
            return self._emit(Decision(i, False, None, tau, Reason.NONFINITE_SCORE))
        bs = score_components(r)
        admit, reason = decide(bs.score, tau)

        event = tau_next = None
        new_tau = drift_update(tau, self._tau0, bs.score, self._quantile, self._drift)
        if new_tau != tau:
            self._tau = new_tau
            event, tau_next = "TAU_DRIFT", new_tau
        return self._emit(Decision(i, admit, bs.score, tau, reason, bs.peak, event, tau_next))


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
