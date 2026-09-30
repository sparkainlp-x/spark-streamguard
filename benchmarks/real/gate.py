"""Run the real :class:`streamguard.StreamGuard` gate (any scorer) under the M4 protocol.

Every number produced here goes through ``StreamGuard.process`` block by block, one guard per
channel, so it measures the shipped gate, not a re-implementation. Alignment with the M4
protocol (``protocol.py``): block 0 seeds the reference (``NO_REFERENCE``), blocks 1..64 are the
warm-up (``NOT_CALIBRATED``; calibration happens on block 64), blocks 65.. are evaluated.
``spark``/``detrend`` needs no seed block, so it is fed from block 1, as in M4.

Per channel the block ratio is ``score / tau`` for scored blocks and ``inf`` for blocks the
gate latched without a score (invalid input, overflow, sequence gap, not calibrated); SMD's
block ratio is the maximum over channels, exactly as for the raw baselines. A block is flagged
iff the gate did not admit it on some channel, which equals ``ratio >= 1``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from streamguard import StreamGuard

import protocol as P

Feed = Callable[[np.ndarray, int], Iterator[tuple[Any, int | None, int | None]]]


def clean_feed(xb: np.ndarray, first: int) -> Iterator[tuple[Any, int | None, int | None]]:
    """Yield ``(block, seq, eval_index)``: ``eval_index`` is the protocol block index."""
    for b in range(first, xb.shape[0]):
        yield xb[b], None, b


def first_block(scorer: str, params: dict) -> int:
    return 1 if scorer == "spark" and params.get("reference") == "detrend" else 0


def gate_channel(
    xb_c: np.ndarray,
    scorer: str,
    params: dict,
    q: float,
    feed: Feed = clean_feed,
    by_block: dict | None = None,
) -> tuple[np.ndarray, Counter, bool]:
    """Ratios (per protocol block, NaN where not evaluated), reason counts over the evaluation
    region, and whether the guard calibrated at block 64. ``by_block`` (optional) receives the
    reason code of every evaluated block. An admitted block without a finite score would be
    counted under ``ADMITTED_NONFINITE`` (the gate never does this; measured, not assumed)."""
    n = xb_c.shape[0]
    g = StreamGuard(scorer=scorer, scorer_params=params, warmup_blocks=P.WARMUP, quantile=q)
    ratio = np.full(n, np.nan)
    reasons: Counter = Counter()
    calibrated_on_time = False
    for blk, seq, b in feed(xb_c, first_block(scorer, params)):
        d = g.process(blk, seq=seq)
        if d.event == "CALIBRATED" and b == P.WARMUP:
            calibrated_on_time = True
        if b is None or b < P.FIRST_EVAL:
            continue
        reasons[d.reason.value] += 1
        if d.admit and (d.score is None or not np.isfinite(d.score)):
            reasons["ADMITTED_NONFINITE"] += 1
        if by_block is not None:
            by_block[b] = d.reason.value
        if d.score is None:
            r = np.inf
        elif d.tau > 0:
            r = d.score / d.tau
        else:
            r = np.inf  # tau == 0 and S >= 0: the gate latches
        ratio[b] = r
    return ratio, reasons, calibrated_on_time


def gate_series(
    s: P.Series,
    scorer: str,
    params: dict,
    q: float,
    feed: Feed = clean_feed,
) -> tuple[np.ndarray, dict]:
    """Evaluation-region block ratio (max over usable channels) and run info."""
    xb = P.blocks_of(s.x)
    per = []
    reasons: Counter = Counter()
    late = 0
    for c in np.flatnonzero(s.channels):
        r, rc, ok = gate_channel(xb[:, :, c], scorer, params, q, feed)
        per.append(r[P.FIRST_EVAL :])
        reasons.update(rc)
        late += not ok
    if not per:
        return np.zeros(xb.shape[0] - P.FIRST_EVAL), {"channels": 0}
    ratio = np.max(np.vstack(per), axis=0)
    return ratio, {"channels": len(per), "not_calibrated_at_64": late, "reasons": dict(reasons)}
