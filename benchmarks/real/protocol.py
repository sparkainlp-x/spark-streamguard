"""Shared protocol, detectors and metrics for the real-data benchmark (REPORTED numbers).

Protocol (fixed before any result was computed; identical for every method):

1. **Blocking.** Each series is cut into contiguous, non-overlapping 32-sample blocks; a
   trailing partial block is dropped. SMD machines are multivariate (38 channels); every method
   scores each channel separately (see 5).
2. **Labels.** A block is positive if any of its samples lies in a labelled anomaly (NAB:
   ``combined_windows.json`` windows, inclusive; SMD: ``test_label == 1``). Labels are used only
   for scoring and for choosing settings on the tuning split.
3. **Warm-up / calibration.** Block 0 seeds references; blocks 1..64 (``WARMUP = 64``, the
   StreamGuard default) are the calibration window for every method; blocks 65.. are the
   evaluation region. Labels inside blocks 0..64 are ignored (warm-up anomalies are counted and
   reported, not removed). A series needs at least ``MIN_BLOCKS = 97`` blocks (warm-up plus 32
   evaluation blocks); shorter series are **excluded** and listed.
4. **Threshold.** For every method and channel, ``tau`` = the ``q`` quantile (numpy linear) of
   that method's warm-up block scores; a block latches iff ``score >= max(tau, 1e-12)`` (a
   zero score never latches). Fixed-threshold rows are marked.
5. **Multivariate aggregation (SMD).** Per channel ``ratio = score / max(tau, 1e-12)``; the
   block ratio is the maximum over channels and the block latches iff it is >= 1. Channels
   whose warm-up samples (blocks 0..64) are exactly constant are excluded for all methods
   (counted). NAB is univariate, so its ratio is the channel ratio.
6. **Settings.** Each method's hyper-parameters and ``q`` are chosen on the tuning split only
   (highest pooled F1 over the tuning series' evaluation blocks; ties: first in grid order), then
   frozen and applied once to the test split. "Fixed" rows use settings fixed in advance.
7. **Split.** Per dataset, the included series are shuffled with ``numpy.random.default_rng(42)``
   and split in half (tune gets the extra series when the count is odd).

Residual used by the ported baselines (as in the oes512q-latch benchmark): for each block, the
median of the trailing ``L = min(288, max(32, floor(0.05 * n)))`` samples ending just before the
block (fewer if not available yet), divided per channel by the robust warm-up scale
(1.4826*MAD, fallback 1.2533*mean absolute deviation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from streamguard.calibrate import CalibrationError, calibrate_tau, robust_scale
from streamguard.reference import make_reference
from streamguard.score import BLOCK, score_components

WARMUP = 64
FIRST_EVAL = WARMUP + 1
MIN_BLOCKS = FIRST_EVAL + 32
TAU_FLOOR = 1e-12
SEED = 42
QUANTILES = (0.99, 0.995, 0.999)
IF_TREES = 100


@dataclass
class Series:
    name: str
    dataset: str
    x: np.ndarray  # (n_blocks * 32, channels), float64
    y: np.ndarray  # (n_blocks,) block labels 0/1
    channels: np.ndarray = field(default=None)  # bool mask of included channels
    meta: dict = field(default_factory=dict)

    @property
    def n_blocks(self) -> int:
        return len(self.y)


def block_labels(sample_labels: np.ndarray) -> np.ndarray:
    n = len(sample_labels) // BLOCK
    return sample_labels[: n * BLOCK].reshape(n, BLOCK).any(axis=1).astype(int)


def make_series(name: str, dataset: str, x: np.ndarray, sample_labels: np.ndarray) -> Series:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name}: non-finite values")
    n = x.shape[0] // BLOCK
    x = x[: n * BLOCK]
    y = block_labels(np.asarray(sample_labels, dtype=int))
    wu = x[: FIRST_EVAL * BLOCK]
    channels = (wu.max(axis=0) > wu.min(axis=0)) if n >= FIRST_EVAL else np.ones(x.shape[1], bool)
    return Series(name, dataset, x, y, channels)


def split(series: list[Series]) -> tuple[list[Series], list[Series]]:
    idx = np.random.default_rng(SEED).permutation(len(series))
    n_tune = (len(series) + 1) // 2
    ordered = [series[i] for i in idx]
    return sorted(ordered[:n_tune], key=lambda s: s.name), sorted(
        ordered[n_tune:], key=lambda s: s.name
    )


# ------------------------------------------------------------------------------ detectors
# Every detector maps one series to block scores of shape (n_blocks, channels); row 0 is NaN
# (seed block). Rows 1..64 are warm-up scores, rows 65.. are scored for evaluation.


def blocks_of(x: np.ndarray) -> np.ndarray:
    """(n*32, C) -> (n, 32, C)."""
    return x.reshape(-1, BLOCK, x.shape[1])


def trailing_len(n_samples: int) -> int:
    return min(288, max(32, int(math.floor(0.05 * n_samples))))


def trailing_residual(x: np.ndarray) -> np.ndarray:
    """Scaled residual vs trailing median (oes512q-latch benchmark residual). Shape (n, 32, C)."""
    n = x.shape[0] // BLOCK
    L = trailing_len(x.shape[0])
    xb = blocks_of(x)
    ref = np.full((n, x.shape[1]), np.nan)
    for b in range(1, n):
        s = b * BLOCK
        ref[b] = np.median(x[max(0, s - L) : s], axis=0)
    raw = xb - ref[:, None, :]
    scale = np.ones(x.shape[1])
    for c in range(x.shape[1]):
        try:
            scale[c], _ = robust_scale(raw[1:FIRST_EVAL, :, c])
        except CalibrationError:
            scale[c] = np.nan  # constant channel: excluded upstream
    return raw / scale[None, None, :]


def weighted_score(r: np.ndarray) -> np.ndarray:
    """0.45*Peak + 0.35*RMS + 0.20*MeanAbs over axis 1 (the 32 samples)."""
    a = np.abs(r)
    return 0.45 * a.max(axis=1) + 0.35 * np.sqrt((r * r).mean(axis=1)) + 0.20 * a.mean(axis=1)


def score_oes_latch(x: np.ndarray, cache: dict) -> np.ndarray:
    return weighted_score(cache_residual(x, cache))


def cache_residual(x: np.ndarray, cache: dict) -> np.ndarray:
    if "r" not in cache:
        cache["r"] = trailing_residual(x)
    return cache["r"]


def score_cusum(x: np.ndarray, cache: dict, k: float) -> np.ndarray:
    """Two-sided tabular CUSUM on the residual stream from block 1 on (no reset); block max."""
    r = cache_residual(x, cache)
    n, _, C = r.shape
    flat = r[1:].reshape(-1, C)
    sp = np.zeros(C)
    sn = np.zeros(C)
    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        v = flat[i]
        sp = np.maximum(0.0, sp + v - k)
        sn = np.maximum(0.0, sn - v - k)
        out[i] = np.maximum(sp, sn)
    res = np.full((n, C), np.nan)
    res[1:] = out.reshape(n - 1, BLOCK, C).max(axis=1)
    return res


def score_ewma(x: np.ndarray, cache: dict, lam: float) -> np.ndarray:
    """EWMA control chart on the residual stream: |z_t| / sqrt(lam / (2 - lam)); block max."""
    from scipy.signal import lfilter

    r = cache_residual(x, cache)
    n, _, C = r.shape
    flat = r[1:].reshape(-1, C)
    z = lfilter([lam], [1.0, -(1.0 - lam)], flat, axis=0)
    stat = np.abs(z) / math.sqrt(lam / (2.0 - lam))
    res = np.full((n, C), np.nan)
    res[1:] = stat.reshape(n - 1, BLOCK, C).max(axis=1)
    return res


def score_zscore(x: np.ndarray, cache: dict, window: int) -> np.ndarray:
    """Rolling z-score: max_j |x_j - trailing mean| / trailing std (window ending before block)."""
    n = x.shape[0] // BLOCK
    xb = blocks_of(x)
    res = np.full((n, x.shape[1]), np.nan)
    tiny = np.finfo(float).tiny
    big = np.finfo(float).max
    for b in range(1, n):
        s = b * BLOCK
        h = x[max(0, s - window) : s]
        sd = np.maximum(h.std(axis=0), tiny)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            z = np.abs((xb[b] - h.mean(axis=0)) / sd).max(axis=0)
        res[b] = np.minimum(np.nan_to_num(z, nan=0.0, posinf=big), big)
    return res


def score_iforest(x: np.ndarray, cache: dict, channels: np.ndarray) -> np.ndarray:
    """IsolationForest (100 trees, seed 42) fitted per channel on the 64 warm-up residual blocks."""
    from sklearn.ensemble import IsolationForest

    r = cache_residual(x, cache)
    n, _, C = r.shape
    res = np.full((n, C), np.nan)
    for c in range(C):
        if not channels[c]:
            continue
        m = IsolationForest(n_estimators=IF_TREES, random_state=SEED).fit(r[1:FIRST_EVAL, :, c])
        res[1:, c] = -m.score_samples(r[1:, :, c])
    return res


def score_streamguard(x: np.ndarray, cache: dict, mode: str, channels: np.ndarray) -> np.ndarray:
    """StreamGuard scores via the library's own reference, calibration scale and scorer.

    Equivalent to ``StreamGuard(reference=mode, warmup_blocks=64).process`` fed from block 1
    (``detrend``) or block 0 (``diff``/``median``): warm-up blocks 1..64, scoring from block 65.
    Warm-up rows hold the warm-up scores on the calibrated scale, so the common threshold rule
    reproduces ``calibrate_tau`` exactly. Tested against ``StreamGuard.process``.
    """
    n = x.shape[0] // BLOCK
    xb = blocks_of(x)
    res = np.full((n, x.shape[1]), np.nan)
    for c in range(x.shape[1]):
        if not channels[c]:
            continue
        ref = make_reference(mode)
        raw = np.empty((n, BLOCK))
        first = 1 if mode == "detrend" else 0
        for b in range(first, n):
            blk = xb[b, :, c]
            rb = ref.ref(blk)
            if rb is not None:
                raw[b] = blk - rb
            ref.push(blk)
        try:
            cal = calibrate_tau(raw[1:FIRST_EVAL], quantile=0.99, min_blocks=WARMUP)
        except CalibrationError:
            continue  # zero spread in residuals: channel never latches (counted by caller)
        r = raw[1:] / cal.scale
        res[1:, c] = [score_components(row).score for row in r]
    return res


# ------------------------------------------------------------------------------ methods


@dataclass(frozen=True)
class Method:
    name: str
    family: str
    params: dict
    fixed_tau: float | None = None  # untuned fixed threshold instead of the quantile rule
    fixed_q: float | None = None  # q fixed in advance (not tuned)

    def label(self) -> str:
        if not self.params:
            return self.name
        return self.name + " (" + ", ".join(f"{k}={v}" for k, v in self.params.items()) + ")"


def method_grid() -> list[Method]:
    """All candidate settings, per family. Grids fixed before any result was computed."""
    g = [
        Method("StreamGuard diff", "streamguard_diff", {"mode": "diff"}),
        Method("StreamGuard median", "streamguard_median", {"mode": "median"}),
        Method("StreamGuard detrend", "streamguard_detrend", {"mode": "detrend"}),
        Method("CUSUM", "cusum", {"k": 0.5}),
        Method("CUSUM", "cusum", {"k": 1.0}),
        Method("CUSUM", "cusum", {"k": 2.0}),
        Method("EWMA chart", "ewma", {"lam": 0.1}),
        Method("EWMA chart", "ewma", {"lam": 0.2}),
        Method("EWMA chart", "ewma", {"lam": 0.3}),
        Method("Rolling z-score", "zscore", {"window": 64}),
        Method("Rolling z-score", "zscore", {"window": 288}),
        Method("Isolation Forest", "iforest", {}),
    ]
    return g


FIXED_METHODS = [
    Method("StreamGuard default (untuned)", "sg_default", {"mode": "diff"}, fixed_q=0.999),
    Method("oes512q-latch v2 config (q=0.99)", "oes_latch", {}, fixed_q=0.99),
    Method("oes512q-latch fixed tau=0.50", "oes_fixed", {}, fixed_tau=0.50),
]


def raw_scores(method: Method, s: Series, cache: dict) -> np.ndarray:
    key = (method.family.replace("sg_default", "streamguard_diff"), tuple(method.params.items()))
    if key in cache:
        return cache[key]
    fam = method.family
    x = s.x
    if fam.startswith("streamguard") or fam == "sg_default":
        out = score_streamguard(x, cache, method.params["mode"], s.channels)
    elif fam in ("oes_latch", "oes_fixed"):
        out = score_oes_latch(x, cache)
    elif fam == "cusum":
        out = score_cusum(x, cache, method.params["k"])
    elif fam == "ewma":
        out = score_ewma(x, cache, method.params["lam"])
    elif fam == "zscore":
        out = score_zscore(x, cache, method.params["window"])
    elif fam == "iforest":
        out = score_iforest(x, cache, s.channels)
    else:  # pragma: no cover
        raise ValueError(fam)
    cache[key] = out
    return out


def ratios(scores: np.ndarray, channels: np.ndarray, q: float | None, fixed_tau: float | None):
    """Evaluation-region block ratio (max over channels of score / tau) and channel stats."""
    wu = scores[1:FIRST_EVAL]
    ev = scores[FIRST_EVAL:]
    usable = channels & np.all(np.isfinite(wu), axis=0) & np.all(np.isfinite(ev), axis=0)
    if fixed_tau is not None:
        tau = np.full(scores.shape[1], fixed_tau)
    else:
        tau = np.quantile(np.where(np.isfinite(wu), wu, 0.0), q, axis=0, method="linear")
    tau = np.maximum(tau, TAU_FLOOR)
    if not usable.any():
        return np.zeros(ev.shape[0]), int(usable.sum())
    r = ev[:, usable] / tau[usable]
    return r.max(axis=1), int(usable.sum())


# ------------------------------------------------------------------------------ metrics


def confusion(y: np.ndarray, flag: np.ndarray) -> dict:
    y = np.asarray(y, bool)
    f = np.asarray(flag, bool)
    return {
        "tp": int((f & y).sum()),
        "fp": int((f & ~y).sum()),
        "fn": int((~f & y).sum()),
        "tn": int((~f & ~y).sum()),
    }


def prf(c: dict) -> dict:
    p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    far = c["fp"] / (c["fp"] + c["tn"]) if c["fp"] + c["tn"] else 0.0
    return {"precision": p, "recall": r, "f1": f1, "false_latch_rate": far}


def average_precision(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y, int)
    if y.sum() == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    ys, ss = y[order], score[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    last = np.r_[ss[1:] != ss[:-1], True]  # ties share one threshold (as scikit-learn)
    tp, fp = tp[last], fp[last]
    prec = tp / (tp + fp)
    rec = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def events(y: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs [start, end) of positive blocks that start inside the evaluation region."""
    out = []
    b = 0
    n = len(y)
    while b < n:
        if y[b]:
            e = b
            while e < n and y[e]:
                e += 1
            out.append((b, e))
            b = e
        else:
            b += 1
    return out


def delays(y_eval: np.ndarray, flag: np.ndarray, starts_in_eval: np.ndarray) -> list[int | None]:
    """Per event: blocks from event start to first latch inside it (None if missed)."""
    res = []
    for a, e in events(y_eval):
        if not starts_in_eval[a]:
            continue
        hit = np.flatnonzero(flag[a:e])
        res.append(int(hit[0]) if hit.size else None)
    return res


def evaluate_series(y_full: np.ndarray, ratio: np.ndarray) -> dict:
    y = y_full[FIRST_EVAL:]
    flag = ratio >= 1.0
    c = confusion(y, flag)
    # an event "starts in eval" unless it continues from the warm-up region
    starts = np.ones(len(y), bool)
    if len(y) and y_full[FIRST_EVAL - 1] and y[0]:
        starts[0] = False
    d = delays(y, flag, starts)
    return {
        **c,
        "ap": average_precision(y, ratio),
        "positives": int(y.sum()),
        "events": len(d),
        "events_detected": sum(v is not None for v in d),
        "delays": [v for v in d if v is not None],
        "f1": prf(c)["f1"],
    }


def pooled(rows: list[dict]) -> dict:
    c = {k: sum(r[k] for r in rows) for k in ("tp", "fp", "fn", "tn")}
    aps = [r["ap"] for r in rows if r["ap"] is not None]
    f1s = [r["f1"] for r in rows if r["positives"] > 0]
    dl = [d for r in rows for d in r["delays"]]
    ev = sum(r["events"] for r in rows)
    return {
        **c,
        **prf(c),
        "mean_ap": float(np.mean(aps)) if aps else None,
        "mean_f1": float(np.mean(f1s)) if f1s else None,
        "events": ev,
        "event_recall": sum(r["events_detected"] for r in rows) / ev if ev else None,
        "median_delay_blocks": float(np.median(dl)) if dl else None,
    }


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10_000, seed: int = SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    m = diffs[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def wilcoxon_p(diffs: np.ndarray) -> float | None:
    if not np.any(diffs != 0):
        return None
    from scipy.stats import wilcoxon

    return float(wilcoxon(diffs, zero_method="wilcox", alternative="two-sided").pvalue)
