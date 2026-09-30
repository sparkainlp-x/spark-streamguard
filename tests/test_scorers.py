"""Pluggable scorers: every scorer is wrapped by the same fail-closed, audited gate."""

import gc
import io
import json
import math
import tracemalloc

import numpy as np
import pytest

from streamguard import (
    SCORERS,
    AuditLogWriter,
    CalibrationError,
    EwmaScorer,
    IsoForestScorer,
    Reason,
    Scorer,
    SparkScorer,
    StreamGuard,
    make_scorer,
)
from streamguard.scorers import quantile_tau

NAN = float("nan")


def blocks(rng, n, sd=1.0):
    return rng.normal(0.0, sd, size=(n, 32))


def run(g, data):
    return [g.process(b) for b in data]


def ewma_reference(x_blocks, lam, window, warmup):
    """Direct per-sample implementation of the documented ewma scorer (for equivalence)."""
    x = np.concatenate(x_blocks)
    n = len(x_blocks)
    raw = np.full((n, 32), np.nan)
    for b in range(1, n):
        s = b * 32
        raw[b] = x_blocks[b] - np.median(x[max(0, s - window) : s])
    wu = raw[1 : warmup + 1]
    dev = np.abs(wu - np.median(wu))
    scale = 1.4826 * np.median(dev)
    r = raw / scale
    z = 0.0
    out = np.full(n, np.nan)
    c = math.sqrt(lam / (2 - lam))
    for b in range(1, n):
        m = 0.0
        for v in r[b]:
            z = lam * v + (1 - lam) * z
            m = max(m, abs(z) / c)
        out[b] = m
    tau = float(np.quantile(out[1 : warmup + 1], 0.99, method="linear"))
    return out, tau, scale


# ------------------------------------------------------------------------------ registry


def test_registry_and_protocol():
    assert sorted(SCORERS) == ["ewma", "isoforest", "spark"]
    for name in ("spark", "ewma"):
        s = make_scorer(name)
        assert isinstance(s, Scorer) and s.name == name
        json.dumps(s.params())
    with pytest.raises(ValueError, match="unknown scorer"):
        make_scorer("nope")
    assert isinstance(make_scorer("spark", reference="median"), SparkScorer)


def test_quantile_tau_rejects_nonfinite():
    assert quantile_tau(np.array([0.0, 1.0]), 0.5) == 0.5
    with pytest.raises(CalibrationError):
        quantile_tau(np.array([np.inf, np.inf]), 0.5)


def test_scorer_options_are_validated():
    for bad in (True, 31, 64.0):
        with pytest.raises(ValueError, match="ref_window"):
            EwmaScorer(ref_window=bad)
    for bad in (0.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="lam"):
            EwmaScorer(lam=bad)
    for bad in (0, True, 2.5):
        with pytest.raises(ValueError, match="n_estimators"):
            IsoForestScorer(n_estimators=bad)
    with pytest.raises(ValueError, match="only apply"):
        StreamGuard(scorer="ewma", reference="diff")
    with pytest.raises(ValueError, match="cannot be combined"):
        StreamGuard(scorer=EwmaScorer(), scorer_params={"lam": 0.1})
    with pytest.raises(TypeError, match="missing"):
        StreamGuard(scorer=object())


# ------------------------------------------------------------------------------ ewma


def test_ewma_matches_direct_implementation(rng):
    data = list(blocks(rng, 150))
    data[120] = data[120] + np.where(np.arange(32) > 10, 6.0, 0.0)  # level shift
    g = StreamGuard(scorer="ewma", scorer_params={"lam": 0.2, "ref_window": 100}, quantile=0.99)
    ds = run(g, data)
    ref, tau, scale = ewma_reference(data, 0.2, 100, 64)
    assert ds[0].reason is Reason.NO_REFERENCE
    assert ds[64].event == "CALIBRATED"
    assert g.tau == pytest.approx(tau, rel=1e-12) and g.scale == pytest.approx(scale)
    np.testing.assert_allclose(g.scorer.warmup_scores, ref[1:65], rtol=1e-12)
    got = np.array([d.score for d in ds[65:]])
    np.testing.assert_allclose(got, ref[65:], rtol=1e-10)
    assert not ds[120].admit and ds[120].reason is Reason.SCORE_OVER_TAU
    assert g.scorer._wu is None  # warm-up buffer released


def test_ewma_config_record_and_audit(rng):
    buf = io.StringIO()
    w = AuditLogWriter(buf)
    g = StreamGuard(scorer="ewma", scorer_params={"lam": 0.3}, warmup_blocks=8, audit=w)
    w.write_record(g.config())
    run(g, blocks(rng, 12))
    recs = [json.loads(line) for line in buf.getvalue().splitlines()]
    assert recs[0]["type"] == "config" and recs[0]["v"] == 2
    assert recs[0]["scorer"] == "ewma"
    assert recs[0]["scorer_params"] == {"lam": 0.3, "ref_window": 288}
    assert [r["reason"] for r in recs[1:3]] == ["NO_REFERENCE", "NOT_CALIBRATED"]
    assert recs[9]["event"] == "CALIBRATED" and len(recs) == 13


def test_ewma_fixed_mode_and_invalid_blocks(rng):
    g = StreamGuard(tau=3.0, scorer="ewma")
    assert g.scale == 1.0
    ds = run(g, [np.zeros(32), np.full(32, NAN), np.zeros(32), [1.0] * 31])
    assert [d.reason for d in ds] == [
        Reason.NO_REFERENCE,
        Reason.NAN_INPUT,
        Reason.OK,  # ewma keeps its history across an invalid block
        Reason.BAD_SHAPE,
    ]


def test_ewma_ring_buffer_wraps_and_memory_is_constant(rng):
    g = StreamGuard(scorer="ewma", scorer_params={"ref_window": 40}, warmup_blocks=8)
    data = blocks(rng, 6000)
    run(g, data[:200])
    assert g.calibrated and g.scorer._count == 40
    tracemalloc.start()
    run(g, data[200:3000])
    gc.collect()
    mid, _ = tracemalloc.get_traced_memory()
    run(g, data[3000:])
    gc.collect()
    end, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # 3,000 more blocks: any per-block growth of >= 6 bytes would exceed this. The slack covers
    # allocator caches of freed numpy temporaries (7.6 KB seen once on CPython 3.11 in CI).
    assert end - mid < 16384


def test_ewma_overflow_is_fail_closed():
    big = np.full(32, 1e308)
    g = StreamGuard(tau=1.0, scorer="ewma")
    ds = run(g, [-big, big, np.zeros(32)])
    assert ds[1].reason is Reason.NONFINITE_SCORE  # residual overflow
    # scaled residual overflow (tiny fixed scale)
    g = StreamGuard(tau=1.0, scorer="ewma", scale=1e-300)
    ds = run(g, [np.zeros(32), np.full(32, 1e10)])
    assert ds[1].reason is Reason.NONFINITE_SCORE and not ds[1].admit
    # block statistic overflow: finite residuals, infinite |z| / c; state is reset
    g = StreamGuard(tau=1.0, scorer="ewma")
    ds = run(g, [np.zeros(32), np.full(32, 1.7e308)])
    assert ds[1].reason is Reason.NONFINITE_SCORE and g.scorer._z == 0.0
    # warm-up residual overflow does not count towards the warm-up
    g = StreamGuard(scorer="ewma", warmup_blocks=4)
    ds = run(g, [-big, big])
    assert ds[1].reason is Reason.NONFINITE_SCORE and g._wu_n == 0


def test_ewma_calibration_failures(rng, monkeypatch):
    g = StreamGuard(scorer="ewma", warmup_blocks=4)
    ds = run(g, [np.ones(32)] * 6)  # constant: zero spread
    assert ds[4].reason is Reason.CALIBRATION_FAILED and not g.calibrated
    assert ds[5].reason is Reason.NOT_CALIBRATED and g._wu_n == 1  # warm-up restarted
    # fixed tiny scale: scaled warm-up residuals overflow
    g = StreamGuard(scorer="ewma", warmup_blocks=4, scale=1e-300)
    ds = run(g, [np.zeros(32)] + [np.full(32, 1e10)] * 4)
    assert ds[-1].reason is Reason.CALIBRATION_FAILED
    # warm-up scores that give no finite tau
    monkeypatch.setattr(EwmaScorer, "_block", lambda self, r: math.inf)
    g = StreamGuard(scorer="ewma", warmup_blocks=4, scale=1.0)
    ds = run(g, blocks(rng, 5))
    assert ds[-1].reason is Reason.CALIBRATION_FAILED and g.scorer._n == 0


# ------------------------------------------------------------------------------ isoforest


def test_isoforest_gate_matches_sklearn(rng):
    from sklearn.ensemble import IsolationForest

    data = list(blocks(rng, 100))
    data[90] = data[90] + 8.0
    g = StreamGuard(scorer="isoforest", scorer_params={"ref_window": 64}, quantile=0.99)
    ds = run(g, data)
    assert ds[64].event == "CALIBRATED"
    x = np.concatenate(data)
    raw = np.array([data[b] - np.median(x[max(0, b * 32 - 64) : b * 32]) for b in range(1, 100)])
    r = raw / g.scale
    m = IsolationForest(n_estimators=100, random_state=42).fit(r[:64])
    np.testing.assert_allclose(g.scorer.warmup_scores, -m.score_samples(r[:64]))
    np.testing.assert_allclose([d.score for d in ds[65:]], -m.score_samples(r[64:]))
    assert g.tau == pytest.approx(np.quantile(-m.score_samples(r[:64]), 0.99))
    assert not ds[90].admit
    cfg = g.config()
    assert cfg["scorer"] == "isoforest"
    assert cfg["scorer_params"] == {"n_estimators": 100, "seed": 42, "ref_window": 64}
    # bounded model: fixed number of trees, each on at most min(256, warm-up) samples
    assert len(g.scorer._model.estimators_) == 100 and g.scorer._model.max_samples_ == 64
    assert g.scorer._wu is None


def test_isoforest_needs_warmup_and_rejects_bad_input(rng):
    with pytest.raises(ValueError, match="warm-up"):
        StreamGuard(tau=1.0, scorer="isoforest")
    g = StreamGuard(scorer="isoforest", scorer_params={"n_estimators": 10}, warmup_blocks=8)
    run(g, blocks(rng, 9))
    assert g.calibrated
    ds = run(g, [np.full(32, NAN), np.full(32, np.inf), ["a"] * 32, blocks(rng, 1)[0]])
    assert [d.reason for d in ds[:3]] == [Reason.NAN_INPUT, Reason.INF_INPUT, Reason.BAD_TYPE]
    assert ds[3].score is not None
    g2 = StreamGuard(scorer="isoforest", warmup_blocks=4)
    assert run(g2, [np.ones(32)] * 5)[-1].reason is Reason.CALIBRATION_FAILED


# ------------------------------------------------------------------------------ custom scorer


class MeanAbs:
    """Minimal third-party scorer: mean |x| (no reference, no state)."""

    name = "meanabs"

    def __init__(self):
        self.scale = 1.0
        self._scores = []

    def start(self, warmup_blocks, scale):
        self._scores = []

    def warmup(self, x):
        self._scores.append(float(np.abs(x).mean()))
        return Reason.NOT_CALIBRATED

    def fit(self, quantile):
        from streamguard import TauCalibration

        tau = quantile_tau(np.array(self._scores), quantile)
        return TauCalibration(tau, 1.0, quantile, len(self._scores))

    def fit_fixed(self, scale):
        pass

    def score(self, x):
        v = float(np.abs(x).mean())
        return (NAN, None) if v > 1e6 else (v, None)

    def on_invalid(self):
        pass

    def params(self):
        return {}


def test_custom_scorer_gets_the_same_gate(rng):
    g = StreamGuard(scorer=MeanAbs(), warmup_blocks=4, quantile=0.5)
    ds = run(g, blocks(rng, 4))
    assert ds[-1].event == "CALIBRATED"
    assert g.config()["scorer"] == "meanabs" and g.config()["scorer_params"] == {}
    d = g.process(np.full(32, 1e7))
    assert not d.admit and d.reason is Reason.NONFINITE_SCORE and d.score is None
    assert g.process(np.full(32, NAN)).reason is Reason.NAN_INPUT
    assert g.process(np.zeros(32)).admit


# ------------------------------------------------------------------------------ sequence gaps


def test_sequence_gap_latches_and_resets(rng):
    g = StreamGuard(tau=10.0, reference="diff")  # implies the spark scorer
    data = blocks(rng, 6)
    ds = [g.process(data[0], seq=0), g.process(data[1], seq=1)]
    assert ds[1].reason is Reason.OK
    d = g.process(data[2], seq=3)  # block 2 was dropped
    assert d.reason is Reason.SEQUENCE_GAP and not d.admit and d.score is None
    assert g.process(data[3], seq=4).reason is Reason.NO_REFERENCE  # diff was reset
    assert g.process(data[4], seq=np.int64(5)).reason is Reason.OK
    d = g.process(np.full(32, NAN), seq=5)  # repeated seq and bad data: gap is reported
    assert d.reason is Reason.SEQUENCE_GAP
    assert g.process(data[5]).reason is Reason.NO_REFERENCE  # no seq: no gap check
    with pytest.raises(TypeError, match="seq"):
        g.process(data[5], seq="7")
    with pytest.raises(TypeError, match="seq"):
        g.process(data[5], seq=True)


def test_isoforest_overflow_and_spark_calibration_failure(rng):
    g = StreamGuard(scorer="isoforest", scorer_params={"n_estimators": 5}, warmup_blocks=8)
    run(g, blocks(rng, 9, sd=0.01))
    d = g.process(np.full(32, 1e308))  # residual finite, scaled residual overflows
    assert d.reason is Reason.NONFINITE_SCORE and not d.admit
    g = StreamGuard(reference="diff", warmup_blocks=4)
    ds = run(g, [np.ones(32)] * 5)
    assert ds[-1].reason is Reason.CALIBRATION_FAILED and g.scorer._n == 0
