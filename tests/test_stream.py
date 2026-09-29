"""StreamGuard state machine, O(1) memory, audit log."""

import io
import json
import tracemalloc

import numpy as np
import pytest

from streamguard import AuditLogWriter, Decision, DriftConfig, Reason, StreamGuard

NAN = float("nan")


def quiet(rng, n=1):
    return rng.normal(0.0, 1.0, size=(n, 32))


def warm(guard, rng, n):
    return [guard.process(b) for b in quiet(rng, n)]


def test_calibrated_lifecycle(rng):
    g = StreamGuard(warmup_blocks=8)
    assert g.mode == "calibrated" and not g.calibrated and g.tau is None
    ds = warm(g, rng, 9)  # 1 seed block + 8 warm-up blocks
    assert ds[0].reason is Reason.NO_REFERENCE
    assert all(not d.admit and d.reason is Reason.NOT_CALIBRATED for d in ds[1:])
    assert [d.event for d in ds].count("CALIBRATED") == 1
    assert ds[-1].event == "CALIBRATED" and ds[-1].tau_next == g.tau
    assert g.calibrated and g.calibration.n_blocks == 8 and g.scale > 0
    d = g.process(np.zeros(32))  # at the reference -> tiny score -> admitted
    assert d.admit and d.reason is Reason.OK and d.score < d.tau
    spike = np.zeros(32)
    spike[3] = 100.0
    d = g.process(spike)
    assert not d.admit and d.reason is Reason.SCORE_OVER_TAU and d.latch == 1
    assert d.peak == pytest.approx(d.peak) and d.index == 10
    assert g.blocks_seen == 11


def test_fixed_mode_default_tau(rng):
    g = StreamGuard(tau=0.5, reference=0.0)
    assert g.mode == "fixed" and g.tau == 0.5 and g.scale == 1.0
    assert g.process([0.01] * 32).admit
    spike = [0.0] * 32
    spike[0] = 1.0
    d = g.process(spike)
    assert d.reason is Reason.SCORE_OVER_TAU and d.score == pytest.approx(0.5181218, abs=1e-7)


def test_fixed_mode_trailing_reference_needs_seed_block():
    g = StreamGuard(tau=0.5, reference="median")
    d0 = g.process(np.full(32, 10.0))
    assert d0.reason is Reason.NO_REFERENCE and not d0.admit
    d1 = g.process(np.full(32, 10.1))  # residual 0.1 vs trailing median 10
    assert d1.admit and d1.score == pytest.approx(0.1)


def test_vector_reference_and_fixed_scale():
    g = StreamGuard(tau=0.5, reference=np.arange(32.0), scale=10.0)
    d = g.process(np.arange(32.0) + 4.0)
    assert d.score == pytest.approx(0.4) and d.admit


@pytest.mark.parametrize(
    "bad, reason",
    [
        ([NAN] * 32, Reason.NAN_INPUT),
        ([float("inf")] + [0.0] * 31, Reason.INF_INPUT),
        ([], Reason.BAD_SHAPE),
        ([0.0] * 5, Reason.BAD_SHAPE),
        (None, Reason.BAD_TYPE),
        (["x"] * 32, Reason.BAD_TYPE),
    ],
)
def test_invalid_input_latches_in_every_state(rng, bad, reason):
    g = StreamGuard(warmup_blocks=4)
    d = g.process(bad)  # before any reference
    assert not d.admit and d.reason is reason and d.score is None
    warm(g, rng, 5)
    assert g.calibrated
    tau = g.tau
    d = g.process(bad)  # after calibration
    assert not d.admit and d.reason is reason and d.tau == tau
    assert g.tau == tau  # invalid data never moves tau


def test_invalid_blocks_do_not_count_toward_warmup(rng):
    g = StreamGuard(warmup_blocks=4)
    g.process(quiet(rng)[0])
    for _ in range(10):
        g.process([NAN] * 32)
    assert not g.calibrated
    warm(g, rng, 4)  # first block after the gap re-seeds the diff reference
    assert not g.calibrated
    warm(g, rng, 1)
    assert g.calibrated


def test_calibration_failure_restarts_warmup(rng):
    g = StreamGuard(warmup_blocks=4)
    ds = [g.process(np.ones(32)) for _ in range(5)]  # constant data: zero spread
    assert ds[-1].reason is Reason.CALIBRATION_FAILED
    assert ds[-1].event == "CALIBRATION_FAILED" and not g.calibrated
    warm(g, rng, 4)
    assert g.calibrated


def test_overflowing_residual_latches():
    g = StreamGuard(tau=0.5, reference=-1e308)
    d = g.process(np.full(32, 1e308))
    assert d.reason is Reason.NONFINITE_SCORE and not d.admit


def test_overflow_after_scaling_latches():
    g = StreamGuard(tau=0.5, reference=0.0, scale=1e-300)
    d = g.process(np.full(32, 1e10))
    assert d.reason is Reason.NONFINITE_SCORE and not d.admit


def test_overflow_during_warmup_latches():
    g = StreamGuard(reference=-1e308, warmup_blocks=4)
    assert g.process(np.full(32, 1e308)).reason is Reason.NONFINITE_SCORE


def test_fixed_scale_in_calibrated_mode(rng):
    g = StreamGuard(warmup_blocks=4, scale=2.0)
    warm(g, rng, 5)
    assert g.scale == 2.0 and g.calibration.scale_method == "fixed"


def test_determinism_seed_42():
    def run():
        r = np.random.default_rng(42)
        g = StreamGuard(warmup_blocks=16, drift=DriftConfig(rate=0.01))
        return [g.process(b).to_json() for b in r.normal(size=(200, 32))]

    assert run() == run()


def test_drift_is_bounded_and_logged(rng):
    g = StreamGuard(warmup_blocks=8, drift=DriftConfig(rate=0.05, max_rel=0.2))
    warm(g, rng, 9)
    tau0 = g.tau
    events = []
    for _ in range(500):  # sustained large blocks push tau up to the cap only
        d = g.process(rng.normal(0, 50, size=32))
        events.append(d.event)
        if d.event == "TAU_DRIFT":
            assert d.tau_next == g.tau
    assert "TAU_DRIFT" in events
    assert g.tau == pytest.approx(1.2 * tau0)


def test_streaming_memory_is_constant(rng):
    g = StreamGuard(warmup_blocks=8, drift=DriftConfig(rate=0.01))
    data = rng.normal(size=(20_000, 32))
    for b in data[:500]:
        g.process(b)
    assert g.calibrated
    # No per-stream attribute is a growing container.
    for name in StreamGuard.__slots__:
        assert not isinstance(getattr(g, name), (list, dict, set, tuple)), name
    assert not hasattr(g, "__dict__")
    ref_state = g._ref.last
    tracemalloc.start()
    for b in data[500:5000]:
        g.process(b)
    mid, _ = tracemalloc.get_traced_memory()
    for b in data[5000:]:
        g.process(b)
    end, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert end - mid < 4096, (mid, end)  # 15k more blocks, no growth
    assert peak < 1_000_000
    assert isinstance(ref_state, float) and g._wu_buf is None


def test_warmup_buffer_released(rng):
    g = StreamGuard(warmup_blocks=4)
    assert g._wu_buf.shape == (4, 32)
    warm(g, rng, 5)
    assert g._wu_buf is None


def test_trailing_reference_ring_buffer_wraps():
    g = StreamGuard(tau=10.0, reference="median", ref_blocks=2)
    for v in (1.0, 2.0, 3.0, 4.0):
        g.process(np.full(32, v))
    # window holds the last two blocks (3 and 4) -> median 3.5
    d = g.process(np.full(32, 3.5))
    assert d.score == pytest.approx(0.0)


@pytest.mark.parametrize(
    "kw, exc",
    [
        ({"tau": -1.0}, ValueError),
        ({"tau": float("nan")}, ValueError),
        ({"warmup_blocks": 2}, ValueError),
        ({"warmup_blocks": 8.0}, TypeError),
        ({"reference": "median", "ref_blocks": 0}, ValueError),
        ({"reference": "ewma", "ewma_alpha": 0.0}, ValueError),
        ({"scale": 0.0}, ValueError),
        ({"reference": "mean"}, ValueError),
        ({"reference": np.zeros(3)}, ValueError),
        ({"reference": NAN}, ValueError),
        ({"quantile": 1.0}, ValueError),
    ],
)
def test_bad_config_raises(kw, exc):
    with pytest.raises(exc):
        StreamGuard(**kw)


def test_decision_json_is_strict():
    d = Decision(3, False, float("inf"), 0.5, Reason.NONFINITE_SCORE, peak=NAN)
    rec = json.loads(d.to_json())
    assert rec["score"] is None and rec["peak"] is None
    assert rec == {
        "type": "decision",
        "v": 1,
        "i": 3,
        "admit": False,
        "latch": 1,
        "score": None,
        "tau": 0.5,
        "reason": "NONFINITE_SCORE",
        "peak": None,
        "event": None,
        "tau_next": None,
    }


def test_audit_log_to_stream_and_path(rng, tmp_path):
    buf = io.StringIO()
    path = tmp_path / "audit.jsonl"
    with AuditLogWriter(path) as fw:
        w = AuditLogWriter(buf)
        g = StreamGuard(warmup_blocks=4, audit=w)
        fw.write_record(g.config())
        for b in quiet(rng, 10):
            fw.write(g.process(b))
        g.process([NAN] * 32)
        w.close()
    assert w.lines == 11 and fw.lines == 11
    lines = buf.getvalue().splitlines()
    assert len(lines) == 11
    recs = [json.loads(line) for line in lines]
    assert [r["i"] for r in recs] == list(range(11))
    assert recs[-1]["reason"] == "NAN_INPUT"
    assert "NaN" not in buf.getvalue()
    file_recs = [json.loads(x) for x in path.read_text().splitlines()]
    assert file_recs[0]["type"] == "config" and file_recs[0]["warmup_blocks"] == 4
    assert all(r["type"] == "decision" for r in file_recs[1:])
    # append mode: a second writer adds to the same file
    with AuditLogWriter(path) as fw2:
        fw2.write_record({"type": "note"})
    assert len(path.read_text().splitlines()) == 12


def test_config_fixed_vector_reference():
    cfg = StreamGuard(tau=0.5, reference=np.zeros(32)).config()
    assert cfg["mode"] == "fixed" and cfg["tau"] == 0.5 and "ref_window" not in cfg
    assert cfg["reference"] == "fixed" and cfg["reference_value"] == [0.0] * 32
    json.dumps(cfg, allow_nan=False)


def test_stream_defaults():
    g = StreamGuard()
    cfg = g.config()
    assert cfg["warmup_blocks"] == 64 and cfg["quantile"] == 0.999 and cfg["reference"] == "diff"
