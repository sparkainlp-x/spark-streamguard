"""Reference modes: values, calibration consistency, determinism, fail-closed, O(1) memory."""

import tracemalloc

import numpy as np
import pytest

from streamguard import STREAM_QUANTILE, Reason, StreamGuard, calibrate_tau
from streamguard.reference import (
    DEFAULT_REFERENCE,
    MODES,
    DetrendRef,
    DiffRef,
    EwmaRef,
    FixedRef,
    MedianRef,
    make_reference,
)

NAN = float("nan")


def test_default_is_diff():
    assert DEFAULT_REFERENCE == "diff"
    assert MODES == ("median", "ewma", "detrend", "diff")
    assert StreamGuard(scorer="spark").config()["scorer_params"]["reference"] == "diff"


def test_make_reference_types_and_alias():
    assert isinstance(make_reference("median"), MedianRef)
    assert isinstance(make_reference("trailing_median"), MedianRef)
    assert isinstance(make_reference("ewma"), EwmaRef)
    assert isinstance(make_reference("detrend"), DetrendRef)
    assert isinstance(make_reference("diff"), DiffRef)
    assert isinstance(make_reference(1.5), FixedRef)
    assert isinstance(make_reference(np.zeros(32)), FixedRef)


@pytest.mark.parametrize(
    "args",
    [
        ("mean",),
        ("median", 0),
        ("median", True),
        ("ewma", 4, 0.0),
        ("ewma", 4, 1.5),
        (np.zeros(3),),
    ],
)
def test_make_reference_rejects(args):
    with pytest.raises(ValueError):
        make_reference(*args)


def test_median_ref():
    m = MedianRef(2)
    x = np.arange(32.0)
    assert m.ref(x) is None
    m.push(x)
    assert float(m.ref(x)) == 15.5
    assert m.config() == {"ref_window": 64}


def test_ewma_closed_form_matches_recursion(rng):
    a = 0.2
    e = EwmaRef(a)
    blocks = rng.normal(size=(5, 32))
    assert e.ref(blocks[0]) is None
    e.push(blocks[0])
    level = float(np.median(blocks[0]))
    assert float(e.ref(blocks[1])) == level
    for b in blocks[1:]:
        e.push(b)
        for v in b:
            level += a * (v - level)
        assert e.level == pytest.approx(level, rel=1e-12)
    assert e.config() == {"ewma_alpha": a}


def test_ewma_alpha_one_is_last_sample():
    e = EwmaRef(1.0)
    e.push(np.zeros(32))
    e.push(np.arange(32.0))
    assert e.level == pytest.approx(31.0)


def test_detrend_fits_a_line():
    d = DetrendRef()
    line = 3.0 + 0.5 * np.arange(32.0)
    np.testing.assert_allclose(d.ref(line), line)
    x = line.copy()
    x[10] += 4.0  # spike: residual is 4 minus the small pull on the fit
    r = x - d.ref(x)
    assert np.argmax(np.abs(r)) == 10 and 3.5 < r[10] < 4.0
    d.push(x)
    assert d.config() == {}


def test_diff_ref_is_first_difference(rng):
    d = DiffRef()
    a, b = rng.normal(size=(2, 32))
    assert d.ref(a) is None
    d.push(a)
    r = b - d.ref(b)
    np.testing.assert_allclose(r, np.diff(np.concatenate([[a[-1]], b])))
    assert d.config() == {}


def test_fixed_ref():
    f = FixedRef(2.0)
    assert float(f.ref(np.zeros(32))) == 2.0
    f.push(np.zeros(32))
    assert f.config() == {"reference_value": 2.0}


@pytest.mark.parametrize("mode", MODES)
def test_calibration_uses_same_reference_mode(rng, mode):
    blocks = rng.normal(size=(9, 32)).cumsum(axis=None).reshape(9, 32)
    g = StreamGuard(warmup_blocks=8, reference=mode)
    for b in blocks:
        g.process(b)
    assert g.calibrated
    shadow = make_reference(mode)
    residuals = []
    for b in blocks:
        ref = shadow.ref(b)
        if ref is not None:
            residuals.append(b - ref)
        shadow.push(b)
    residuals = residuals[:8]
    assert g.calibration == calibrate_tau(np.array(residuals), quantile=STREAM_QUANTILE)


def test_detrend_needs_no_seed_block(rng):
    g = StreamGuard(warmup_blocks=4, reference="detrend")
    ds = [g.process(b) for b in rng.normal(size=(4, 32))]
    assert all(d.reason is Reason.NOT_CALIBRATED for d in ds)
    assert ds[-1].event == "CALIBRATED"


@pytest.mark.parametrize("mode", MODES)
def test_determinism_per_mode(mode):
    def run():
        r = np.random.default_rng(42)
        g = StreamGuard(warmup_blocks=8, reference=mode)
        return [g.process(b).to_json() for b in r.normal(size=(60, 32)).cumsum(axis=0)]

    assert run() == run()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "bad, reason",
    [([NAN] * 32, Reason.NAN_INPUT), ([0.0] * 7, Reason.BAD_SHAPE), (None, Reason.BAD_TYPE)],
)
def test_fail_closed_per_mode(rng, mode, bad, reason):
    g = StreamGuard(warmup_blocks=4, reference=mode)
    assert g.process(bad).reason is reason
    for b in rng.normal(size=(6, 32)):
        g.process(b)
    assert g.calibrated
    d = g.process(bad)
    assert not d.admit and d.reason is reason and d.score is None


@pytest.mark.parametrize("mode", MODES)
def test_spike_latches_per_mode(rng, mode):
    g = StreamGuard(warmup_blocks=16, reference=mode)
    for b in rng.normal(size=(17, 32)):
        g.process(b)
    spike = rng.normal(size=32)
    spike[12] += 25.0
    assert g.process(spike).reason is Reason.SCORE_OVER_TAU


@pytest.mark.parametrize("mode", MODES)
def test_constant_memory_per_mode(rng, mode):
    g = StreamGuard(warmup_blocks=8, reference=mode)
    data = rng.normal(size=(12_000, 32)).cumsum(axis=0)
    for b in data[:500]:
        g.process(b)
    tracemalloc.start()
    for b in data[500:4000]:
        g.process(b)
    mid, _ = tracemalloc.get_traced_memory()
    for b in data[4000:]:
        g.process(b)
    end, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert end - mid < 4096, (mode, mid, end)
    ref = g.scorer._ref
    for name in type(ref).__slots__:
        assert not isinstance(getattr(ref, name), (list, dict, set, tuple))


def test_diff_resets_after_invalid_block():
    g = StreamGuard(tau=100.0, reference="diff")
    g.process(np.full(32, 5.0))
    assert g.process(np.full(32, 5.0)).admit
    assert g.process([NAN] * 32).reason is Reason.NAN_INPUT
    d = g.process(np.full(32, 50.0))  # no difference across the gap
    assert d.reason is Reason.NO_REFERENCE and not d.admit and d.score is None
    d = g.process(np.full(32, 50.0))
    assert d.admit and d.score == 0.0


def test_diff_reset_during_warmup_does_not_count_block(rng):
    g = StreamGuard(warmup_blocks=4, reference="diff")
    g.process(rng.normal(size=32))  # seed
    g.process(rng.normal(size=32))  # warm-up 1
    g.process([NAN] * 32)
    assert g.process(rng.normal(size=32)).reason is Reason.NO_REFERENCE
    ds = [g.process(b) for b in rng.normal(size=(2, 32))]  # warm-up 2 and 3
    assert not g.calibrated and ds[-1].reason is Reason.NOT_CALIBRATED
    assert g.process(rng.normal(size=32)).event == "CALIBRATED"


@pytest.mark.parametrize("mode", ["median", "ewma", "detrend"])
def test_other_modes_keep_state_after_invalid_block(mode):
    g = StreamGuard(tau=100.0, reference=mode)
    g.process(np.full(32, 5.0))
    g.process([NAN] * 32)
    d = g.process(np.full(32, 5.0))
    assert d.reason is Reason.OK and d.score == pytest.approx(0.0)


def test_fixed_reference_on_invalid_is_noop():
    f = FixedRef(1.0)
    f.on_invalid()
    assert float(f.value) == 1.0
