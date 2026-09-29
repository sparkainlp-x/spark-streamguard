"""Calibration determinism and drift bounds."""

import math

import numpy as np
import pytest

from streamguard.calibrate import (
    DEFAULT_QUANTILE,
    DEFAULT_TAU,
    MAD_TO_SIGMA,
    MEANAD_TO_SIGMA,
    CalibrationError,
    DriftConfig,
    TauCalibration,
    calibrate_tau,
    drift_update,
    robust_scale,
    validate_quantile,
)
from streamguard.score import block_score


def test_defaults():
    assert DEFAULT_TAU == 0.50
    assert DEFAULT_QUANTILE == 0.99


def test_calibration_is_deterministic_with_seed_42():
    a = calibrate_tau(np.random.default_rng(42).normal(size=(64, 32)))
    b = calibrate_tau(np.random.default_rng(42).normal(size=(64, 32)))
    assert a == b
    assert isinstance(a, TauCalibration)
    assert a.n_blocks == 64 and a.scale_method == "mad" and a.quantile == 0.99


def test_calibration_is_order_independent(rng):
    w = rng.normal(size=(40, 32))
    a = calibrate_tau(w)
    b = calibrate_tau(w[::-1])
    assert a.tau == pytest.approx(b.tau, rel=1e-12)
    assert a.scale == pytest.approx(b.scale, rel=1e-12)


def test_calibration_matches_manual_computation(rng):
    w = rng.normal(2.0, 3.0, size=(20, 32))
    cal = calibrate_tau(w, quantile=0.9)
    med = np.median(w)
    sigma = MAD_TO_SIGMA * np.median(np.abs(w - med))
    scores = [block_score(row, 0.0, sigma) for row in w]
    assert cal.scale == pytest.approx(sigma)
    assert cal.tau == pytest.approx(np.quantile(scores, 0.9))
    assert cal.as_dict()["method"] == "warmup-quantile-type7"


def test_scale_invariance_of_latch(rng):
    w = rng.normal(size=(32, 32))
    a = calibrate_tau(w)
    b = calibrate_tau(w * 1000.0)
    assert a.tau == pytest.approx(b.tau)
    assert b.scale == pytest.approx(1000.0 * a.scale)


def test_fixed_scale(rng):
    w = rng.normal(size=(8, 32))
    cal = calibrate_tau(w, scale=1.0)
    assert cal.scale == 1.0 and cal.scale_method == "fixed"


def test_mean_abs_dev_fallback():
    # >50% zeros -> MAD = 0 -> fall back to sqrt(pi/2) * mean|v - median|
    w = np.zeros((4, 32))
    w[0, :8] = 1.0
    s, method = robust_scale(w)
    assert method == "mean_abs_dev"
    assert s == pytest.approx(MEANAD_TO_SIGMA * 8 / 128)
    cal = calibrate_tau(w)
    assert cal.scale_method == "mean_abs_dev"


def test_constant_warmup_fails_closed():
    with pytest.raises(CalibrationError):
        calibrate_tau(np.ones((8, 32)))


@pytest.mark.parametrize(
    "w, exc",
    [
        (np.zeros((3, 32)) + np.arange(32), CalibrationError),  # too few blocks
        (np.zeros((8, 31)), ValueError),
        (np.zeros(32), ValueError),
        (np.full((8, 32), np.nan), ValueError),
    ],
)
def test_bad_warmup(w, exc):
    with pytest.raises(exc):
        calibrate_tau(w)


@pytest.mark.parametrize(
    "kw",
    [
        {"scale": "mad"},
        {"scale": 0.0},
        {"scale": float("nan")},
        {"min_blocks": 0},
        {"min_blocks": True},
    ],
)
def test_bad_calibration_args(rng, kw):
    with pytest.raises(ValueError):
        calibrate_tau(rng.normal(size=(8, 32)), **kw)


@pytest.mark.parametrize("q", [0.0, 1.0, -0.5, 1.5])
def test_bad_quantile_value(q):
    with pytest.raises(ValueError):
        validate_quantile(q)


@pytest.mark.parametrize("q", [True, "0.9", None])
def test_bad_quantile_type(q):
    with pytest.raises(TypeError):
        validate_quantile(q)


def test_robust_scale_errors():
    with pytest.raises(CalibrationError):
        robust_scale([])
    with pytest.raises(ValueError):
        robust_scale([1.0, float("inf")])


def test_drift_disabled_by_default():
    cfg = DriftConfig()
    assert not cfg.enabled
    assert drift_update(1.0, 1.0, 5.0, 0.99, cfg) == 1.0


def test_drift_step_sizes_and_bounds():
    cfg = DriftConfig(rate=0.01, max_rel=0.1)
    up = drift_update(1.0, 1.0, 2.0, 0.99, cfg)  # hit: +0.01*(1-0.01)
    down = drift_update(1.0, 1.0, 0.1, 0.99, cfg)  # miss: -0.01*0.01
    assert up == pytest.approx(1.0 + 0.0099)
    assert down == pytest.approx(1.0 - 0.0001)
    tau = 1.0
    for _ in range(10_000):  # sustained anomalies cannot push tau past the band
        tau = drift_update(tau, 1.0, 1e9, 0.99, cfg)
    assert tau == pytest.approx(1.1)
    for _ in range(100_000):
        tau = drift_update(tau, 1.0, 0.0, 0.99, cfg)
    assert tau == pytest.approx(0.9)


def test_drift_ignores_nonfinite_score():
    cfg = DriftConfig(rate=0.01)
    assert drift_update(1.0, 1.0, math.nan, 0.99, cfg) == 1.0


@pytest.mark.parametrize(
    "kw", [{"rate": -0.1}, {"rate": 0.5}, {"max_rel": 1.0}, {"rate": math.nan}, {"rate": True}]
)
def test_bad_drift_config(kw):
    with pytest.raises(ValueError):
        DriftConfig(**kw)


def test_parity_with_oes512q_latch_reference():
    # Values produced by oes_calibration.calibrate_tau in sparkainlp-x/oes512q-latch v0.2.0
    # (commit 1801db1) on this deterministic input.
    w = np.array(
        [[math.sin(0.37 * (32 * i + j)) * (1 + 0.1 * i) for j in range(32)] for i in range(16)]
    )
    cal = calibrate_tau(w)
    assert cal.tau == pytest.approx(1.244330740394901, rel=1e-12)
    assert cal.scale == pytest.approx(1.6282805934186648, rel=1e-12)
    assert calibrate_tau(w, quantile=0.9).tau == pytest.approx(1.1834576869014541, rel=1e-12)
