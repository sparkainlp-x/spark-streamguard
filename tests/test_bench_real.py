"""Tiny generated fixture for the real-data benchmark code (no downloads in CI)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

REAL = Path(__file__).resolve().parents[1] / "benchmarks" / "real"
sys.path.insert(0, str(REAL))

from streamguard import StreamGuard  # noqa: E402

import download  # noqa: E402
import gate as G  # noqa: E402
import make_tables  # noqa: E402
import pivot  # noqa: E402
import protocol as P  # noqa: E402
import run  # noqa: E402
from data import load_nab, load_smd  # noqa: E402

NAB_KEYS = ["realFix/a.csv", "realFix/b.csv", "realFix/c.csv", "realFix/d.csv", "realFix/short.csv"]


def write_fixture(root: Path) -> None:
    rng = np.random.default_rng(42)
    windows = {}
    t0 = np.datetime64("2020-01-01T00:00:00")
    for i, key in enumerate(NAB_KEYS):
        n = 1000 if "short" in key else 3400 + 32 * i
        x = np.cumsum(rng.normal(0, 0.2, n)) + rng.normal(0, 1, n)
        spike = (500 if "short" in key else 2600) + 40 * i
        x[spike] += 25
        ts = [str(t0 + np.timedelta64(300 * k, "s")).replace("T", " ") for k in range(n)]
        path = root / "nab" / key
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            fh.write("timestamp,value\n")
            fh.writelines(f"{t},{v}\n" for t, v in zip(ts, x, strict=True))
        windows[key] = [[ts[spike - 20] + ".000000", ts[spike + 20] + ".000000"]]
    (root / "nab" / "combined_windows.json").write_text(json.dumps(windows))
    for m in ("machine-9-1", "machine-9-2", "machine-9-3"):
        n = 3300
        x = rng.normal(0, 1, (n, 3))
        x[:, 2] = 1.0  # constant channel: excluded
        lab = np.zeros(n, int)
        x[3000:3010, 1] += 30
        lab[3000:3010] = 1
        for part in ("test", "test_label"):
            (root / "smd" / part).mkdir(parents=True, exist_ok=True)
        np.savetxt(root / "smd" / "test" / f"{m}.txt", x, delimiter=",")
        np.savetxt(root / "smd" / "test_label" / f"{m}.txt", lab, fmt="%d")


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("bench")
    write_fixture(root)
    return root


@pytest.fixture(scope="module")
def report(fixture_root):
    out = fixture_root / "results.json"
    code = run.main(["--data-dir", str(fixture_root), "--out", str(out), "--nab-keys", *NAB_KEYS])
    assert code == 0
    return json.loads(out.read_text())


def test_report_structure(report):
    nab, smd = report["datasets"]["nab"], report["datasets"]["smd"]
    assert nab["excluded_short"] == ["realFix/short.csv"]
    assert len(nab["tune"]) == 2 and len(nab["test"]) == 2
    assert smd["channels_constant_in_warmup"] == 3
    fams = set(nab["results"])
    assert {"sg_default", "streamguard_diff", "cusum", "ewma", "zscore", "iforest"} <= fams
    assert nab["results"]["oes_fixed"]["q"] is None
    assert nab["results"]["sg_default"]["q"] == 0.999
    for fam, r in nab["results"].items():
        t = r["test"]
        assert 0 <= t["f1"] <= 1 and 0 <= t["false_latch_rate"] <= 1, fam
    assert report["streaming_throughput_nab"]["blocks_per_s"] > 0
    assert report["memory_nab_longest"]["StreamGuard.process (per stream)"]["peak_bytes"] > 0
    assert report["evidence"].startswith("REPORTED")


def test_tables_render(report, fixture_root):
    md = make_tables.render(report)
    assert "| Method (setting chosen on tune split)" in md and "NAB" in md and "SMD" in md
    make_tables.main([str(fixture_root / "results.json")])
    assert (fixture_root / "results.md").is_file()


def test_streamguard_scores_match_library(fixture_root):
    s = [x for x in load_nab(fixture_root, NAB_KEYS) if x.n_blocks >= P.MIN_BLOCKS][0]
    for mode in ("diff", "median", "detrend"):
        sc = P.score_streamguard(s.x, {}, mode, s.channels)
        ratio, _ = P.ratios(sc, s.channels, 0.999, None)
        g = StreamGuard(reference=mode, quantile=0.999)
        xb = P.blocks_of(s.x)[:, :, 0]
        ds = [g.process(b) for b in (xb[1:] if mode == "detrend" else xb)]
        if mode == "detrend":
            ds = [None, *ds]
        lib = np.array([d.score for d in ds[P.FIRST_EVAL :]])
        np.testing.assert_allclose(lib, sc[P.FIRST_EVAL :, 0], rtol=1e-12)
        assert (np.array([not d.admit for d in ds[P.FIRST_EVAL :]]) == (ratio >= 1)).all()


def test_smd_loader_and_constant_channel(fixture_root):
    ss = load_smd(fixture_root)
    assert len(ss) == 3 and ss[0].x.shape[1] == 3
    assert list(ss[0].channels) == [True, True, False]


def test_metrics_helpers():
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.2).astype(int)
    s = np.round(rng.random(200), 1)  # ties
    assert P.average_precision(y, s) == pytest.approx(average_precision_score(y, s))
    assert P.average_precision(np.zeros(5, int), s[:5]) is None
    assert P.events(np.array([0, 1, 1, 0, 1])) == [(1, 3), (4, 5)]
    y_full = np.zeros(P.FIRST_EVAL + 10, int)
    y_full[P.FIRST_EVAL - 1 : P.FIRST_EVAL + 2] = 1  # continues from warm-up: not an event
    y_full[P.FIRST_EVAL + 5 : P.FIRST_EVAL + 8] = 1
    ratio = np.zeros(10)
    ratio[6] = 2.0
    ev = P.evaluate_series(y_full, ratio)
    assert ev["events"] == 1 and ev["events_detected"] == 1 and ev["delays"] == [1]
    c = P.confusion(np.array([1, 0, 1, 0]), np.array([1, 1, 0, 0]))
    assert P.prf(c) == {"precision": 0.5, "recall": 0.5, "f1": 0.5, "false_latch_rate": 0.5}
    assert P.prf({"tp": 0, "fp": 0, "fn": 0, "tn": 0})["f1"] == 0.0
    ci = P.bootstrap_ci(np.array([0.1, 0.2, 0.3]))
    assert ci == P.bootstrap_ci(np.array([0.1, 0.2, 0.3]))
    assert P.wilcoxon_p(np.zeros(4)) is None
    assert 0 < P.wilcoxon_p(np.array([0.1, 0.2, -0.05, 0.3, 0.4, 0.2])) <= 1


def test_split_is_seeded_and_disjoint():
    ss = [P.Series(f"s{i}", "X", np.zeros((32 * 100, 1)), np.zeros(100, int)) for i in range(7)]
    a, b = P.split(ss)
    assert len(a) == 4 and len(b) == 3
    assert not {s.name for s in a} & {s.name for s in b}
    assert [s.name for s in a] == [s.name for s in P.split(ss)[0]]


def test_download_helpers(tmp_path, monkeypatch):
    monkeypatch.setenv("STREAMGUARD_BENCH_DATA", str(tmp_path))
    assert download.cache_dir() == tmp_path
    monkeypatch.delenv("STREAMGUARD_BENCH_DATA")
    assert download.cache_dir().name == "spark-streamguard-bench"
    files = download.nab_files()
    assert len(files) == 48 and all("ea702d75" in u for u, _ in files.values())
    smd = download.smd_files()
    assert len(smd) == 56 and all(download.SMD_COMMIT in u for u, _ in smd.values())
    f = tmp_path / "x.txt"
    f.write_text("abc")
    sha = download.sha256_file(f)
    download.verify({"x.txt": ("u", sha)}, tmp_path)
    with pytest.raises(ValueError):
        download.verify({"x.txt": ("u", "0" * 64)}, tmp_path)
    with pytest.raises(FileNotFoundError):
        download.verify({"y.txt": ("u", sha)}, tmp_path)


GATE_CASES = [
    ("ewma", {"lam": 0.2}, "ewma", {"lam": 0.2}),
    ("iforest", {}, "isoforest", {}),
    ("streamguard_diff", {"mode": "diff"}, "spark", {"reference": "diff"}),
    ("streamguard_detrend", {"mode": "detrend"}, "spark", {"reference": "detrend"}),
]


@pytest.mark.parametrize(("fam", "fparams", "scorer", "sparams"), GATE_CASES)
def test_gate_matches_raw_baselines(fixture_root, fam, fparams, scorer, sparams):
    """The gate around a scorer reproduces the raw M4 baseline block for block."""
    for s in load_nab(fixture_root, NAB_KEYS)[:2] + load_smd(fixture_root)[:1]:
        if scorer != "spark":
            sparams = {**sparams, "ref_window": P.trailing_len(len(s.x))}
        raw = P.raw_scores(P.Method("x", fam, fparams), s, {})
        rr, _ = P.ratios(raw, s.channels, 0.99, None)
        gr, info = G.gate_series(s, scorer, sparams, 0.99)
        assert info["not_calibrated_at_64"] == 0
        np.testing.assert_allclose(gr, rr, rtol=1e-9)
        assert ((gr >= 1) == (rr >= 1)).all()


def test_gate_series_without_channels(fixture_root):
    s = load_smd(fixture_root)[0]
    s.channels[:] = False
    r, info = G.gate_series(s, "ewma", {}, 0.99)
    assert info == {"channels": 0} and not r.any()


def test_fault_injection_helpers(fixture_root, monkeypatch):
    s = [x for x in load_nab(fixture_root, NAB_KEYS) if x.n_blocks >= P.MIN_BLOCKS][0]
    rng = np.random.default_rng(0)
    plan = pivot.fault_plan(s.n_blocks, rng)
    assert len(plan) == pivot.K_FAULTS and all(b > P.FIRST_EVAL for b, _ in plan)
    for kind in pivot.FAULTS:
        xf, fb = pivot.faulted_samples(s.x, kind, plan)
        if kind in ("nan", "inf"):
            assert fb == [b for b, _ in plan] and (~np.isfinite(xf)).sum() == len(plan)
        else:
            assert len(xf) % 32 == 0 and len(xf) < len(s.x)
        lab, fam, params, q = pivot.RAW[0]
        raw = pivot._robust_raw((s, lab, fam, params, q, kind, plan))
        assert raw["crash"] == 0 and raw["faults"] == len(plan)
        if kind == "nan":  # NaN propagates through the EWMA state: never flagged again
            assert raw["fault_flagged"] == 0 and raw["nonfinite_admitted"] > 0
        for with_seq in (False, True):
            lab, scorer, params, q = pivot.GATES[0]
            g = pivot._robust_gate((s, lab, scorer, params, q, kind, plan, with_seq))
            assert g["crash"] == 0 and g["nonfinite_admitted"] == 0
            if kind != "drop" or with_seq:
                assert g["fault_flagged"] == len(plan)
    for raw_method in pivot.RAW:  # raw baselines: NaN faults are never flagged (or crash)
        r = pivot._robust_raw((s, *raw_method, "nan", plan))
        assert r["crash"] == 1 or r["fault_flagged"] == 0

    def boom(series, x, *a):  # a detector that raises on faulted input is counted as a crash
        if not np.all(np.isfinite(x)):
            raise ValueError("Input contains NaN")
        return real(series, x, *a)

    real = pivot.raw_flags
    monkeypatch.setattr(pivot, "raw_flags", boom)
    r = pivot._robust_raw((s, *pivot.RAW[3], "nan", plan))
    assert r == {"crash": 1, "error": "ValueError: Input contains NaN"}


def test_pivot_tables_render_committed_results():
    report = json.loads((REAL / "pivot_results.json").read_text())
    md = pivot.render(report)
    for head in ("Default-scorer rule", "gate vs raw", "Fault injection", "Throughput"):
        assert head in md
    assert "**(chosen)**" in md and "SEQUENCE_GAP 48" in md
    assert pivot.render({}) == "# Post-M4 pivot results (REPORTED)\n"
