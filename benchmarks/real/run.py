"""Real-data benchmark runner (REPORTED numbers). See protocol.py for the fixed protocol.

Usage::

    python benchmarks/real/download.py          # once; data cached outside the repo
    python benchmarks/real/run.py               # NAB + SMD -> results.json + results.md

``--data-dir`` points at a cache or at a fixture directory with the same layout (tests use a
tiny fixture). Requires the optional ``bench`` extra (scikit-learn, scipy).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from streamguard import StreamGuard, __version__  # noqa: E402

import protocol as P  # noqa: E402
from data import LOADERS  # noqa: E402
from download import cache_dir  # noqa: E402

BASELINE_FAMILIES = ("cusum", "ewma", "zscore", "iforest", "oes_latch", "oes_fixed")
SG_FAMILIES = ("sg_default", "streamguard_diff", "streamguard_median", "streamguard_detrend")


def candidates(method: P.Method) -> list[tuple[P.Method, float | None]]:
    if method.fixed_tau is not None:
        return [(method, None)]
    if method.fixed_q is not None:
        return [(method, method.fixed_q)]
    return [(method, q) for q in P.QUANTILES]


def run_series(s: P.Series, methods: list[P.Method]) -> dict:
    """Per-series evaluation dict for every (method, q) candidate."""
    cache: dict = {}
    out = {}
    for m in methods:
        sc = P.raw_scores(m, s, cache)
        for mm, q in candidates(m):
            ratio, used = P.ratios(sc, s.channels, q, mm.fixed_tau)
            out[(mm.label(), mm.family, q)] = {**P.evaluate_series(s.y, ratio), "channels": used}
    return out


def batch_throughput(series: list[P.Series], chosen: dict, methods: list[P.Method]) -> dict:
    """Scoring time per chosen setting on the test series, fresh cache (nothing shared)."""
    out = {}
    for fam, (key, _) in chosen.items():
        m = next(mm for mm in methods if mm.label() == key[0] and mm.family == fam)
        n, t = 0, 0.0
        for s in series:
            t0 = time.perf_counter()
            P.raw_scores(m, s, {})
            t += time.perf_counter() - t0
            n += (s.n_blocks - 1) * int(s.channels.sum())
        out[fam] = {"blocks_x_channels": n, "seconds": t, "per_s": n / t if t else None}
    return out


def select(tune_rows: dict[str, dict], methods: list[P.Method]) -> dict:
    """Per family, the candidate with the highest pooled F1 on the tuning split."""
    chosen = {}
    for m in methods:
        for mm, q in candidates(m):
            key = (mm.label(), mm.family, q)
            f1 = P.pooled([rows[key] for rows in tune_rows.values()])["f1"]
            best = chosen.get(mm.family)
            if best is None or f1 > best[1]:
                chosen[mm.family] = (key, f1)
    return chosen


def significance(test_rows: dict, key_a, key_b, metric: str) -> dict:
    a, b = [], []
    for rows in test_rows.values():
        ra, rb = rows[key_a], rows[key_b]
        if ra["positives"] == 0:
            continue
        va, vb = ra[metric], rb[metric]
        if va is None or vb is None:
            continue
        a.append(va)
        b.append(vb)
    d = np.array(a) - np.array(b)
    if len(d) == 0:
        return {"n": 0}
    return {
        "n": len(d),
        "mean_diff": float(d.mean()),
        "ci95": P.bootstrap_ci(d),
        "wilcoxon_p": P.wilcoxon_p(d),
        "wins": int((d > 0).sum()),
        "losses": int((d < 0).sum()),
    }


def streaming_throughput(series: list[P.Series], max_blocks: int = 200_000) -> dict:
    """StreamGuard.process() on real blocks, one stream per channel, single thread."""
    blocks = []
    for s in series:
        xb = P.blocks_of(s.x)
        for c in np.flatnonzero(s.channels):
            blocks.append(xb[:, :, c])
            if sum(len(b) for b in blocks) >= max_blocks:
                break
    n = 0
    t0 = time.perf_counter()
    for arr in blocks:
        g = StreamGuard()
        for blk in arr:
            g.process(blk)
        n += len(arr)
    el = time.perf_counter() - t0
    return {"blocks": n, "seconds": el, "blocks_per_s": n / el}


def memory_profile(s: P.Series, methods: list[P.Method]) -> dict:
    """Peak traced allocation while scoring one series; StreamGuard retained per-stream state."""
    out = {}
    x1 = P.blocks_of(s.x)[:, :, 0]
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    g = StreamGuard()
    for blk in x1:
        g.process(blk)
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del g
    out["StreamGuard.process (per stream)"] = {
        "retained_bytes": cur - base,
        "peak_bytes": peak - base,
    }
    uni = P.Series(s.name, s.dataset, s.x[:, :1], s.y, s.channels[:1])
    for m in methods:
        tracemalloc.start()
        P.raw_scores(m, uni, {})
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        out[m.label() + " (batch scoring)"] = {"peak_bytes": peak}
    return out


def hardware() -> dict:
    cpu = platform.processor() or ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:  # pragma: no cover
        pass
    return {
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "threads_used": 1,
    }


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        return None


def run_dataset(name: str, series: list[P.Series], methods: list[P.Method]) -> dict:
    excluded = [s.name for s in series if s.n_blocks < P.MIN_BLOCKS]
    kept = [s for s in series if s.n_blocks >= P.MIN_BLOCKS]
    tune, test = P.split(kept)
    tune_rows = {s.name: run_series(s, methods) for s in tune}
    test_rows = {s.name: run_series(s, methods) for s in test}
    chosen = select(tune_rows, methods)
    tuning_table = {
        f"{lab} q={q}": P.pooled([rows[(lab, fam, q)] for rows in tune_rows.values()])["f1"]
        for (lab, fam, q) in next(iter(tune_rows.values()))
    }

    results = {}
    for fam, (key, tune_f1) in chosen.items():
        agg = P.pooled([rows[key] for rows in test_rows.values()])
        results[fam] = {"setting": key[0], "q": key[2], "tune_f1": tune_f1, "test": agg}

    base_fams = [f for f in BASELINE_FAMILIES if f in results]
    best_base = max(base_fams, key=lambda f: results[f]["test"]["f1"]) if base_fams else None
    sig = {}
    if best_base:
        kb = chosen[best_base][0]
        for fam in SG_FAMILIES:
            if fam in chosen:
                ka = chosen[fam][0]
                sig[fam] = {
                    metric: significance(test_rows, ka, kb, metric) for metric in ("f1", "ap")
                }
    all_s = tune + test
    contaminated = sum(int(s.y[: P.FIRST_EVAL].any()) for s in all_s)
    return {
        "series_total": len(series),
        "excluded_short": excluded,
        "tune": [s.name for s in tune],
        "test": [s.name for s in test],
        "channels_total": int(sum(s.x.shape[1] for s in kept)),
        "channels_constant_in_warmup": int(sum((~s.channels).sum() for s in kept)),
        "series_with_warmup_labels": contaminated,
        "test_positive_blocks": int(sum(s.y[P.FIRST_EVAL :].sum() for s in test)),
        "test_eval_blocks": int(sum(len(s.y) - P.FIRST_EVAL for s in test)),
        "tuning_pooled_f1": tuning_table,
        "results": results,
        "best_baseline": best_base,
        "significance_vs_best_baseline": sig,
        "throughput_scoring": batch_throughput(test, chosen, methods),
        "per_series_test": {
            sname: {
                fam: {k: v for k, v in rows[chosen[fam][0]].items() if k != "delays"}
                for fam in chosen
            }
            for sname, rows in test_rows.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Real-data benchmark (REPORTED).")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--datasets", nargs="+", default=["nab", "smd"], choices=sorted(LOADERS))
    p.add_argument("--out", type=Path, default=HERE / "results.json")
    p.add_argument("--no-perf", action="store_true", help="skip throughput/memory runs")
    p.add_argument("--nab-keys", nargs="+", default=None, help="subset of NAB keys (fixtures)")
    args = p.parse_args(argv)
    root = args.data_dir or cache_dir()
    methods = P.method_grid() + P.FIXED_METHODS
    report = {
        "evidence": "REPORTED (public data, this code; settings chosen on tuning splits only)",
        "date": dt.date.today().isoformat(),
        "streamguard_version": __version__,
        "code_commit": git_commit(),
        "protocol": P.__doc__,
        "hardware": hardware(),
        "datasets": {},
    }
    loaded = {}
    for name in args.datasets:
        series = LOADERS[name](root, args.nab_keys) if name == "nab" else LOADERS[name](root)
        loaded[name] = series
        report["datasets"][name] = run_dataset(name, series, methods)
        print(f"{name}: done", flush=True)
    if not args.no_perf and "nab" in loaded:
        kept = [s for s in loaded["nab"] if s.n_blocks >= P.MIN_BLOCKS]
        report["streaming_throughput_nab"] = streaming_throughput(kept)
        longest = max(kept, key=lambda s: s.n_blocks)
        report["memory_nab_longest"] = {
            "series": longest.name,
            "blocks": longest.n_blocks,
            **memory_profile(longest, methods),
        }
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
