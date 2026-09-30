"""Render results.json as Markdown tables (``python benchmarks/real/make_tables.py``)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = (
    "sg_default",
    "streamguard_diff",
    "streamguard_median",
    "streamguard_detrend",
    "cusum",
    "ewma",
    "zscore",
    "iforest",
    "oes_latch",
    "oes_fixed",
)


def fmt(v, nd=3, pct=False):
    if v is None:
        return "-"
    return f"{100 * v:.1f}%" if pct else f"{v:.{nd}f}"


def dataset_tables(name: str, d: dict) -> str:
    lines = [
        f"**{name.upper()}**: {d['series_total']} series, {len(d['excluded_short'])} excluded "
        f"as too short (< 97 blocks), {len(d['tune'])} tune / {len(d['test'])} test; test "
        f"evaluation blocks {d['test_eval_blocks']}, positive {d['test_positive_blocks']}.",
        "",
        "| Method (setting chosen on tune split) | q | P | R | F1 | mean F1 | mean AP "
        "| False-latch rate | Event recall | Median delay (blocks) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for fam in ORDER:
        if fam not in d["results"]:
            continue
        r = d["results"][fam]
        t = r["test"]
        q = "fixed tau" if r["q"] is None else r["q"]
        lines.append(
            f"| {r['setting']} | {q} | {fmt(t['precision'])} | {fmt(t['recall'])} "
            f"| **{fmt(t['f1'])}** | {fmt(t['mean_f1'])} | {fmt(t['mean_ap'])} "
            f"| {fmt(t['false_latch_rate'], pct=True)} | {fmt(t['event_recall'], pct=True)} "
            f"| {fmt(t['median_delay_blocks'], 1)} |"
        )
    lines += [
        "",
        f"Significance vs the best baseline on test ({d['best_baseline']}), paired "
        "per test series with at least one positive evaluation block (difference = "
        "StreamGuard minus baseline; 95% bootstrap CI, 10,000 resamples; two-sided "
        "Wilcoxon signed-rank):",
        "",
        "| StreamGuard setting | Metric | n | Mean diff | 95% CI | Wilcoxon p | W/L |",
        "|---|---|---|---|---|---|---|",
    ]
    for fam, sig in d["significance_vs_best_baseline"].items():
        for metric, s in sig.items():
            if s.get("n", 0) == 0:
                continue
            p = "-" if s["wilcoxon_p"] is None else f"{s['wilcoxon_p']:.3f}"
            lines.append(
                f"| {d['results'][fam]['setting']} | {metric.upper()} | {s['n']} "
                f"| {s['mean_diff']:+.3f} | [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}] | {p} "
                f"| {s['wins']}/{s['losses']} |"
            )
    return "\n".join(lines)


def perf_tables(rep: dict) -> str:
    out = []
    hw = rep["hardware"]
    out.append(
        f"Hardware: {hw['cpu']}, {hw['logical_cpus']} logical CPUs, single thread, "
        f"Python {hw['python']}, numpy {hw['numpy']}."
    )
    if "streaming_throughput_nab" in rep:
        t = rep["streaming_throughput_nab"]
        out.append(
            f"\n`StreamGuard.process()` streaming throughput on NAB blocks: "
            f"{t['blocks_per_s']:,.0f} blocks/s ({t['blocks']:,} blocks)."
        )
    for name, d in rep["datasets"].items():
        out += [
            "",
            f"Batch scoring throughput, {name.upper()} (blocks x channels per second, "
            "scores only, test series, chosen settings; batch numpy "
            "implementations):",
            "",
            "| Family | per s |",
            "|---|---|",
        ]
        for fam, v in sorted(d["throughput_scoring"].items()):
            out.append(f"| {fam} | {v['per_s']:,.0f} |")
    if "memory_nab_longest" in rep:
        m = rep["memory_nab_longest"]
        out += [
            "",
            f"Memory (tracemalloc, longest NAB series {m['series']}, {m['blocks']} blocks):",
            "",
            "| What | Retained | Peak |",
            "|---|---|---|",
        ]
        for k, v in m.items():
            if isinstance(v, dict):
                ret = f"{v['retained_bytes'] / 1024:.1f} KiB" if "retained_bytes" in v else "-"
                out.append(f"| {k} | {ret} | {v['peak_bytes'] / 1024:.1f} KiB |")
    if "nab_official_score" in rep:
        n = rep["nab_official_score"]
        out += [
            "",
            f"NAB official scorer, {n['series']} NAB test series (normalised; see caveats):",
            "",
            "| Detector | standard | low FP | low FN |",
            "|---|---|---|---|",
        ]
        for d, v in n["normalized"].items():
            out.append(
                f"| {d} | {v['standard']} | {v['reward_low_FP_rate']} | {v['reward_low_FN_rate']} |"
            )
    return "\n".join(out)


def render(rep: dict) -> str:
    parts = [f"Date {rep['date']}, StreamGuard {rep['streamguard_version']}. REPORTED."]
    for name, d in rep["datasets"].items():
        parts += ["", dataset_tables(name, d)]
    parts += ["", perf_tables(rep)]
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    src = Path(argv[0]) if argv else HERE / "results.json"
    md = render(json.loads(src.read_text(encoding="utf-8")))
    (src.parent / "results.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
