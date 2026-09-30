"""Post-M4 pivot benchmark: StreamGuard as a fail-closed gate around any scorer (REPORTED).

Three phases, all through the shipped ``StreamGuard.process`` (see ``gate.py``) and the M4
protocol, data, split (seed 42) and metrics (``protocol.py``):

``choose``  Default-scorer rule, **stated before it was run** (2026-09-29) and applied to the
            M4 *tuning* splits only:

            * Candidates (numpy-only built-in scorers, shipped parameters, ``warmup_blocks=64``):
              ``spark`` with reference ``diff``, ``median``, ``detrend``; ``ewma`` with
              ``lam`` in {0.1, 0.2, 0.3} and ``ref_window=288``; each with quantile ``q`` in
              {0.99, 0.995, 0.999}, in that order (scorer list outer, q inner).
            * Criterion: mean over {NAB, SMD} of the tuning-split pooled F1 (each dataset
              weighted 1/2). Highest wins; ties (difference < 1e-9) go to the earlier candidate.
            * ``isoforest`` is not a candidate (optional scikit-learn dependency).
            * The winner's scorer, parameters and ``q`` become the library defaults.

``test``    The M4 test protocol, run once: StreamGuard+ewma and StreamGuard+isoforest with the
            M4-tuned settings of the raw EWMA / Isolation Forest baselines (per dataset; the
            trailing-median window is the M4 per-series ``L``), next to the raw baselines
            recomputed from ``protocol.py``; plus the chosen default with shipped parameters.
            Reports pooled metrics and block-level flag disagreements (gate vs raw).

``robust``  Fault injection on the NAB test series (seed 42). For each fault type, a separate
            run per series with ``K = 3`` faults at distinct evaluation blocks:

            * ``nan``: one sample of the block set to NaN;  ``inf``: one sample set to +inf;
            * ``short``: one sample of the block missing (31 values);
            * ``drop``: the whole 32-sample block missing.

            Raw baselines consume the faulted *sample stream* (their natural input; ``short`` and
            ``drop`` shift every later sample) with the M4 thresholds (``score >= tau``; NaN
            compares false). StreamGuard consumes producer-framed blocks (``short`` arrives as a
            31-value block, ``drop`` as a missing block), with and without sequence numbers.
            Measured: crashes, fault blocks flagged / passed, blocks admitted with a non-finite
            score, and decision changes on clean blocks versus the fault-free run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from streamguard import __version__

import gate as G
import protocol as P
from data import LOADERS
from download import cache_dir
from run import git_commit

HERE = Path(__file__).resolve().parent
OUT = HERE / "pivot_results.json"

CANDIDATES = [
    ("spark", {"reference": "diff"}),
    ("spark", {"reference": "median"}),
    ("spark", {"reference": "detrend"}),
    ("ewma", {"lam": 0.1, "ref_window": 288}),
    ("ewma", {"lam": 0.2, "ref_window": 288}),
    ("ewma", {"lam": 0.3, "ref_window": 288}),
]
TIE = 1e-9
FAULTS = ("nan", "inf", "short", "drop")
K_FAULTS = 3
WORKERS = 4


def label(scorer: str, params: dict, q: float | None = None) -> str:
    p = ", ".join(f"{k}={v}" for k, v in params.items())
    return f"{scorer} ({p})" + (f" q={q}" if q is not None else "")


def load(root: Path) -> dict[str, tuple[list[P.Series], list[P.Series]]]:
    out = {}
    for name in ("nab", "smd"):
        kept = [s for s in LOADERS[name](root) if s.n_blocks >= P.MIN_BLOCKS]
        out[name] = P.split(kept)
    return out


def _gate_eval(args: tuple) -> dict:
    s, scorer, params, q = args
    ratio, info = G.gate_series(s, scorer, params, q)
    return {**P.evaluate_series(s.y, ratio), "info": info, "flags": (ratio >= 1).tolist()}


def gate_rows(pool, series, scorer, params, q) -> list[dict]:
    return list(pool.map(_gate_eval, [(s, scorer, params, q) for s in series]))


# --------------------------------------------------------------------------- choose


def phase_choose(data, pool) -> dict:
    table = []
    for scorer, params in CANDIDATES:
        for q in P.QUANTILES:
            f1 = {}
            for ds, (tune, _) in data.items():
                f1[ds] = P.pooled(gate_rows(pool, tune, scorer, params, q))["f1"]
            table.append(
                {
                    "scorer": scorer,
                    "params": params,
                    "q": q,
                    "f1": f1,
                    "mean": float(np.mean(list(f1.values()))),
                }
            )
            print(label(scorer, params, q), f1, flush=True)
    best = table[0]
    for row in table[1:]:
        if row["mean"] > best["mean"] + TIE:
            best = row
    return {
        "rule": __doc__.split("``choose``")[1].split("``test``")[0].strip(),
        "table": table,
        "winner": best,
    }


# --------------------------------------------------------------------------- test


def _raw_eval(args: tuple) -> dict:
    s, fam, params, q = args
    sc = P.raw_scores(P.Method("raw", fam, params), s, {})
    ratio, _ = P.ratios(sc, s.channels, q, None)
    return {**P.evaluate_series(s.y, ratio), "flags": (ratio >= 1).tolist()}


def summarize(rows: list[dict]) -> dict:
    return {
        k: v
        for k, v in P.pooled(rows).items()
        if k
        in (
            "tp",
            "fp",
            "fn",
            "tn",
            "precision",
            "recall",
            "f1",
            "mean_ap",
            "event_recall",
            "median_delay_blocks",
        )
    }


def phase_test(data, pool, m4: dict, default: dict) -> dict:
    out = {}
    for ds, (_, test) in data.items():
        res = m4["datasets"][ds]["results"]
        pairs = [
            ("ewma", "ewma", res["ewma"]),
            ("iforest", "isoforest", res["iforest"]),
        ]
        rows_ds = {}
        for fam, scorer, chosen in pairs:
            q = chosen["q"]
            lam = (
                {"lam": float(chosen["setting"].split("lam=")[1].rstrip(")"))}
                if fam == "ewma"
                else {}
            )
            raw = list(pool.map(_raw_eval, [(s, fam, lam, q) for s in test]))
            gate = list(
                pool.map(
                    _gate_eval,
                    [(s, scorer, {**lam, "ref_window": P.trailing_len(len(s.x))}, q) for s in test],
                )
            )
            disagree = sum(
                int(np.sum(np.array(a["flags"]) != np.array(b["flags"])))
                for a, b in zip(raw, gate, strict=True)
            )
            rows_ds[fam] = {
                "setting": chosen["setting"],
                "q": q,
                "raw": summarize(raw),
                "gate": summarize(gate),
                "m4_reported_raw": {k: chosen["test"][k] for k in ("f1", "mean_ap")},
                "flag_disagreements": disagree,
                "eval_blocks": sum(len(r["flags"]) for r in raw),
                "gate_reasons": dict(
                    sum((Counter(g["info"].get("reasons", {})) for g in gate), Counter())
                ),
            }
            print(
                ds, fam, rows_ds[fam]["raw"]["f1"], rows_ds[fam]["gate"]["f1"], disagree, flush=True
            )
        g = gate_rows(pool, test, default["scorer"], default["params"], default["q"])
        rows_ds["default"] = {
            "setting": label(default["scorer"], default["params"], default["q"]),
            "gate": summarize(g),
        }
        out[ds] = rows_ds
    return out


# --------------------------------------------------------------------------- robust


def fault_plan(n_blocks: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """``K_FAULTS`` (block, sample offset) pairs, distinct eval blocks >= 2 apart, sorted."""
    lo, hi = P.FIRST_EVAL + 1, n_blocks - 2
    while True:
        b = np.sort(rng.choice(np.arange(lo, hi), size=K_FAULTS, replace=False))
        if np.all(np.diff(b) >= 2):
            return [(int(x), int(rng.integers(0, P.BLOCK))) for x in b]


def faulted_samples(x: np.ndarray, kind: str, plan) -> tuple[np.ndarray, list[int]]:
    """Faulted sample stream and the block index (in that stream) of each fault / splice."""
    x = x.copy()
    if kind in ("nan", "inf"):
        for b, o in plan:
            x[b * P.BLOCK + o, 0] = np.nan if kind == "nan" else np.inf
        return x, [b for b, _ in plan]
    drop = []
    for b, o in plan:
        drop += (
            [b * P.BLOCK + o] if kind == "short" else list(range(b * P.BLOCK, (b + 1) * P.BLOCK))
        )
    keep = np.ones(len(x), bool)
    keep[drop] = False
    xs = x[keep]
    n = len(xs) // P.BLOCK
    # splice block = where the first sample after the removed one(s) lands
    shift = 1 if kind == "short" else P.BLOCK
    return xs[: n * P.BLOCK], [
        (b * P.BLOCK + o - i * shift) // P.BLOCK if kind == "short" else b - i
        for i, (b, o) in enumerate(plan)
    ]


RAW = [
    ("EWMA chart (lam=0.2, q=0.99)", "ewma", {"lam": 0.2}, 0.99),
    ("CUSUM (k=2.0, q=0.99)", "cusum", {"k": 2.0}, 0.99),
    ("Rolling z-score (window=64, q=0.99)", "zscore", {"window": 64}, 0.99),
    ("Isolation Forest (q=0.99)", "iforest", {}, 0.99),
]


def raw_flags(s: P.Series, x: np.ndarray, fam: str, params: dict, q: float):
    ss = P.Series(s.name, s.dataset, x, np.zeros(len(x) // P.BLOCK, int), s.channels)
    sc = P.raw_scores(P.Method("raw", fam, params), ss, {})[:, 0]
    tau = max(float(np.quantile(sc[1 : P.FIRST_EVAL], q, method="linear")), P.TAU_FLOOR)
    with np.errstate(invalid="ignore"):
        flag = sc >= tau  # NaN compares false: a plain threshold admits it
    return sc, flag


def _robust_raw(args: tuple) -> dict:
    s, lab, fam, params, q, kind, plan = args
    _, clean = raw_flags(s, s.x, fam, params, q)
    xf, fblocks = faulted_samples(s.x, kind, plan)
    try:
        sc, flag = raw_flags(s, xf, fam, params, q)
    except Exception as exc:  # noqa: BLE001 - measuring crashes is the point
        return {"crash": 1, "error": type(exc).__name__ + ": " + str(exc)[:120]}
    ev = slice(P.FIRST_EVAL, None)
    nonfinite_admitted = int(np.sum(~np.isfinite(sc[ev]) & ~flag[ev]))
    changes = None
    if kind in ("nan", "inf"):
        clean_blocks = np.ones(len(flag), bool)
        clean_blocks[fblocks] = False
        clean_blocks[: P.FIRST_EVAL] = False
        changes = int(np.sum((flag != clean) & clean_blocks))
    return {
        "crash": 0,
        "faults": len(fblocks),
        "fault_flagged": int(sum(bool(flag[b]) for b in fblocks)),
        "fault_nonfinite_score": int(sum(not np.isfinite(sc[b]) for b in fblocks)),
        "nonfinite_admitted": nonfinite_admitted,
        "clean_decision_changes": changes,
    }


def gate_feed(kind: str, plan, with_seq: bool):
    faults = {b: o for b, o in plan}

    def feed(xb_c, first):
        for b in range(first, xb_c.shape[0]):
            seq = b if with_seq else None
            blk = xb_c[b]
            if b in faults:
                o = faults[b]
                if kind == "drop":
                    continue
                blk = blk.copy()
                if kind == "short":
                    blk = np.delete(blk, o)
                else:
                    blk[o] = np.nan if kind == "nan" else np.inf
            yield blk, seq, b

    return feed


GATES = [
    ("StreamGuard+ewma (lam=0.2, q=0.99)", "ewma", {"lam": 0.2}, 0.99),
    ("StreamGuard+spark (diff, q=0.99)", "spark", {"reference": "diff"}, 0.99),
    ("StreamGuard+isoforest (q=0.99)", "isoforest", {}, 0.99),
]


def _robust_gate(args: tuple) -> dict:
    s, lab, scorer, params, q, kind, plan, with_seq = args
    params = {**params, "ref_window": P.trailing_len(len(s.x))} if scorer != "spark" else params
    xb = P.blocks_of(s.x)[:, :, 0]
    clean, _, _ = G.gate_channel(xb, scorer, params, q)
    try:
        by_block: dict = {}
        r, reasons, _ = G.gate_channel(
            xb, scorer, params, q, gate_feed(kind, plan, with_seq), by_block
        )
    except Exception as exc:  # noqa: BLE001
        return {"crash": 1, "error": type(exc).__name__ + ": " + str(exc)[:120]}
    fb = [b for b, _ in plan]
    ev = np.zeros(len(r), bool)
    ev[P.FIRST_EVAL :] = True
    # the dropped block does not exist; judge the block after it
    judged = [b + 1 for b in fb] if kind == "drop" else fb
    others = ev.copy()
    others[fb] = False
    if kind == "drop":
        others[judged] = False
    changes = int(np.sum(((r >= 1) != (clean >= 1)) & others))
    return {
        "crash": 0,
        "faults": len(fb),
        "fault_flagged": int(sum(bool(r[b] >= 1) for b in judged)),
        "nonfinite_admitted": reasons.get("ADMITTED_NONFINITE", 0),
        "clean_decision_changes": changes,
        "reasons": dict(reasons),
        "fault_reasons": dict(Counter(by_block.get(b, "NOT_FED") for b in judged)),
    }


def phase_robust(data, pool) -> dict:
    _, test = data["nab"]
    rng = np.random.default_rng(P.SEED)
    plans = {k: {s.name: fault_plan(s.n_blocks, rng) for s in test} for k in FAULTS}
    out = {"k_faults_per_series": K_FAULTS, "series": len(test), "plans": plans, "rows": []}
    jobs = []
    for kind in FAULTS:
        for lab, fam, params, q in RAW:
            jobs.append(
                (
                    "raw",
                    lab,
                    kind,
                    None,
                    [(s, lab, fam, params, q, kind, plans[kind][s.name]) for s in test],
                )
            )
        for lab, scorer, params, q in GATES:
            for with_seq in (False, True):
                if with_seq and kind != "drop":
                    continue
                jobs.append(
                    (
                        "gate",
                        lab,
                        kind,
                        with_seq,
                        [
                            (s, lab, scorer, params, q, kind, plans[kind][s.name], with_seq)
                            for s in test
                        ],
                    )
                )
    for typ, lab, kind, with_seq, args in jobs:
        fn = _robust_raw if typ == "raw" else _robust_gate
        rows = list(pool.map(fn, args))
        agg: dict = {"method": lab + (" +seq" if with_seq else ""), "type": typ, "fault": kind}
        agg["crashed_series"] = sum(r["crash"] for r in rows)
        ok = [r for r in rows if not r["crash"]]
        agg["errors"] = sorted({r["error"] for r in rows if r["crash"]})
        for key in ("faults", "fault_flagged", "nonfinite_admitted"):
            agg[key] = sum(r[key] for r in ok)
        ch = [r["clean_decision_changes"] for r in ok if r["clean_decision_changes"] is not None]
        agg["clean_decision_changes"] = sum(ch) if ch else None
        if typ == "raw":
            agg["fault_nonfinite_score"] = sum(r["fault_nonfinite_score"] for r in ok)
        else:
            agg["reasons"] = dict(sum((Counter(r["reasons"]) for r in ok), Counter()))
            agg["fault_reasons"] = dict(sum((Counter(r["fault_reasons"]) for r in ok), Counter()))
        out["rows"].append(agg)
        print(agg["method"], kind, agg["crashed_series"], agg.get("fault_flagged"), flush=True)
    return out


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("phase", choices=["choose", "test", "robust"])
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--workers", type=int, default=WORKERS)
    args = p.parse_args(argv)
    data = load(args.data_dir or cache_dir())
    report = json.loads(args.out.read_text()) if args.out.exists() else {}
    report.update(
        {"evidence": "REPORTED (public data, this code)", "streamguard_version": __version__}
    )
    with ProcessPoolExecutor(args.workers) as pool:
        if args.phase == "choose":
            res = phase_choose(data, pool)
        elif args.phase == "test":
            m4 = json.loads((HERE / "results.json").read_text())
            res = phase_test(data, pool, m4, report["choose"]["winner"])
        else:
            res = phase_robust(data, pool)
    report[args.phase] = {"date": dt.date.today().isoformat(), "code_commit": git_commit(), **res}
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
