"""SYNTHETIC warm-up length x quantile sweep for the ``diff`` reference.

Usage::

    python benchmarks/warmup_sweep.py tune      # tuning set: sweep + rule -> chosen setting
    python benchmarks/warmup_sweep.py heldout   # held-out set: chosen setting + baselines
    python benchmarks/warmup_sweep.py           # both

All data are generated here (seeds derived from 42). Results are SYNTHETIC and say nothing
about real-world performance.

Fair corpus design (fixes the earlier 32-block check, where a longer warm-up left fewer spikes
to score): every series has 166 blocks of 32 samples. Block 0 seeds the reference and blocks
1..W are the warm-up for warm-up length W (at most 64). Every configuration is scored on the
SAME evaluation region, blocks 66..165 (100 blocks), which is after the longest warm-up. All
injected events are inside that region: 6 spike blocks per series (one spike of +-U(6, 12)
noise sd each) and, for the step family, one level jump of +-U(8, 15) in a block in [80, 150]
(its neighbours never get a spike). The warm-up region is clean.

Families (10 series each, 40 per set): random walk (N(0, 1) steps), sine plus noise
(5*sin(2*pi*t/P), P ~ U(150, 600), N(0, 1) noise), stationary N(0, 1) noise, and step change.
Tuning seeds: ``[42, 0, family, k]``; held-out seeds: ``[42, 1, family, k]``, k = 0..9.

Metrics on the evaluation region: recall = spike blocks latched with SCORE_OVER_TAU / spike
blocks; false-latch rate = other blocks latched with SCORE_OVER_TAU / other blocks (the step
block is excluded from both and reported as "steps caught").

Selection rule, fixed before any held-out result was computed:

1. Among configurations with false-latch rate <= 1% on the tuning set, pick the highest
   recall (ties: lower false-latch rate, then shorter warm-up).
2. If no configuration meets 1%, pick the lowest false-latch rate among configurations with
   recall >= 90% (ties: higher recall, then shorter warm-up), and state that the 1% target is
   not met.
3. If no configuration has recall >= 90% either, pick the highest recall (ties: lower false
   rate, then shorter warm-up).

The chosen setting is then evaluated once on the held-out set.
"""

from __future__ import annotations

import sys

import numpy as np

from streamguard import StreamGuard

N_BLOCKS = 166
EVAL_START = 66
N_SPIKES = 6
FAMILIES = ("random_walk", "sine_noise", "stationary", "step")
PER_FAMILY = 10
WARMUPS = (16, 24, 32, 48, 64)
QUANTILES = (0.99, 0.995, 0.999)
FALSE_TARGET = 0.01
RECALL_FLOOR = 0.90
TUNING, HELDOUT = 0, 1

BASELINES = {
    "diff, W=16, q=0.99 (previous eval setting)": {
        "reference": "diff",
        "warmup_blocks": 16,
        "quantile": 0.99,
    },
    "median rb=4, W=16, q=0.99 (original default ref)": {
        "reference": "median",
        "warmup_blocks": 16,
        "quantile": 0.99,
    },
}


def make_series(split: int, fam: int, k: int) -> dict:
    rng = np.random.default_rng([42, split, fam, k])
    family = FAMILIES[fam]
    n = N_BLOCKS * 32
    t = np.arange(n)
    step = None
    if family == "random_walk":
        x = np.cumsum(rng.normal(0.0, 1.0, n))
    elif family == "sine_noise":
        x = 5.0 * np.sin(2 * np.pi * t / rng.uniform(150, 600)) + rng.normal(0.0, 1.0, n)
    else:
        x = rng.normal(0.0, 1.0, n)
    if family == "step":
        step = int(rng.integers(80, 151))
        x[step * 32 + int(rng.integers(0, 32)) :] += rng.choice([-1.0, 1.0]) * rng.uniform(8, 15)
    banned = set() if step is None else {step - 1, step, step + 1}
    cand = [b for b in range(EVAL_START, N_BLOCKS) if b not in banned]
    spikes = sorted(int(b) for b in rng.choice(cand, size=N_SPIKES, replace=False))
    for b in spikes:
        x[b * 32 + int(rng.integers(0, 32))] += rng.choice([-1.0, 1.0]) * rng.uniform(6, 12)
    return {"family": family, "x": x.reshape(N_BLOCKS, 32), "spikes": spikes, "step": step}


def corpus(split: int) -> list[dict]:
    return [make_series(split, f, k) for f in range(len(FAMILIES)) for k in range(PER_FAMILY)]


def evaluate(data: list[dict], **cfg) -> dict:
    tot = {"spikes": 0, "caught": 0, "negatives": 0, "false": 0, "steps": 0, "step_caught": 0}
    for s in data:
        g = StreamGuard(scorer="spark", **cfg)  # 0.1.0.dev0 scorer (sweep predates M4)
        latched = set()
        for b in range(N_BLOCKS):
            d = g.process(s["x"][b])
            if b >= EVAL_START and d.reason.value == "SCORE_OVER_TAU":
                latched.add(b)
        spikes, step = set(s["spikes"]), s["step"]
        tot["spikes"] += len(spikes)
        tot["caught"] += len(spikes & latched)
        tot["false"] += len(latched - spikes - {step})
        tot["negatives"] += N_BLOCKS - EVAL_START - len(spikes) - int(step is not None)
        tot["steps"] += int(step is not None)
        tot["step_caught"] += int(step is not None and step in latched)
    tot["recall"] = tot["caught"] / tot["spikes"]
    tot["false_rate"] = tot["false"] / tot["negatives"]
    return tot


def sweep(data: list[dict]) -> dict:
    return {
        (w, q): evaluate(data, reference="diff", warmup_blocks=w, quantile=q)
        for w in WARMUPS
        for q in QUANTILES
    }


def choose(results: dict) -> tuple[tuple[int, float], str]:
    """Apply the selection rule from the module docstring."""
    ok = [k for k, r in results.items() if r["false_rate"] <= FALSE_TARGET]
    if ok:
        best = min(ok, key=lambda k: (-results[k]["recall"], results[k]["false_rate"], k[0]))
        return best, "rule 1: max recall with false-latch rate <= 1% (target met)"
    floor = [k for k, r in results.items() if r["recall"] >= RECALL_FLOOR]
    if floor:
        best = min(floor, key=lambda k: (results[k]["false_rate"], -results[k]["recall"], k[0]))
        return best, "rule 2: no setting meets 1%; lowest false-latch rate with recall >= 90%"
    best = min(results, key=lambda k: (-results[k]["recall"], results[k]["false_rate"], k[0]))
    return best, "rule 3: no setting has recall >= 90%; highest recall"


def row(label: str, r: dict) -> str:
    steps = f"{r['step_caught']}/{r['steps']}" if r["steps"] else "-"
    return (
        f"| {label} | {r['caught']}/{r['spikes']} ({100 * r['recall']:.1f}%) "
        f"| {r['false']}/{r['negatives']} ({100 * r['false_rate']:.2f}%) | {steps} |"
    )


HEADER = (
    "| Setting | Spikes caught (recall) | False latches (rate) | Steps caught |\n|---|---|---|---|"
)


def sweep_table(results: dict) -> str:
    lines = [HEADER]
    lines += [row(f"diff, W={w}, q={q}", r) for (w, q), r in results.items()]
    return "\n".join(lines)


def tune() -> tuple[tuple[int, float], str, dict]:
    results = sweep(corpus(TUNING))
    best, why = choose(results)
    return best, why, results


def heldout(best: tuple[int, float]) -> dict:
    data = corpus(HELDOUT)
    w, q = best
    out = {
        f"chosen: diff, W={w}, q={q}": evaluate(data, reference="diff", warmup_blocks=w, quantile=q)
    }
    for label, cfg in BASELINES.items():
        out[label] = evaluate(data, **cfg)
    per_family = {
        fam: evaluate(
            [s for s in data if s["family"] == fam], reference="diff", warmup_blocks=w, quantile=q
        )
        for fam in FAMILIES
    }
    return {"overall": out, "per_family": per_family}


def main(argv: list[str]) -> None:
    mode = argv[1] if len(argv) > 1 else "both"
    print("SYNTHETIC warm-up x quantile sweep, diff reference (seed 42; generated data only)\n")
    best, why, results = tune()
    print("Tuning set (40 series, evaluation blocks 66..165):\n")
    print(sweep_table(results))
    print(f"\nChosen: W={best[0]}, q={best[1]} by {why}")
    if mode == "tune":
        return
    ho = heldout(best)
    print("\nHeld-out set (40 fresh series, same design):\n")
    print(HEADER)
    for label, r in ho["overall"].items():
        print(row(label, r))
    print(f"\nHeld-out per family, chosen setting W={best[0]}, q={best[1]}:\n")
    print(HEADER.replace("Setting", "Family"))
    for fam, r in ho["per_family"].items():
        print(row(fam, r))


if __name__ == "__main__":
    main(sys.argv)
