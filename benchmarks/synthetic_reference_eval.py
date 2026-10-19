"""SYNTHETIC evaluation of the reference modes (``python benchmarks/synthetic_reference_eval.py``).

Everything here is generated data (seeds derived from 42). Results are SYNTHETIC and say
nothing about how it performs on real data.

Corpus: the README demo series (unchanged) plus 24 series, 6 per family:

* ``random_walk``  cumulative sum of N(0, 1) steps
* ``sine_noise``   5*sin(2*pi*t/P), P ~ U(150, 600) samples, plus N(0, 1) noise
* ``stationary``   N(0, 1) noise
* ``step``         N(0, 1) noise plus a level jump of +-U(8, 15) at a random sample inside a
                   random block in [30, 60]

Each generated series has 80 blocks of 32 samples and 4 spike blocks drawn from blocks
[18, 79] (never the step block or its neighbours), each with one spike of +-U(6, 12) at a
random position. The guard uses warm-up 16 blocks, quantile 0.99, no drift.

Scoring (block level, only blocks >= 18 so every mode is past warm-up):

* spike caught: the spike block latches with ``SCORE_OVER_TAU``;
* false latch: any other block latching with ``SCORE_OVER_TAU``, except the step block itself
  (reported separately as "step caught"); blocks right after a spike or step count as false;
* detections: all ``SCORE_OVER_TAU`` latches.

Two sensitivity runs reuse the same 24 series: smaller spikes U(3, 6), and a 32-block warm-up
(scored from block 34, so spikes before block 34 are ignored).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from streamguard import StreamGuard

N_BLOCKS = 80
WARMUP = 16
FIRST_EVAL = WARMUP + 2
N_SPIKES = 4
FAMILIES = ("random_walk", "sine_noise", "stationary", "step")
PER_FAMILY = 6

CONFIGS = {
    "median (rb=4, old default)": {"reference": "median", "ref_blocks": 4},
    "median (rb=1)": {"reference": "median", "ref_blocks": 1},
    "ewma (a=0.05)": {"reference": "ewma", "ewma_alpha": 0.05},
    "ewma (a=0.2)": {"reference": "ewma", "ewma_alpha": 0.2},
    "detrend": {"reference": "detrend"},
    "diff": {"reference": "diff"},
}


def make_series(family: str, k: int, spike: tuple[float, float] = (6.0, 12.0)) -> dict:
    rng = np.random.default_rng([42, FAMILIES.index(family), k])
    n = N_BLOCKS * 32
    t = np.arange(n)
    step_block = None
    if family == "random_walk":
        x = np.cumsum(rng.normal(0.0, 1.0, n))
    elif family == "sine_noise":
        x = 5.0 * np.sin(2 * np.pi * t / rng.uniform(150, 600)) + rng.normal(0.0, 1.0, n)
    elif family == "stationary":
        x = rng.normal(0.0, 1.0, n)
    else:
        x = rng.normal(0.0, 1.0, n)
        step_block = int(rng.integers(30, 61))
        at = step_block * 32 + int(rng.integers(0, 32))
        x[at:] += rng.choice([-1.0, 1.0]) * rng.uniform(8, 15)
    banned = set() if step_block is None else {step_block - 1, step_block, step_block + 1}
    candidates = [b for b in range(FIRST_EVAL, N_BLOCKS) if b not in banned]
    spikes = sorted(int(b) for b in rng.choice(candidates, size=N_SPIKES, replace=False))
    for b in spikes:
        x[b * 32 + int(rng.integers(0, 32))] += rng.choice([-1.0, 1.0]) * rng.uniform(*spike)
    return {"name": f"{family}_{k}", "family": family, "x": x, "spikes": spikes, "step": step_block}


def demo_series() -> dict:
    path = Path(__file__).resolve().parents[1] / "examples" / "synthetic_demo.py"
    spec = importlib.util.spec_from_file_location("synthetic_demo", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {
        "name": "demo",
        "family": "demo",
        "x": mod.make_series(),
        "spikes": list(mod.SPIKE_BLOCKS),
        "step": None,
    }


def evaluate(series: dict, cfg: dict, warmup: int = WARMUP, first_eval: int = FIRST_EVAL) -> dict:
    g = StreamGuard(scorer="spark", warmup_blocks=warmup, quantile=0.99, **cfg)
    x = series["x"].reshape(N_BLOCKS, 32)
    latched = set()
    for b in range(N_BLOCKS):
        d = g.process(x[b])
        if b >= first_eval and d.reason.value == "SCORE_OVER_TAU":
            latched.add(b)
    spikes = {b for b in series["spikes"] if b >= first_eval}
    step = series["step"] if (series["step"] or -1) >= first_eval else None
    return {
        "spikes": len(spikes),
        "caught": len(spikes & latched),
        "false": len(latched - spikes - {step}),
        "detections": len(latched),
        "negatives": N_BLOCKS - first_eval - len(spikes) - int(step is not None),
        "steps": int(step is not None),
        "step_caught": int(step is not None and step in latched),
    }


def corpus(spike: tuple[float, float] = (6.0, 12.0), demo: bool = True) -> list[dict]:
    gen = [make_series(f, k, spike) for f in FAMILIES for k in range(PER_FAMILY)]
    return ([demo_series()] if demo else []) + gen


def run(data: list[dict] | None = None, warmup: int = WARMUP, first_eval: int = FIRST_EVAL) -> dict:
    data = corpus() if data is None else data
    out: dict = {}
    for label, cfg in CONFIGS.items():
        out[label] = {
            s["name"]: {**evaluate(s, cfg, warmup, first_eval), "family": s["family"]} for s in data
        }
    return out


def summarize(results: dict, families: tuple[str, ...]) -> dict:
    summary = {}
    for label, rows in results.items():
        sel = [r for r in rows.values() if r["family"] in families]
        summary[label] = {
            k: sum(r[k] for r in sel)
            for k in (
                "spikes",
                "caught",
                "false",
                "detections",
                "steps",
                "step_caught",
                "negatives",
            )
        }
    return summary


def table(summary: dict, title: str) -> str:
    lines = [
        f"{title}",
        "",
        "| Reference mode | Spikes caught | False latches | Detections | Steps caught |",
        "|---|---|---|---|---|",
    ]
    for label, s in summary.items():
        steps = f"{s['step_caught']}/{s['steps']}" if s["steps"] else "-"
        lines.append(
            f"| {label} | {s['caught']}/{s['spikes']} | {s['false']}/{s['negatives']} "
            f"| {s['detections']} | {steps} |"
        )
    return "\n".join(lines)


def main() -> None:
    results = run()
    print("SYNTHETIC reference-mode evaluation (seed 42; generated data only)\n")
    print(table(summarize(results, ("demo",)), "Demo series (examples/synthetic_demo.py)"))
    print()
    print(table(summarize(results, FAMILIES), f"24 generated series ({PER_FAMILY} per family)"))
    for fam in FAMILIES:
        print()
        print(table(summarize(results, (fam,)), f"Family: {fam}"))
    print()
    small = run(corpus(spike=(3.0, 6.0), demo=False))
    print(table(summarize(small, FAMILIES), "Sensitivity: smaller spikes U(3, 6), 24 series"))
    print()
    wu32 = run(corpus(demo=False), warmup=32, first_eval=34)
    print(table(summarize(wu32, FAMILIES), "Sensitivity: warm-up 32 blocks (scored from block 34)"))
    if "--json" in sys.argv:
        Path("benchmarks/synthetic_reference_eval.json").write_text(
            json.dumps(results, indent=1, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
