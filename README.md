# Spark StreamGuard

A small, streaming, **fail-closed** guardrail for numeric data. It reads a stream in blocks of
32 values, scores each block against a reference, and either **admits** it or **latches**
(blocks) it with a reason code. It can tune its own threshold from a warm-up window, keeps
constant memory per stream and writes every decision to a JSONL audit log.

Status: research prototype, milestone 1 (`0.1.0.dev0`). Runtime dependency: numpy only.
Python 3.10 to 3.13.

> **No benchmarks run yet; all performance figures are TARGET or UNRUN.**

The scorer, latch rule and self-calibrated threshold are ported from
[oes512q-latch](https://github.com/sparkainlp-x/oes512q-latch) (v0.2.0) and cleaned up; a
parity test checks the calibrated tau and scale against values produced by that repository.

## Quick start

```bash
pip install -e ".[dev]"      # from a clone of this repository
python -m pytest --cov        # run the tests
```

Library:

```python
import numpy as np
from streamguard import StreamGuard, AuditLogWriter

rng = np.random.default_rng(42)
with AuditLogWriter("audit.jsonl") as log:
    guard = StreamGuard(warmup_blocks=32, audit=log)   # self-calibrated tau (default)
    for block in rng.normal(size=(100, 32)):
        d = guard.process(block)        # Decision(index, admit, score, tau, reason, ...)
        if not d.admit:
            print(d.index, d.reason.value, d.score)

fixed = StreamGuard(tau=0.50)           # original fixed threshold, still available
```

Command line (reads a CSV file or stdin, writes JSONL to stdout, a summary to stderr):

```bash
python -m streamguard data.csv --header --column value --warmup 32 --audit-log audit.jsonl
cat numbers.txt | python -m streamguard --tau 0.5
```

Example run (fixed `tau = 0.5`, 4 blocks: one seed block, one quiet block, one block with a
single unit spike, one block ending in `nan`). Output actually observed:

```text
$ python -m streamguard --tau 0.5 --ref-blocks 1 < numbers.txt
streamguard: 4 blocks, 1 admitted, 3 latched, tau=0.5
{"type": "decision", "v": 1, "i": 0, "admit": false, "latch": 1, "score": null, "tau": 0.5, "reason": "NO_REFERENCE", "peak": null, "event": null, "tau_next": null}
{"type": "decision", "v": 1, "i": 1, "admit": true, "latch": 0, "score": 0.0, "tau": 0.5, "reason": "OK", "peak": 0.0, "event": null, "tau_next": null}
{"type": "decision", "v": 1, "i": 2, "admit": false, "latch": 1, "score": 0.5129406249202847, "tau": 0.5, "reason": "SCORE_OVER_TAU", "peak": 0.99, "event": null, "tau_next": null}
{"type": "decision", "v": 1, "i": 3, "admit": false, "latch": 1, "score": null, "tau": 0.5, "reason": "NAN_INPUT", "peak": null, "event": null, "tau_next": null}
```

The first line of stdout is a `config` record; each following line is one `decision` record
with the keys `type, v, i, admit, latch, score, tau, reason, peak, event, tau_next`.
Non-finite numbers are written as `null`, so every line is strict JSON.

## How it works

For each block `x` (exactly 32 finite values) and a reference `x_ref`:

* residual `r_j = (x_j - x_ref_j) / scale`, `j = 0..31`; the block residual is
  `R = max_j |x_j - x_ref_j|` (reported, scaled, as `peak`);
* `Peak = max_j |r_j|`, `RMS = sqrt(mean_j r_j^2)`, `MeanAbs = mean_j |r_j|`, all three over the
  same 32 residuals of that one block (same definitions as `oes32_hilbert_hook.block_score` in
  oes512q-latch);
* score `S = 0.45*Peak + 0.35*RMS + 0.20*MeanAbs` (so `MeanAbs <= S <= Peak`);
* **latch iff `S >= tau`** (equality latches).

Reference (`reference=`): by default the median of the last `ref_blocks * 32` valid samples
(a causal trailing median, as in the oes512q-latch benchmark), kept in a fixed ring buffer.
A fixed scalar or 32-vector reference can be given instead.

Threshold:

* **Self-calibrated (default).** After `warmup_blocks` valid blocks: `scale = 1.4826 * MAD`
  of all warm-up residuals (fallback `sqrt(pi/2) * mean absolute deviation`, and fail-closed if
  both are 0), then `tau` = the 0.99 quantile (type 7 / numpy `linear`) of the warm-up block
  scores. Deterministic, label-free.
* **Fixed.** `StreamGuard(tau=0.50)` uses `scale = 1.0`. The 0.50 default is known to behave
  poorly on real data (in the oes512q-latch NAB benchmark it behaved like flagging every block,
  REPORTED there), which is why the
  calibrated mode is the default.
* **Optional bounded drift** (`DriftConfig(rate, max_rel)`, off by default): on every scored
  block `tau += rate * tau0 * (hit - (1 - q))`, clamped to `tau0 * [1 - max_rel, 1 + max_rel]`.
  Every change is logged with `event = "TAU_DRIFT"` and `tau_next`.

Fail-closed reason codes: `OK` (the only admitting code), `SCORE_OVER_TAU`, `NAN_INPUT`,
`INF_INPUT`, `BAD_SHAPE` (wrong length, empty, wrong dimensions), `BAD_TYPE` (non-numeric,
boolean, unparseable CLI token), `NONFINITE_SCORE` (overflow), `NO_REFERENCE` (first block that
seeds the trailing reference), `NOT_CALIBRATED` (warm-up), `CALIBRATION_FAILED`, `BAD_TAU`.
Invalid blocks never update the reference, the warm-up buffer or tau. In the CLI a trailing
partial block is emitted and latches with `BAD_SHAPE`.

Memory: each `StreamGuard` holds only fixed-size numpy buffers allocated at construction
(the reference ring buffer and a `(warmup_blocks, 32)` warm-up buffer that is released after
calibration). A test runs 20,000 blocks under `tracemalloc` and checks there is no growth.

## Mapping of the original ideas

From the Spark StreamGuard product spec v0.1 (draft). Row labels are paraphrased where the
original label fails this project's claim filter.

| Original idea | Real feature | Status in this repo |
|---|---|---|
| Membrane | Admit/latch gate on every block. Invalid input (NaN, wrong shape) = latch. Fail-closed. | M1: implemented |
| Loop | Closed-loop threshold: tau self-calibrates from a warm-up window and drifts slowly with a bounded update rule. Every change is logged. | M1: implemented |
| Redundancy frame | 512-value frame = 16 blocks x 32 with parity blocks, so up to k missing blocks can be rebuilt. Measured by erasure tests. | M2: not started |
| Unbounded streaming | O(1) memory per stream, runs indefinitely on unbounded input. | M1: implemented (constant-memory test) |
| Alternative compute | None claimed; runs fully classical (no quantum claims). The optional feature-encoding hook from oes512q-latch is not ported; if added it stays experimental and off by default. | not ported |

## SYNTHETIC demo

`examples/synthetic_demo.py`: a seeded (42) Gaussian random walk of 80 blocks x 32 samples,
with one +8.0 spike injected into blocks 30, 47 and 63 and a single `NaN` in block 70,
processed with the default settings (`warmup_blocks=16`, trailing-median reference over 4
blocks, calibrated tau). Output actually observed (**SYNTHETIC**):

```text
$ python examples/synthetic_demo.py
SYNTHETIC demo | seed=42 blocks=80 warmup=16 spikes=(30, 47, 63)
block 16: CALIBRATED tau=2.0432 scale=0.5065
block 19: LATCH reason=SCORE_OVER_TAU score=2.0550
block 20: LATCH reason=SCORE_OVER_TAU score=2.3138
block 22: LATCH reason=SCORE_OVER_TAU score=2.8091
block 23: LATCH reason=SCORE_OVER_TAU score=2.6049
block 28: LATCH reason=SCORE_OVER_TAU score=2.6120
block 30: LATCH reason=SCORE_OVER_TAU score=7.8221
block 38: LATCH reason=SCORE_OVER_TAU score=2.3095
block 45: LATCH reason=SCORE_OVER_TAU score=2.1181
block 47: LATCH reason=SCORE_OVER_TAU score=7.7300
block 48: LATCH reason=SCORE_OVER_TAU score=2.7286
block 49: LATCH reason=SCORE_OVER_TAU score=4.6185
block 50: LATCH reason=SCORE_OVER_TAU score=3.3358
block 53: LATCH reason=SCORE_OVER_TAU score=2.4201
block 54: LATCH reason=SCORE_OVER_TAU score=3.1274
block 55: LATCH reason=SCORE_OVER_TAU score=2.1935
block 56: LATCH reason=SCORE_OVER_TAU score=2.3623
block 59: LATCH reason=SCORE_OVER_TAU score=2.6912
block 60: LATCH reason=SCORE_OVER_TAU score=4.9277
block 61: LATCH reason=SCORE_OVER_TAU score=6.0431
block 62: LATCH reason=SCORE_OVER_TAU score=3.9273
block 63: LATCH reason=SCORE_OVER_TAU score=7.7827
block 70: LATCH reason=NAN_INPUT score=-
block 73: LATCH reason=SCORE_OVER_TAU score=2.3956
block 74: LATCH reason=SCORE_OVER_TAU score=2.2508
block 77: LATCH reason=SCORE_OVER_TAU score=2.5825
block 79: LATCH reason=SCORE_OVER_TAU score=2.3170
reason counts: {'NAN_INPUT': 1, 'NOT_CALIBRATED': 16, 'NO_REFERENCE': 1, 'OK': 37, 'SCORE_OVER_TAU': 25}
```

Reading it honestly (**SYNTHETIC**, not a performance claim): the three injected spikes and the
NaN dropout all latch, but so do 22 other blocks with no injected spike. A random walk is not
stationary, and a block-constant trailing-median reference lags it, so ordinary drift inside a
block scores above the tau learned on a calmer warm-up window. Choosing a better reference and
measuring false-latch rates on real data is work for the benchmark milestone.

## Evidence tags

Every number in this project is tagged **SYNTHETIC** (generated data, e.g. the demo and tests),
**REPORTED** (measured on public data), **TARGET** (a goal) or **UNRUN** (planned, not run).
Spec targets such as throughput of 100k blocks/s per core, memory under 10 MB per stream, and a
false-latch rate of 1% or less on clean warm-up data are **TARGET** and have not been measured.

## Scope and limits

Research prototype. Not a certified safety system, not a diagnostic device and not a product.
No quantum claims. Parity rebuild (M2), benchmarks (M4) and the LLM adapter (M5) are not in
this milestone.

## Roadmap

* M1: scorer, latch, calibrator, stream guard, CLI, tests, CI (this release).
* M2: parity rebuild and erasure tests.
* M3: streaming CLI extensions (MQTT input).
* M4: benchmark suite vs CUSUM, EWMA, Isolation Forest, rolling z-score and oes512q-latch on
  NAB / Yahoo S5 / SMD.
* M5: LLM output adapter, v0.1.0 release and Zenodo DOI.

## Citation and license

See [CITATION.cff](CITATION.cff). MIT License, copyright Spark AI NLP.
