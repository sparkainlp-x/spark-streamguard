# Spark StreamGuard

**A fail-closed, auditable gate for any streaming detector.** StreamGuard reads a numeric
stream in blocks of 32 values and wraps a pluggable anomaly scorer with the parts a scorer does
not give you: input validation, reason codes, a self-calibrated threshold, optional sequence-gap
detection, constant memory per stream and a JSONL audit log of every decision. Each block is
either **admitted** or **latched** (blocked) with a reason code; anything it cannot score
(NaN, inf, wrong length, non-numeric, overflow, not yet calibrated, missing block) latches.

Status: research prototype (`0.1.0`). Runtime dependency: numpy only (the optional
`isoforest` scorer needs scikit-learn). Python 3.10 to 3.13.

> **The M4 real-data benchmark was lost, and this is why the project changed focus.** The
> original StreamGuard score (`spark`) is significantly worse than a plain EWMA control chart on
> NAB and ties at a poor level on SMD; see
> [Real-data benchmark (REPORTED)](#real-data-benchmark-reported). The value that remains is
> the gate, not the score. The scorer is now pluggable; the default is an EWMA chart chosen by
> a rule fixed in advance on the M4 tuning splits
> ([Choosing a scorer](#choosing-a-scorer-reported-tuning-splits-only)). Wrapping a scorer in the
> gate changes none of its test-set decisions on clean data
> ([Gate vs raw detectors](#gate-vs-raw-detectors-on-the-m4-test-split-reported)). What the gate
> adds is measured under fault injection
> ([Robustness under faulty input](#robustness-under-faulty-input-reported)). Other numbers in
> this README are SYNTHETIC (generated data) or TARGET/UNRUN.

The `spark` score, latch rule and self-calibrated threshold are ported from
[oes512q-latch](https://github.com/sparkainlp-x/oes512q-latch) (v0.2.0) and cleaned up; a
parity test checks the calibrated tau and scale against values produced by that repository.

## Quick start

```bash
git clone https://github.com/sparkainlp-x/spark-streamguard
cd spark-streamguard
pip install .                  # core: numpy only
pip install ".[isoforest]"     # optional IsolationForest scorer (scikit-learn)
pip install -e ".[dev]"        # development: pytest, coverage, ruff
python -m pytest --cov         # tests; scikit-learn tests are skipped when it is not installed
```

Library:

```python
import numpy as np
from streamguard import StreamGuard, AuditLogWriter

rng = np.random.default_rng(42)
with AuditLogWriter("audit.jsonl") as log:
    guard = StreamGuard(audit=log)  # defaults: ewma scorer, 64-block warm-up, q=0.995
    log.write_record(guard.config())  # config line: scorer name and parameters
    for seq, block in enumerate(rng.normal(size=(200, 32))):
        d = guard.process(block, seq=seq)  # seq is optional; gaps latch SEQUENCE_GAP
        if not d.admit:
            print(d.index, d.reason.value, d.score)

spark = StreamGuard(scorer="spark", reference="median")  # original score, any reference mode
iso = StreamGuard(scorer="isoforest")  # needs scikit-learn (pip install ".[isoforest]")
fixed = StreamGuard(tau=0.50, scorer="spark")  # original fixed-threshold contract
```

Command line (reads a CSV file or stdin, writes JSONL to stdout, a summary to stderr):

```bash
python -m streamguard data.csv --header --column value --audit-log audit.jsonl
python -m streamguard data.csv --scorer spark --reference diff --quantile 0.999
cat numbers.txt | python -m streamguard --tau 0.5
```

Example run (default `ewma` scorer with a fixed `tau = 0.5`, 4 blocks: one seed block, one
quiet block, one block ending in a single 0.99 spike, one block ending in `nan`). Output actually
observed:

```text
$ python -m streamguard --tau 0.5 < numbers.txt
streamguard: 4 blocks, 1 admitted, 3 latched, tau=0.5
{"type": "config", "v": 2, "mode": "fixed", "tau": 0.5, "warmup_blocks": null, "quantile": 0.995, "scale": null, "scorer": "ewma", "scorer_params": {"lam": 0.2, "ref_window": 288}, "drift_rate": 0.0, "drift_max_rel": 0.25, "block": 32}
{"type": "decision", "v": 2, "i": 0, "admit": false, "latch": 1, "score": null, "tau": 0.5, "reason": "NO_REFERENCE", "peak": null, "event": null, "tau_next": null}
{"type": "decision", "v": 2, "i": 1, "admit": true, "latch": 0, "score": 0.0, "tau": 0.5, "reason": "OK", "peak": 0.0, "event": null, "tau_next": null}
{"type": "decision", "v": 2, "i": 2, "admit": false, "latch": 1, "score": 0.594, "tau": 0.5, "reason": "SCORE_OVER_TAU", "peak": 0.99, "event": null, "tau_next": null}
{"type": "decision", "v": 2, "i": 3, "admit": false, "latch": 1, "score": null, "tau": 0.5, "reason": "NAN_INPUT", "peak": null, "event": null, "tau_next": null}
```

The first line of stdout is a `config` record (schema `v: 2`: it names the `scorer` and its
`scorer_params`); each following line is one `decision` record with the keys
`type, v, i, admit, latch, score, tau, reason, peak, event, tau_next`. Non-finite numbers are
written as `null`, so every line is strict JSON. The CLI chunks a flat value stream, so it cannot
see a missing sample or block; use `process(block, seq=...)` from a framed producer for that.

## How it works

### The gate (the same for every scorer)

For every block, in this order:

1. **Sequence check** (only if the caller passes `seq`): a block whose `seq` is not the previous
   `seq + 1` (dropped, repeated or reordered) latches with `SEQUENCE_GAP`, is not scored, and
   resets the scorer like an invalid block.
2. **Validation**: exactly 32 finite real numbers, else `NAN_INPUT`, `INF_INPUT`, `BAD_SHAPE`
   (wrong length, empty, wrong dimensions) or `BAD_TYPE` (non-numeric, boolean, unparseable CLI
   token). Invalid blocks never reach the scorer's state, the warm-up buffer or tau.
3. **Warm-up** (self-calibrated mode, the default): the scorer consumes `warmup_blocks` valid
   blocks (latched, `NOT_CALIBRATED`; a scorer that needs history latches its first block with
   `NO_REFERENCE`). Then the scorer fixes its scale/model and `tau` = the `quantile` (type 7 /
   numpy `linear`) of its own warm-up block scores (`event = "CALIBRATED"`). If that fails (for
   example constant data), the block latches with `CALIBRATION_FAILED` and warm-up restarts.
   Nothing is admitted before a tau exists. Defaults: `warmup_blocks = 64` (SYNTHETIC sweep),
   `quantile = 0.995` (chosen with the scorer, see below); with them nothing is admitted until
   65 valid blocks (2,080 samples) have been seen.
4. **Scoring**: admit iff `S < tau` (equality latches, `SCORE_OVER_TAU`). A score that is not a
   finite number latches with `NONFINITE_SCORE`; a scorer can also report `NO_REFERENCE`.
5. **Optional bounded drift** (`DriftConfig(rate, max_rel)`, off by default): on every scored
   block `tau += rate * tau0 * (hit - (1 - q))`, clamped to `tau0 * [1 - max_rel, 1 + max_rel]`.
   Every change is logged with `event = "TAU_DRIFT"` and `tau_next`.
6. **Audit**: every decision can be written to an append-only JSONL log (`AuditLogWriter`).

Fixed mode (`StreamGuard(tau=0.50)`) skips the warm-up and uses `scale = 1.0`. The 0.50 default
of the original contract is known to behave poorly on real data (in the oes512q-latch NAB
benchmark it behaved like flagging every block, REPORTED there).

Reason codes: `OK` (the only admitting code), `SCORE_OVER_TAU`, `NAN_INPUT`, `INF_INPUT`,
`BAD_SHAPE`, `BAD_TYPE`, `NONFINITE_SCORE`, `NO_REFERENCE`, `NOT_CALIBRATED`,
`CALIBRATION_FAILED`, `BAD_TAU`, `SEQUENCE_GAP`. In the CLI a trailing partial block is emitted
and latches with `BAD_SHAPE`.

### Scorers

`scorer=` / `--scorer`. Each keeps O(1) or bounded state per stream:

| Scorer | Block score | State per stream |
|---|---|---|
| `ewma` (**default**) | EWMA control chart on the trailing-median residual `r = (x - median of the last ref_window samples) / scale`: `z_t = lam*r_t + (1 - lam)*z_(t-1)`, block score `max_t abs(z_t) / sqrt(lam/(2 - lam))`. `lam = 0.2`, `ref_window = 288` samples. numpy only | `ref_window` ring buffer, 1 float, a 32x32 matrix |
| `spark` | the original score (below) against a reference mode | see reference modes |
| `isoforest` | scikit-learn IsolationForest (100 trees, seed 42) fitted on the scaled warm-up residual blocks only; score `-score_samples`. Optional (`pip install ".[isoforest]"`) | fixed-size model: 100 trees on at most `min(256, warmup_blocks)` samples each |
| your own | any object with `name`, `start`, `warmup`, `fit`, `fit_fixed`, `score`, `on_invalid`, `params` (see `streamguard.scorers.Scorer`) | yours |

`scale` is the robust warm-up scale: `1.4826 * MAD` of all warm-up residuals (fallback
`sqrt(pi/2) * mean absolute deviation`; fail-closed if both are 0). Passing `reference=`,
`ref_blocks=` or `ewma_alpha=` without `scorer=` selects `spark`, so pre-pivot code keeps its
behaviour except for the default quantile.

**`spark` score.** For each block `x` and a reference `x_ref`: residual
`r_j = (x_j - x_ref_j) / scale`; `Peak = max_j |r_j|`, `RMS = sqrt(mean_j r_j^2)`,
`MeanAbs = mean_j |r_j|` over the 32 residuals of that block (same definitions as
`oes32_hilbert_hook.block_score` in oes512q-latch); `S = 0.45*Peak + 0.35*RMS + 0.20*MeanAbs`.
Reference modes (`reference=` / `--reference`):

| Mode | `x_ref` for a block | State |
|---|---|---|
| `diff` (spark default) | first difference: `[last sample of the previous block, x_0, ..., x_30]`. An invalid block resets it, so the next valid block latches with `NO_REFERENCE` | 1 float |
| `detrend` | least-squares line fitted to the block itself | none |
| `ewma` | exponentially weighted level of earlier samples (`--ewma-alpha`, default 0.05) | 1 float |
| `median` | median of the last `ref_blocks * 32` valid samples (`--ref-blocks`, default 4); alias `"trailing_median"` | ring buffer |
| scalar / 32-vector | given by the caller | the value |

The `ewma` and `isoforest` scorers keep their trailing history across an invalid block (they do
not reset); the `spark`/`diff` reference does.

Memory: each `StreamGuard` holds only fixed-size state allocated at construction (the scorer
state above and a `(warmup_blocks, 32)` warm-up buffer that is released after calibration).
Tests run thousands of blocks per scorer and reference mode under `tracemalloc` and check there
is no growth.

## Mapping of the original ideas

From the Spark StreamGuard product spec v0.1 (draft). Row labels are paraphrased where the
original label fails this project's claim filter.

| Original idea | Real feature | Status in this repo |
|---|---|---|
| Membrane | Admit/latch gate on every block. Invalid input (NaN, wrong shape) = latch. Fail-closed. | M1: implemented; post-M4: wraps any scorer, optional `SEQUENCE_GAP` check |
| Loop | Closed-loop threshold: tau self-calibrates from a warm-up window and drifts slowly with a bounded update rule. Every change is logged. | M1: implemented |
| Redundancy frame | 512-value frame = 16 blocks x 32 with parity blocks, so up to k missing blocks can be rebuilt. Measured by erasure tests. | M2: not started |
| Unbounded streaming | O(1) memory per stream, runs indefinitely on unbounded input. | M1: implemented (constant-memory test) |
| Alternative compute | None claimed; runs fully classical (no quantum claims). The optional feature-encoding hook from oes512q-latch is not ported; if added it stays experimental and off by default. | not ported |

## SYNTHETIC demo

`examples/synthetic_demo.py`: a seeded (42) Gaussian random walk of 80 blocks x 32 samples
(step sd 0.1), with one +8.0 spike injected into blocks 30, 47 and 63 and a single `NaN` in
block 70. The series, seed and spikes are unchanged from the first M1 commit. It pins the
`spark` scorer with the `diff` reference and q=0.999 (the 0.1.0.dev0 defaults this demo was
written for) and keeps `warmup_blocks=16`: the series is only 80 blocks long, and the 64-block
default warm-up would cover two of the spikes. Output actually observed (**SYNTHETIC**):

```text
$ python examples/synthetic_demo.py
SYNTHETIC demo | seed=42 blocks=80 warmup=16 spikes=(30, 47, 63)
block 16: CALIBRATED tau=2.0482 scale=0.0961
block 29: LATCH reason=SCORE_OVER_TAU score=2.3179
block 30: LATCH reason=SCORE_OVER_TAU score=45.6126
block 47: LATCH reason=SCORE_OVER_TAU score=46.6259
block 63: LATCH reason=SCORE_OVER_TAU score=45.6708
block 70: LATCH reason=NAN_INPUT score=-
reason counts: {'NAN_INPUT': 1, 'NOT_CALIBRATED': 16, 'NO_REFERENCE': 2, 'OK': 57, 'SCORE_OVER_TAU': 4}
```

Reading it (**SYNTHETIC**, not a performance claim): the three spikes and the NaN dropout latch,
and one block with no injected spike (29) also latches. Block 71, after the NaN block, now
latches with `NO_REFERENCE` because the `diff` reference is reset by the invalid block; before
that fix it scored a difference across the gap and latched as `SCORE_OVER_TAU`. History on this
demo: 22 false latches with the original 4-block `median` reference, 3 with `diff` at q=0.99, 2
after the reset fix, 1 at q=0.999.

**The new default `ewma` scorer does badly on this demo** (SYNTHETIC, `warmup_blocks=16`,
q=0.995): it latches 37 of the 62 post-warm-up blocks and misses the spike in block 30. Its
trailing-median residual treats random-walk drift as a level shift, and the EWMA smooths a
single-sample spike. The default was chosen on real data (next sections), not on this demo; pick
the scorer for your data.

## Choosing the reference (SYNTHETIC)

This section and the next two predate the pivot and apply to the `spark` scorer.

`benchmarks/synthetic_reference_eval.py` compares the modes on the demo series above and on 24
more generated series (seeds derived from 42): 6 each of random walk, sine plus noise,
stationary noise, and noise with one step change. Each series has 80 blocks and 4 spike blocks
(one spike of +-U(6, 12) noise sd each), warm-up 16 blocks, quantile 0.99 (the settings at the
time; the warm-up and quantile are tuned separately below), no drift. Blocks from
18 on are scored. A false latch is any `SCORE_OVER_TAU` on a block without an injected spike,
except the step block, which is reported separately. Parameters were fixed before the run and
not tuned per mode. All numbers below are **SYNTHETIC** (generated data, measured on this code)
and say nothing about how it performs on real data.

Demo series:

| Reference mode | Spikes caught | False latches | Detections | Steps caught |
|---|---|---|---|---|
| median (rb=4, old default) | 3/3 | 22/59 | 25 | - |
| median (rb=1) | 3/3 | 16/59 | 19 | - |
| ewma (a=0.05) | 3/3 | 13/59 | 16 | - |
| ewma (a=0.2) | 3/3 | 12/59 | 15 | - |
| detrend | 3/3 | 6/59 | 9 | - |
| diff | 3/3 | 2/59 | 5 | - |

24 generated series:

| Reference mode | Spikes caught | False latches | Detections | Steps caught |
|---|---|---|---|---|
| median (rb=4, old default) | 58/96 | 135/1386 | 199 | 6/6 |
| median (rb=1) | 72/96 | 125/1386 | 203 | 6/6 |
| ewma (a=0.05) | 75/96 | 123/1386 | 204 | 6/6 |
| ewma (a=0.2) | 74/96 | 92/1386 | 172 | 6/6 |
| detrend | 95/96 | 107/1386 | 208 | 6/6 |
| diff | 96/96 | 84/1386 | 186 | 6/6 |

<details><summary>Per family and sensitivity runs (SYNTHETIC)</summary>

Random walk:

| Reference mode | Spikes caught | False latches | Detections | Steps caught |
|---|---|---|---|---|
| median (rb=4, old default) | 1/24 | 29/348 | 30 | - |
| median (rb=1) | 2/24 | 19/348 | 21 | - |
| ewma (a=0.05) | 5/24 | 16/348 | 21 | - |
| ewma (a=0.2) | 3/24 | 12/348 | 15 | - |
| detrend | 23/24 | 20/348 | 43 | - |
| diff | 24/24 | 16/348 | 40 | - |

Sine plus noise:

| Reference mode | Spikes caught | False latches | Detections | Steps caught |
|---|---|---|---|---|
| median (rb=4, old default) | 9/24 | 36/348 | 45 | - |
| median (rb=1) | 22/24 | 42/348 | 64 | - |
| ewma (a=0.05) | 22/24 | 38/348 | 60 | - |
| ewma (a=0.2) | 23/24 | 20/348 | 43 | - |
| detrend | 24/24 | 30/348 | 54 | - |
| diff | 24/24 | 11/348 | 35 | - |

Stationary noise:

| Reference mode | Spikes caught | False latches | Detections | Steps caught |
|---|---|---|---|---|
| median (rb=4, old default) | 24/24 | 23/348 | 47 | - |
| median (rb=1) | 24/24 | 34/348 | 58 | - |
| ewma (a=0.05) | 24/24 | 33/348 | 57 | - |
| ewma (a=0.2) | 24/24 | 30/348 | 54 | - |
| detrend | 24/24 | 35/348 | 59 | - |
| diff | 24/24 | 32/348 | 56 | - |

Step change:

| Reference mode | Spikes caught | False latches | Detections | Steps caught |
|---|---|---|---|---|
| median (rb=4, old default) | 24/24 | 47/342 | 77 | 6/6 |
| median (rb=1) | 24/24 | 30/342 | 60 | 6/6 |
| ewma (a=0.05) | 24/24 | 36/342 | 66 | 6/6 |
| ewma (a=0.2) | 24/24 | 30/342 | 60 | 6/6 |
| detrend | 24/24 | 22/342 | 52 | 6/6 |
| diff | 24/24 | 25/342 | 55 | 6/6 |

Smaller spikes U(3, 6), same 24 series:

| Reference mode | Spikes caught | False latches | Detections | Steps caught |
|---|---|---|---|---|
| median (rb=4, old default) | 47/96 | 135/1386 | 188 | 6/6 |
| median (rb=1) | 52/96 | 125/1386 | 183 | 6/6 |
| ewma (a=0.05) | 53/96 | 124/1386 | 183 | 6/6 |
| ewma (a=0.2) | 53/96 | 88/1386 | 147 | 6/6 |
| detrend | 68/96 | 107/1386 | 181 | 6/6 |
| diff | 75/96 | 82/1386 | 163 | 6/6 |

Warm-up 32 blocks (scored from block 34, so only spikes from block 34 on count):

| Reference mode | Spikes caught | False latches | Detections | Steps caught |
|---|---|---|---|---|
| median (rb=4, old default) | 24/67 | 34/1032 | 63 | 5/5 |
| median (rb=1) | 31/67 | 21/1032 | 57 | 5/5 |
| ewma (a=0.05) | 30/67 | 20/1032 | 55 | 5/5 |
| ewma (a=0.2) | 28/67 | 11/1032 | 44 | 5/5 |
| detrend | 45/67 | 4/1032 | 54 | 5/5 |
| diff | 43/67 | 4/1032 | 52 | 5/5 |

</details>

Why `diff` became the default: on the 24 series it caught all spikes (96/96, against 58/96 for
the old `median` default) with the fewest false latches (84/1386, against 135/1386). The gain
comes from the random-walk and sine families, where a block-constant reference cannot follow the
signal. `detrend` came close (95/96, 107/1386) and was slightly better on the step family.

The 32-block row in the sensitivity table is not a fair comparison, because fewer spikes fell
after the longer warm-up (67 instead of 96). The next section fixes that.

## Choosing warm-up and quantile (SYNTHETIC)

`benchmarks/warmup_sweep.py` sweeps warm-up length W in {16, 24, 32, 48, 64} blocks and quantile
q in {0.99, 0.995, 0.999} for the `diff` reference. For a fair comparison every setting is
scored on the **same** blocks with the **same** injected events. Each series has 166 blocks.
Block 0 seeds the reference, blocks 1..W are the (clean) warm-up, and all settings are scored on
blocks 66..165, after the longest warm-up. Each series has 6 spike blocks (+-U(6, 12) noise sd),
all in that region. Step series also get one level jump of +-U(8, 15) there. Families as above,
10 series each: 40 tuning series (seeds `[42, 0, family, k]`) and 40 **held-out** series with
fresh seeds (`[42, 1, family, k]`).

Selection rule, written into the script before any held-out result was computed:

1. among settings with false-latch rate <= 1% on the tuning set, take the highest recall;
2. if none reaches 1%, take the lowest false-latch rate among settings with recall >= 90%, and
   say the target is not met;
3. otherwise take the highest recall. Ties: lower false rate, then shorter warm-up.

Tuning set (40 series, 240 spikes, 3,750 other blocks):

| Setting | Spikes caught (recall) | False latches (rate) | Steps caught |
|---|---|---|---|
| diff, W=16, q=0.99 | 238/240 (99.2%) | 245/3750 (6.53%) | 10/10 |
| diff, W=16, q=0.995 | 237/240 (98.8%) | 231/3750 (6.16%) | 10/10 |
| diff, W=16, q=0.999 | 237/240 (98.8%) | 227/3750 (6.05%) | 10/10 |
| diff, W=24, q=0.99 | 238/240 (99.2%) | 162/3750 (4.32%) | 10/10 |
| diff, W=24, q=0.995 | 237/240 (98.8%) | 147/3750 (3.92%) | 10/10 |
| diff, W=24, q=0.999 | 237/240 (98.8%) | 142/3750 (3.79%) | 10/10 |
| diff, W=32, q=0.99 | 235/240 (97.9%) | 117/3750 (3.12%) | 10/10 |
| diff, W=32, q=0.995 | 235/240 (97.9%) | 110/3750 (2.93%) | 10/10 |
| diff, W=32, q=0.999 | 235/240 (97.9%) | 102/3750 (2.72%) | 10/10 |
| diff, W=48, q=0.99 | 235/240 (97.9%) | 95/3750 (2.53%) | 10/10 |
| diff, W=48, q=0.995 | 235/240 (97.9%) | 81/3750 (2.16%) | 10/10 |
| diff, W=48, q=0.999 | 235/240 (97.9%) | 69/3750 (1.84%) | 10/10 |
| diff, W=64, q=0.99 | 234/240 (97.5%) | 82/3750 (2.19%) | 10/10 |
| diff, W=64, q=0.995 | 234/240 (97.5%) | 68/3750 (1.81%) | 10/10 |
| diff, W=64, q=0.999 | 234/240 (97.5%) | 57/3750 (1.52%) | 10/10 |

**No setting reaches the spec's 1% false-latch TARGET.** Rule 2 therefore applies and picks
**W = 64, q = 0.999**, now the `StreamGuard` and CLI default. The false-latch rate fell steadily
as the warm-up grew, and the best setting is at the edge of the grid (64). Longer warm-ups were not
tested; that would need a new, pre-stated sweep.

Held-out set (40 fresh series, same design), evaluated once after the choice:

| Setting | Spikes caught (recall) | False latches (rate) | Steps caught |
|---|---|---|---|
| chosen: diff, W=64, q=0.999 | 237/240 (98.8%) | 65/3750 (1.73%) | 10/10 |
| diff, W=16, q=0.99 (previous eval setting) | 238/240 (99.2%) | 203/3750 (5.41%) | 10/10 |
| median rb=4, W=16, q=0.99 (original default ref) | 156/240 (65.0%) | 338/3750 (9.01%) | 10/10 |

Held-out per family, chosen setting:

| Family | Spikes caught (recall) | False latches (rate) | Steps caught |
|---|---|---|---|
| random_walk | 60/60 (100.0%) | 12/940 (1.28%) | - |
| sine_noise | 58/60 (96.7%) | 14/940 (1.49%) | - |
| stationary | 60/60 (100.0%) | 21/940 (2.23%) | - |
| step | 59/60 (98.3%) | 18/930 (1.94%) | 10/10 |

On held-out data the chosen setting keeps recall at 98.8% and cuts false latches from 5.41%
(W=16, q=0.99) to 1.73%. That is still above the 1% TARGET. All of these numbers are
**SYNTHETIC**.

## Remaining weaknesses (SYNTHETIC evidence)

* **The 1% false-latch TARGET is not met**: 1.73% on the held-out series with the chosen
  defaults (1.52% on the tuning set). The worst family is stationary noise, at 2.23%.
* **The long warm-up has a cost**: nothing is admitted (fail-closed) for the first 65 valid
  blocks. Warm-up data are assumed clean; the sweep never contaminated the warm-up, so the
  effect of anomalies during warm-up is not measured.
* `diff` only sees **changes between neighbouring samples**. A slow ramp or a gradual drift
  anomaly is invisible, and a level shift is flagged only in the block where it happens.
* On **stationary noise** `diff` is slightly worse than `median` (differencing doubles the noise
  variance).
* A spike in the last sample of a block also enters the first difference of the next block, which
  can latch that block too.
* After any invalid block, the next valid block latches with `NO_REFERENCE` (fail-closed but not
  scored). A gap of one bad block therefore costs two latched blocks.
* Smaller spikes are harder: with U(3, 6) spikes (16-block warm-up, q=0.99) `diff` caught 75/96.
  The sweep used U(6, 12) spikes only.
* `detrend` cannot see a whole-block offset or a purely linear block.

`median` stays available (`reference="median"`, `--reference median`) for data where a level
reference fits better, for example stationary data with slow anomalies.

## Choosing a scorer (REPORTED, tuning splits only)

After M4, the default scorer was re-chosen by a rule **written down and committed before it was
run** (`benchmarks/real/pivot.py`, phase `choose`, 2026-09-29), using only the M4 tuning splits
(NAB 16 series, SMD 14 machines; same data, split, warm-up and metrics as M4):

* candidates: the numpy-only built-in scorers with shipped parameters, `spark` with `diff`,
  `median`, `detrend`, and `ewma` with lambda in {0.1, 0.2, 0.3} (`ref_window = 288`), each with
  q in {0.99, 0.995, 0.999}, `warmup_blocks = 64`;
* criterion: the mean of NAB and SMD tuning-split pooled F1 (each dataset weighted 1/2); ties go
  to the earlier candidate; `isoforest` is excluded because it is an optional dependency;
* every number goes through `StreamGuard.process`, block by block.

| Candidate | q | NAB tune F1 | SMD tune F1 | Mean |
|---|---|---|---|---|
| spark (reference=diff) | 0.99 | 0.158 | 0.185 | 0.1718 |
| spark (reference=diff) | 0.995 | 0.141 | 0.193 | 0.1672 |
| spark (reference=diff) | 0.999 | 0.141 | 0.198 | 0.1695 |
| spark (reference=median) | 0.99 | 0.202 | 0.180 | 0.1912 |
| spark (reference=median) | 0.995 | 0.200 | 0.184 | 0.1918 |
| spark (reference=median) | 0.999 | 0.201 | 0.186 | 0.1937 |
| spark (reference=detrend) | 0.99 | 0.132 | 0.186 | 0.1591 |
| spark (reference=detrend) | 0.995 | 0.131 | 0.194 | 0.1627 |
| spark (reference=detrend) | 0.999 | 0.127 | 0.198 | 0.1626 |
| ewma (lam=0.1, ref_window=288) | 0.99 | 0.227 | 0.171 | 0.1992 |
| ewma (lam=0.1, ref_window=288) | 0.995 | 0.227 | 0.174 | 0.2006 |
| ewma (lam=0.1, ref_window=288) | 0.999 | 0.226 | 0.177 | 0.2015 |
| ewma (lam=0.2, ref_window=288) | 0.99 | 0.228 | 0.176 | 0.2020 |
| ewma (lam=0.2, ref_window=288) **(chosen)** | 0.995 | 0.227 | 0.181 | 0.2038 |
| ewma (lam=0.2, ref_window=288) | 0.999 | 0.217 | 0.183 | 0.2001 |
| ewma (lam=0.3, ref_window=288) | 0.99 | 0.215 | 0.177 | 0.1957 |
| ewma (lam=0.3, ref_window=288) | 0.995 | 0.214 | 0.179 | 0.1967 |
| ewma (lam=0.3, ref_window=288) | 0.999 | 0.210 | 0.183 | 0.1964 |

Winner: **`ewma`, lambda = 0.2, `ref_window = 288`, q = 0.995**, now the library and CLI default.
The margins are small (0.2038 against 0.2020 for q = 0.99) and nothing here is a significance
test. Caveat: the M4 tuning numbers for the raw EWMA baseline were already public when the rule
was written, so the rule was not blind to the fact that EWMA did well on NAB.

## Gate vs raw detectors on the M4 test split (REPORTED)

The M4 test protocol, run once (`pivot.py test`): StreamGuard wrapped around `ewma` and
`isoforest`, with the settings M4 tuned for the raw EWMA chart and Isolation Forest (per
dataset; the trailing-median window is M4's per-series `L`), next to the same raw baselines
recomputed by the M4 code.

| Dataset | Detector (M4-tuned setting) | Raw F1 | Gate F1 | Raw AP | Gate AP | Flag disagreements |
|---|---|---|---|---|---|---|
| NAB | EWMA chart (lam=0.2) q=0.99 | 0.272 | 0.272 | 0.369 | 0.369 | 0 / 2621 |
| NAB | Isolation Forest q=0.99 | 0.251 | 0.251 | 0.392 | 0.392 | 0 / 2621 |
| NAB | new default: ewma (lam=0.2, ref_window=288) q=0.995 | | 0.246 | | 0.355 | |
| SMD | EWMA chart (lam=0.2) q=0.999 | 0.192 | 0.192 | 0.496 | 0.496 | 0 / 10071 |
| SMD | Isolation Forest q=0.999 | 0.179 | 0.179 | 0.345 | 0.345 | 0 / 10071 |
| SMD | new default: ewma (lam=0.2, ref_window=288) q=0.995 | | 0.190 | | 0.503 | |

On clean data **the gate changes no test decision**: 0 flag disagreements over 2,621 NAB and
10,071 SMD evaluation blocks, identical F1 and AP to the M4 raw rows. So wrapping costs no
accuracy, but it also adds none: StreamGuard+ewma is exactly as good (and as limited) as an EWMA
chart. The new default (fixed parameters, q = 0.995) reaches NAB test F1 0.246 (the M4 default
reached 0.161; M4's tuned raw EWMA 0.272) and SMD 0.190. The default was not tested for
significance against the baselines.

## Robustness under faulty input (REPORTED)

`pivot.py robust`, on the 16 NAB test series with seed 42: for each fault type, a separate run
per series with 3 faults at distinct evaluation blocks.

* `nan` / `inf`: one sample of the block set to NaN / +inf.
* `short`: one sample of the block missing. `drop`: the whole 32-sample block missing.
* **Raw baselines** (the M4 implementations, M4-tuned NAB settings) consume the faulted *sample
  stream*, their natural input, and flag `score >= tau`; a NaN score compares false and is
  admitted. For `short` and `drop` every later sample shifts, so a fault is "flagged" only if the
  block at the splice latches.
* **StreamGuard** (q = 0.99, same scorers) consumes producer-framed blocks: `short` arrives as a
  31-value block, `drop` as a missing block, with and without `seq` numbers.
* "Clean-block decision changes": evaluation blocks without a fault whose admit/latch decision
  differs from the fault-free run (n/a where the raw stream is shifted).

| Fault | Method | Crashed series | Faults flagged | Non-finite scores admitted | Clean-block decision changes | Reason codes at faults |
|---|---|---|---|---|---|---|
| nan | EWMA chart (lam=0.2, q=0.99) | 0 | 0 / 48 | 1498 | 177 | - |
| nan | CUSUM (k=2.0, q=0.99) | 0 | 0 / 48 | 1498 | 587 | - |
| nan | Rolling z-score (window=64, q=0.99) | 0 | 0 / 48 | 0 | 2 | - |
| nan | Isolation Forest (q=0.99) | 0 | 1 / 48 | 0 | 16 | - |
| nan | StreamGuard+ewma (lam=0.2, q=0.99) | 0 | 48 / 48 | 0 | 1 | NAN_INPUT 48 |
| nan | StreamGuard+spark (diff, q=0.99) | 0 | 48 / 48 | 0 | 45 | NAN_INPUT 48 |
| nan | StreamGuard+isoforest (q=0.99) | 0 | 48 / 48 | 0 | 0 | NAN_INPUT 48 |
| inf | EWMA chart (lam=0.2, q=0.99) | 0 | 0 / 48 | 2104 | 193 | - |
| inf | CUSUM (k=2.0, q=0.99) | 0 | 48 / 48 | 0 | 1298 | - |
| inf | Rolling z-score (window=64, q=0.99) | 0 | 47 / 48 | 0 | 2 | - |
| inf | Isolation Forest (q=0.99) | 0 | 5 / 48 | 0 | 0 | - |
| inf | StreamGuard+ewma (lam=0.2, q=0.99) | 0 | 48 / 48 | 0 | 1 | INF_INPUT 48 |
| inf | StreamGuard+spark (diff, q=0.99) | 0 | 48 / 48 | 0 | 47 | INF_INPUT 48 |
| inf | StreamGuard+isoforest (q=0.99) | 0 | 48 / 48 | 0 | 1 | INF_INPUT 48 |
| short | EWMA chart (lam=0.2, q=0.99) | 0 | 8 / 48 | 0 | n/a | - |
| short | CUSUM (k=2.0, q=0.99) | 0 | 20 / 48 | 0 | n/a | - |
| short | Rolling z-score (window=64, q=0.99) | 0 | 1 / 48 | 0 | n/a | - |
| short | Isolation Forest (q=0.99) | 0 | 4 / 48 | 0 | n/a | - |
| short | StreamGuard+ewma (lam=0.2, q=0.99) | 0 | 48 / 48 | 0 | 3 | BAD_SHAPE 48 |
| short | StreamGuard+spark (diff, q=0.99) | 0 | 48 / 48 | 0 | 45 | BAD_SHAPE 48 |
| short | StreamGuard+isoforest (q=0.99) | 0 | 48 / 48 | 0 | 2 | BAD_SHAPE 48 |
| drop | EWMA chart (lam=0.2, q=0.99) | 0 | 4 / 48 | 0 | n/a | - |
| drop | CUSUM (k=2.0, q=0.99) | 0 | 21 / 48 | 0 | n/a | - |
| drop | Rolling z-score (window=64, q=0.99) | 0 | 3 / 48 | 0 | n/a | - |
| drop | Isolation Forest (q=0.99) | 0 | 3 / 48 | 0 | n/a | - |
| drop | StreamGuard+ewma (lam=0.2, q=0.99) | 0 | 4 / 48 | 0 | 3 | OK 44, SCORE_OVER_TAU 4 |
| drop | StreamGuard+ewma (lam=0.2, q=0.99) +seq | 0 | 48 / 48 | 0 | 4 | SEQUENCE_GAP 48 |
| drop | StreamGuard+spark (diff, q=0.99) | 0 | 2 / 48 | 0 | 0 | OK 46, SCORE_OVER_TAU 2 |
| drop | StreamGuard+spark (diff, q=0.99) +seq | 0 | 48 / 48 | 0 | 43 | SEQUENCE_GAP 48 |
| drop | StreamGuard+isoforest (q=0.99) | 0 | 3 / 48 | 0 | 1 | OK 45, SCORE_OVER_TAU 3 |
| drop | StreamGuard+isoforest (q=0.99) +seq | 0 | 48 / 48 | 0 | 3 | SEQUENCE_GAP 48 |

What this shows:

* **No method crashed.** scikit-learn 1.9's IsolationForest accepts NaN, so the expected crash
  did not happen: it silently scored the faulted blocks and flagged 1 of 48 NaN faults.
* **Raw EWMA and CUSUM are poisoned by one bad sample.** After a NaN, their state stays NaN for
  the rest of the series, so every later block is admitted with a NaN score (1,498 blocks per
  detector). After an inf, EWMA goes NaN the same way (2,104 blocks), and CUSUM latches every
  later block (1,298 clean blocks change to latched). The M4 rolling z-score maps NaN to a score
  of 0, so NaN faults pass silently.
* **The gate latched every NaN, inf and short block (48/48 each) with the matching reason code,
  and admitted no non-finite score.** Clean-block decisions barely moved for `ewma` and
  `isoforest` (0 to 4 changes). `spark`/`diff` changes about 45 clean blocks per fault type,
  because its reset latches the block after every fault with `NO_REFERENCE`.
* **A dropped block is invisible without sequence numbers**, to the gate as much as to the raw
  detectors (StreamGuard flagged 2 to 4 of 48, only when the next block happened to score high).
  With `seq`, all 48 latch with `SEQUENCE_GAP`. The CLI reads a flat value stream and cannot
  detect drops.
* The raw baselines are our own batch implementations without input checks; a production
  detector may validate its input. The measured difference is that validation, framing checks
  and reason codes come built in and logged, independent of the scorer, not that the gate
  detects anomalies better.

Throughput and memory of the gate per scorer (`pivot.py perf`, one thread, all 9,141 NAB blocks
per scorer, one stream per series):

| Gate + scorer | Blocks | Blocks/s | Retained bytes | Peak bytes |
|---|---|---|---|---|
| ewma (default: lam=0.2, ref_window=288, q=0.995) | 9141 | 27,373 | 12,756 | 64,872 |
| spark (diff, q=0.999; 0.1.0.dev0 default) | 9141 | 40,319 | 1,584 | 53,628 |
| isoforest (100 trees, q=0.99) | 9141 | 317 | 375,486 | 1,020,505 |

The `ewma` default is slower than the old `spark` default (27k against 40k blocks/s, both below
the 100k TARGET), and `isoforest` is about 100 times slower, since it calls scikit-learn for every
block.

## Real-data benchmark (REPORTED)

Run on 2026-09-29 with StreamGuard 0.1.0.dev0 (`benchmarks/real/`,
results in `benchmarks/real/results.json` and `results.md`). **REPORTED**: public data, this
code, settings chosen on tuning splits only. This section is kept unchanged after the pivot:
"StreamGuard" here is the `spark` scorer, and "default" is the 0.1.0.dev0 default (`spark`/`diff`,
q = 0.999).

**Data.**
* **NAB** at commit `ea702d7` (47 labelled real series; SHA-256 manifest).
* **SMD** (Server Machine Dataset, OmniAnomaly repo) at commit `7fb0e0a`: 28 machines x 38
  channels, `test` split with `test_label`, SHA-256 manifest.
* **Yahoo S5 is not included**: it needs a Yahoo Webscope licence and registration, and
  unofficial mirrors were not used.
* Data are downloaded by `benchmarks/real/download.py` into a cache outside git
  (`~/.cache/spark-streamguard-bench`). CI never downloads; it runs a tiny generated fixture.

**Protocol** (fixed before any result was computed; details in `benchmarks/real/protocol.py`):
* **Blocks and labels.** Every method sees the same 32-sample blocks and the same labels: a block
  is positive if it overlaps a labelled anomaly (NAB windows, SMD `test_label`).
* **Warm-up.** Block 0 seeds references and blocks 1..64 calibrate every method. The evaluation
  region is blocks 65 onward. Labels inside the warm-up are ignored, not removed: 16 of the 32
  kept NAB series and 16 of the 28 SMD machines have labelled blocks there.
* **Short series.** Series shorter than 97 blocks are **excluded**: 15 of 47 NAB series (listed in
  `results.json`), none of SMD.
* **Threshold.** The same rule for every method: `tau` = the q quantile of that method's warm-up
  block scores.
* **Tuning.** Each method's parameters and q are chosen from small fixed grids by pooled F1 on a
  tuning split: series shuffled with seed 42 and split in half per dataset (NAB 16/16,
  SMD 14/14). They are then applied once to the test split. "Untuned" rows use settings fixed in
  advance.
* **SMD is multivariate.** Each method scores each channel. The block latches if any channel's
  score / tau is >= 1. 225 of 1,064 channels are constant during warm-up and are excluded for all
  methods.

**Methods.**
* StreamGuard `diff` (the shipped default, untuned: W=64, q=0.999), plus `diff`, `median` and
  `detrend` with q tuned.
* Two-sided CUSUM (k in {0.5, 1, 2}).
* EWMA control chart (lambda in {0.1, 0.2, 0.3}).
* Rolling z-score (window 64 or 288).
* Isolation Forest (scikit-learn, 100 trees, seed 42, fitted per series on the warm-up blocks).
* The oes512q-latch v2 configuration (trailing-median residual, q=0.99, untuned) and its fixed
  tau=0.50.
* CUSUM, EWMA, Isolation Forest and oes512q-latch use the oes512q-latch trailing-median
  residual.

In the tables:
* **P / R / F1** are pooled over all test evaluation blocks.
* **mean F1 / mean AP** average over test series with at least one positive block.
* **Event recall** and **delay** use runs of positive blocks. Delay counts from the start of the
  labelled window, and NAB windows start before the anomaly.

**NAB**: 47 series, 15 excluded as too short (< 97 blocks), 16 tune / 16 test; test evaluation blocks 2621, positive 277.

| Method (setting chosen on tune split) | q | P | R | F1 | mean F1 | mean AP | False-latch rate | Event recall | Median delay (blocks) |
|---|---|---|---|---|---|---|---|---|---|
| StreamGuard default (untuned) (mode=diff) | 0.999 | 0.349 | 0.105 | **0.161** | 0.158 | 0.295 | 2.3% | 71.4% | 4.0 |
| StreamGuard diff (mode=diff) | 0.99 | 0.340 | 0.130 | **0.188** | 0.170 | 0.295 | 3.0% | 81.0% | 4.0 |
| StreamGuard median (mode=median) | 0.99 | 0.249 | 0.181 | **0.209** | 0.214 | 0.329 | 6.4% | 76.2% | 4.0 |
| StreamGuard detrend (mode=detrend) | 0.99 | 0.302 | 0.137 | **0.189** | 0.176 | 0.267 | 3.8% | 76.2% | 4.0 |
| CUSUM (k=2.0) | 0.99 | 0.153 | 0.538 | **0.238** | 0.255 | 0.310 | 35.2% | 81.0% | 4.0 |
| EWMA chart (lam=0.2) | 0.99 | 0.316 | 0.238 | **0.272** | 0.281 | 0.369 | 6.1% | 85.7% | 4.0 |
| Rolling z-score (window=288) | 0.99 | 0.323 | 0.036 | **0.065** | 0.071 | 0.259 | 0.9% | 47.6% | 4.0 |
| Isolation Forest | 0.99 | 0.517 | 0.166 | **0.251** | 0.258 | 0.392 | 1.8% | 81.0% | 4.0 |
| oes512q-latch v2 config (q=0.99) | 0.99 | 0.261 | 0.195 | **0.223** | 0.208 | 0.338 | 6.5% | 76.2% | 4.0 |
| oes512q-latch fixed tau=0.50 | fixed tau | 0.103 | 0.921 | **0.185** | 0.230 | 0.338 | 94.8% | 100.0% | 0.0 |

Significance vs the best baseline on test (ewma), paired per test series with at least one positive evaluation block (difference = StreamGuard minus baseline; 95% bootstrap CI, 10,000 resamples; two-sided Wilcoxon signed-rank):

| StreamGuard setting | Metric | n | Mean diff | 95% CI | Wilcoxon p | W/L |
|---|---|---|---|---|---|---|
| StreamGuard default (untuned) (mode=diff) | F1 | 15 | -0.123 | [-0.220, -0.035] | 0.028 | 3/9 |
| StreamGuard default (untuned) (mode=diff) | AP | 15 | -0.075 | [-0.126, -0.030] | 0.005 | 3/12 |
| StreamGuard diff (mode=diff) | F1 | 15 | -0.111 | [-0.212, -0.021] | 0.033 | 3/10 |
| StreamGuard diff (mode=diff) | AP | 15 | -0.075 | [-0.126, -0.030] | 0.005 | 3/12 |
| StreamGuard median (mode=median) | F1 | 15 | -0.067 | [-0.172, +0.026] | 0.182 | 4/8 |
| StreamGuard median (mode=median) | AP | 15 | -0.041 | [-0.079, -0.008] | 0.041 | 3/12 |
| StreamGuard detrend (mode=detrend) | F1 | 15 | -0.105 | [-0.205, -0.017] | 0.034 | 3/9 |
| StreamGuard detrend (mode=detrend) | AP | 15 | -0.103 | [-0.155, -0.058] | 0.000 | 2/13 |

**SMD**: 28 series, 0 excluded as too short (< 97 blocks), 14 tune / 14 test; test evaluation blocks 10071, positive 529.

| Method (setting chosen on tune split) | q | P | R | F1 | mean F1 | mean AP | False-latch rate | Event recall | Median delay (blocks) |
|---|---|---|---|---|---|---|---|---|---|
| StreamGuard default (untuned) (mode=diff) | 0.999 | 0.106 | 0.881 | **0.189** | 0.212 | 0.507 | 41.2% | 97.5% | 0.0 |
| StreamGuard diff (mode=diff) | 0.999 | 0.106 | 0.881 | **0.189** | 0.212 | 0.507 | 41.2% | 97.5% | 0.0 |
| StreamGuard median (mode=median) | 0.999 | 0.106 | 0.898 | **0.190** | 0.204 | 0.498 | 42.0% | 97.5% | 0.0 |
| StreamGuard detrend (mode=detrend) | 0.999 | 0.115 | 0.887 | **0.203** | 0.228 | 0.511 | 37.9% | 97.5% | 0.0 |
| CUSUM (k=2.0) | 0.999 | 0.054 | 1.000 | **0.102** | 0.101 | 0.217 | 97.3% | 100.0% | 0.0 |
| EWMA chart (lam=0.2) | 0.999 | 0.107 | 0.917 | **0.192** | 0.208 | 0.496 | 42.4% | 95.0% | 0.0 |
| Rolling z-score (window=64) | 0.999 | 0.088 | 0.592 | **0.153** | 0.159 | 0.180 | 34.1% | 92.5% | 0.0 |
| Isolation Forest | 0.999 | 0.102 | 0.733 | **0.179** | 0.184 | 0.345 | 35.8% | 69.4% | 0.0 |
| oes512q-latch v2 config (q=0.99) | 0.99 | 0.099 | 0.945 | **0.179** | 0.190 | 0.519 | 47.6% | 99.4% | 0.0 |
| oes512q-latch fixed tau=0.50 | fixed tau | 0.053 | 1.000 | **0.100** | 0.098 | 0.521 | 100.0% | 100.0% | 0.0 |

Significance vs the best baseline on test (ewma), paired per test series with at least one positive evaluation block (difference = StreamGuard minus baseline; 95% bootstrap CI, 10,000 resamples; two-sided Wilcoxon signed-rank):

| StreamGuard setting | Metric | n | Mean diff | 95% CI | Wilcoxon p | W/L |
|---|---|---|---|---|---|---|
| StreamGuard default (untuned) (mode=diff) | F1 | 14 | +0.003 | [-0.021, +0.027] | 0.808 | 8/6 |
| StreamGuard default (untuned) (mode=diff) | AP | 14 | +0.012 | [-0.049, +0.070] | 0.715 | 8/6 |
| StreamGuard diff (mode=diff) | F1 | 14 | +0.003 | [-0.021, +0.027] | 0.808 | 8/6 |
| StreamGuard diff (mode=diff) | AP | 14 | +0.012 | [-0.049, +0.070] | 0.715 | 8/6 |
| StreamGuard median (mode=median) | F1 | 14 | -0.005 | [-0.029, +0.020] | 0.855 | 7/7 |
| StreamGuard median (mode=median) | AP | 14 | +0.002 | [-0.034, +0.037] | 0.917 | 6/7 |
| StreamGuard detrend (mode=detrend) | F1 | 14 | +0.019 | [-0.010, +0.050] | 0.241 | 8/6 |
| StreamGuard detrend (mode=detrend) | AP | 14 | +0.016 | [-0.043, +0.070] | 0.626 | 8/6 |

**Where StreamGuard wins and loses (REPORTED):**
* **NAB: StreamGuard loses.** The best baseline is the EWMA chart (F1 0.272, mean AP 0.369),
  followed by Isolation Forest (0.251 / 0.392) and CUSUM (F1 0.238). The best StreamGuard setting
  is `median` (F1 0.209); the shipped default reaches only F1 0.161.
  * Against EWMA, per series, the default, `diff` and `detrend` are significantly worse on F1
    (Wilcoxon p about 0.03) and on AP (p <= 0.005).
  * `median` is not significantly worse on F1 (p = 0.18) but is on AP (p = 0.04).
  * The default's strength is precision: its false-latch rate is 2.3%, against 6.1% for EWMA
    and 35% for CUSUM. It pays for that with low recall (0.105).
* **SMD: a tie at a poor level.** All quantile-threshold methods except CUSUM land at F1 0.15
  to 0.20 with false-latch rates of 34% to 48%. CUSUM reaches F1 0.10 with 97% false latches.
  * StreamGuard `detrend` has the highest F1 (0.203), and the default is at 0.189 against 0.192
    for EWMA. None of the differences is significant (p 0.24 to 0.92).
  * The oes512q-latch score has the highest mean AP (0.519 to 0.521, against 0.507 for the
    default).
  * The "any of 38 channels" rule with a per-channel q=0.999 threshold from only 64 warm-up
    blocks produces many false alarms for every method. This is a weakness of the protocol and
    of per-channel calibration, not only of StreamGuard.
* **Settings at grid edges.** On NAB the tuning split chose q = 0.99 for every
  quantile-threshold method (the lowest value in the grid), and on SMD it chose q = 0.999 (the
  highest). CUSUM chose k = 2.0 on both. So the grids may be too narrow. They were not widened
  after seeing test results.

**Throughput and memory** (Intel(R) Xeon(R) Processor, single thread, Python
3.13.5, numpy 2.5.3):
* `StreamGuard.process()` streams **43,122 blocks/s** on real NAB blocks. That
  is below the spec's TARGET of 100k blocks/s.
* A `StreamGuard` stream retains about 1 KiB after calibration, with a peak of about 53 KiB while
  the 64-block warm-up buffer is alive (tracemalloc, longest NAB series). That is well within the
  10 MB TARGET.
* The baselines are batch numpy implementations, so their throughput and memory (in `results.md`)
  are not comparable to a streaming guard.

**NAB official scorer** (NAB's own code at `ea702d7`, 16 NAB test series, normalised with the
null detector on the same sub-corpus). Caveats:
* Our detectors emit one detection at the end of each latched block and are scored at threshold
  0.5 with **no** NAB threshold optimisation.
* They are silent during the first 2,080 samples (the warm-up), so NAB windows there count as
  misses.
* The published NAB detectors are re-scored from NAB's result files with NAB's official
  thresholds, which were optimised on the full corpus *with labels*.
* The sub-corpus is not NAB's full corpus. This is **not** comparable to the public NAB
  scoreboard.

| Detector | standard | low FP | low FN |
|---|---|---|---|
| streamguard_diff | 34.57 | 23.43 | 40.22 |
| streamguard_median | 21.98 | -2.7 | 31.82 |
| streamguard_detrend | 29.44 | 15.3 | 35.79 |
| cusum | -81.54 | -215.92 | -35.17 |
| ewma | 28.48 | 5.2 | 38.18 |
| zscore | 24.44 | 20.94 | 26.4 |
| iforest | 37.77 | 31.0 | 42.35 |
| sg_default | 31.33 | 22.78 | 36.04 |
| oes_latch | 21.78 | -3.09 | 31.69 |
| oes_fixed | -293.93 | -657.69 | -171.71 |
| numenta | 79.68 | 72.75 | 84.43 |
| windowedGaussian | 32.45 | 1.88 | 44.87 |
| relativeEntropy | 53.19 | 46.96 | 56.67 |
| knncad | 66.83 | 52.57 | 73.84 |
| bayesChangePt | 18.5 | -6.72 | 34.76 |
| random | 20.17 | 14.63 | 29.61 |

On the NAB score, Isolation Forest (37.8) and StreamGuard `diff` (34.6) come out ahead of EWMA
(28.5). The shipped default scores 31.3. All of our detectors are far below the published
`numenta` (79.7) and `knncad` (66.8) detectors, and StreamGuard `median` (22.0) is close to
NAB's `random` detector (20.2).

## Evidence tags

Every number in this project is tagged **SYNTHETIC** (generated data, e.g. the demo and tests),
**REPORTED** (measured on public data), **TARGET** (a goal) or **UNRUN** (planned, not run).
Spec TARGETs against measurements:
* throughput of 100k blocks/s per core: not met. REPORTED: 43,122 blocks/s for the M4 `spark`
  default; after the pivot 27,373 blocks/s for the `ewma` default and 317 for `isoforest`;
* memory under 10 MB per stream: met. REPORTED peak: about 53 KiB (`spark`), 63 KiB (`ewma`),
  1.0 MB (`isoforest`);
* false-latch rate of 1% or less: not met. 1.73% SYNTHETIC held-out (`spark`); REPORTED on the
  NAB test split: 2.3% for the M4 default, 3.1% for the new `ewma` default (72 of 2,344
  negative evaluation blocks); on SMD 43.8% for the new default.

## Limitations

Research prototype, not a certified safety system and not a product. No quantum claims.

* The original `spark` score loses to a plain EWMA chart on NAB and ties at a poor level on SMD
  (REPORTED above). The gate does not make any scorer more accurate on clean data.
* The 1% false-latch TARGET is not met (see Evidence tags), and the long warm-up admits nothing
  for the first 65 valid blocks; see [Remaining weaknesses](#remaining-weaknesses-synthetic-evidence).
* Thresholds are self-calibrated on a warm-up that is assumed clean; contaminated warm-ups are
  UNRUN.
* Parity rebuild (M2), streaming CLI extensions (M3) and the LLM adapter (M5) are not done yet
  (UNRUN).

## Roadmap

* M1: scorer, latch, calibrator, stream guard, CLI, tests, CI (done).
* M2: parity rebuild and erasure tests.
* M3: streaming CLI extensions (MQTT input).
* M4: benchmark vs CUSUM, EWMA, Isolation Forest, rolling z-score and oes512q-latch on NAB and
  SMD (done, REPORTED above; Yahoo S5 not included for licence reasons).
* Post-M4 pivot: pluggable scorers (`spark`, `ewma`, `isoforest`, custom), default re-chosen on
  tuning splits, gate-vs-raw check and fault-injection benchmark (done, REPORTED above).
* v0.1.0 release and Zenodo DOI (prepared).
* M5: LLM output adapter.

## How to cite

See [CITATION.cff](CITATION.cff). Cite the concept DOI [10.5281/zenodo.23462518](https://doi.org/10.5281/zenodo.23462518) for all versions,
or the version DOI [10.5281/zenodo.23462519](https://doi.org/10.5281/zenodo.23462519) for v0.1.0.

> Brisson, J.-F. (2026). *Spark StreamGuard: a fail-closed, auditable gate for any streaming
> detector* (Version 0.1.0) [Computer software]. Spark AI NLP, Fredericton, New Brunswick,
> Canada. https://github.com/sparkainlp-x/spark-streamguard

If you use the real-data benchmark, please also cite the datasets: NAB (Ahmad et al., 2017,
[doi:10.1016/j.neucom.2017.04.070](https://doi.org/10.1016/j.neucom.2017.04.070)) and SMD
(Su et al., 2019, [doi:10.1145/3292500.3330672](https://doi.org/10.1145/3292500.3330672)).

## License

Copyright (c) 2026 Jean-François Brisson, Spark AI NLP.

Released under the MIT License; see [LICENSE](LICENSE).
