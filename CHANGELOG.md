# Changelog

## 0.1.0 (unreleased; date set when tagged)

First public release candidate. It bundles the development milestones listed below.

- Version set to 0.1.0 in `pyproject.toml`, `streamguard.__version__`, CITATION.cff and
  .zenodo.json.
- scikit-learn stays optional: the core installs with numpy only; isoforest tests carry a
  `sklearn` marker and are skipped cleanly when it is missing, and `IsoForestScorer` checks its
  options before importing scikit-learn. New test for the clear ImportError message.
- Tests treat warnings as errors (`filterwarnings = ["error"]`); CI runs on every branch, on
  Python 3.10-3.13 with all extras, plus a core-only job without scikit-learn.
- Wording: input checks are described as "checked"; SYNTHETIC caveats now say they say nothing
  about performance on real data.
- Copyright line names Jean-François Brisson, Spark AI NLP (licence stays MIT). CITATION.cff and
  .zenodo.json carry the full affiliation; .zenodo.json `references` oes512q-latch and the NAB
  and SMD dataset papers.
- README: install options, Limitations, How to cite (DOI placeholder), License.

## Unreleased: post-M4 pivot, a fail-closed gate for any scorer

Positioning changed after M4 lost: the value is the fail-closed, audited gate, not the score.

- New `streamguard.scorers`: a small `Scorer` protocol (`start`, `warmup`, `fit`, `fit_fixed`,
  `score`, `on_invalid`, `params`) with bounded state. Built-ins:
  - `spark`: the original residual score with every reference mode.
  - `ewma`: an EWMA control chart on the trailing-median residual, numpy only.
  - `isoforest`: an optional scikit-learn adapter, fitted on the warm-up only, with a
    fixed-size model (new extra `isoforest`).
  Custom scorer objects are accepted too.
- `StreamGuard(scorer=..., scorer_params=...)` and CLI `--scorer`, `--ewma-lambda`,
  `--ref-window`, `--trees`. Validation, reason codes, calibration, drift and the audit log are
  unchanged and shared by every scorer. `reference=` / `ref_blocks=` / `ewma_alpha=` without
  `scorer=` select `spark`.
- **Default changed**: scorer `ewma` (lambda 0.2, `ref_window` 288) and quantile 0.995 (was
  `spark`/`diff`, 0.999). Chosen by a rule committed before it was run, on the M4 tuning splits
  only: highest mean of the NAB and SMD tuning pooled F1 (`benchmarks/real/pivot.py choose`).
- New optional `seq` argument, `process(block, seq=...)`, and new reason `SEQUENCE_GAP` for
  dropped, repeated or reordered blocks.
- Audit schema `v: 2`: the config record has `scorer` and `scorer_params` instead of flat
  reference fields.
- `benchmarks/real/gate.py` and `pivot.py` (REPORTED 2026-09-29, `pivot_results.json` / `.md`):
  - The gate matches the raw EWMA and Isolation Forest baselines on the M4 test split with 0
    flag disagreements (NAB 2,621 and SMD 10,071 blocks).
  - New default test F1: NAB 0.246, SMD 0.190.
  - Fault injection on the NAB test series: the gate latched 48/48 NaN, inf and short blocks
    with reason codes. Raw EWMA and CUSUM were poisoned by one NaN, with 1,498 later blocks
    admitted with NaN scores. Dropped blocks are caught only with `seq`.
  - Throughput: `ewma` 27k blocks/s, `isoforest` 317 blocks/s.
- The M4 section of the README, `results.json` and `run.py` are unchanged; `run.py` now pins the
  M4 guard settings.
- The demo and the SYNTHETIC sweeps pin the `spark` scorer they were written for. The README
  documents that the `ewma` default does badly on the random-walk demo.

## Unreleased: milestone 4, real-data benchmark (REPORTED)

- `benchmarks/real/`:
  - `download.py`: pinned NAB `ea702d7` and SMD `7fb0e0a` files, SHA-256 manifests, cached
    outside git.
  - `protocol.py`: shared blocking, labels, warm-up and thresholds; baselines; metrics.
  - `run.py`: tune/test split, significance, throughput, memory.
  - `make_tables.py`: renders the tables.
  - `nab_official.py`: optional NAB official scorer.
- Results (`results.json`, `results.md`, README), REPORTED 2026-09-29:
  - NAB: StreamGuard loses to an EWMA chart (default F1 0.161 vs 0.272; significant on F1
    and AP).
  - SMD: a tie at F1 around 0.19 with false-latch rates around 40% for all methods.
  - `StreamGuard.process()` runs at about 43k blocks/s (below the 100k TARGET) with about
    53 KiB peak per stream.
- Yahoo S5 not included (licence/registration required).
- New optional extra `bench` (scikit-learn, scipy); CI installs it and runs a tiny generated
  fixture, with no downloads.

## Unreleased: warm-up and quantile defaults

- `StreamGuard` / CLI defaults changed to `warmup_blocks = 64` (was 32) and `quantile = 0.999`
  (was 0.99), chosen by a SYNTHETIC sweep (`benchmarks/warmup_sweep.py`). The sweep covered
  W in {16, 24, 32, 48, 64} and q in {0.99, 0.995, 0.999} for `diff`. Every setting is scored
  on the same post-warm-up blocks with the same spikes, the selection rule was fixed in
  advance, and the result was checked on 40 held-out series. No setting met the 1% false-latch
  TARGET (rule 2 applied). Held-out: recall 237/240, false latches 65/3750 (1.73%), against
  203/3750 (5.41%) for W=16, q=0.99. SYNTHETIC only.
- `calibrate_tau()` keeps its own default quantile of 0.99 (oes512q-latch parity). New
  constants `STREAM_WARMUP_BLOCKS`, `STREAM_QUANTILE`.
- `diff` reference is reset by an invalid block: the next valid block latches with
  `NO_REFERENCE` instead of scoring a difference across the gap (was a false `SCORE_OVER_TAU` in
  the demo). Reference models gained `on_invalid()`.
- `benchmarks/synthetic_reference_eval.py` pins quantile 0.99 so its historical comparison
  stays reproducible.

## Unreleased: trend-following reference modes

- New `streamguard.reference` module with O(1)-state reference models: `diff` (first
  difference), `detrend` (in-block least-squares line), `ewma` (level tracking) and `median`
  (trailing median, alias `trailing_median`), plus fixed scalar/vector references.
- **Default reference changed from `median` (4 blocks) to `diff`**, chosen by a SYNTHETIC
  comparison (`benchmarks/synthetic_reference_eval.py`, demo plus 24 seeded series). On the 24
  series: spikes caught 96/96 vs 58/96, false latches 84/1386 vs 135/1386. On the demo: false
  latches 3 vs 22. SYNTHETIC only; no real-data benchmark.
- Warm-up calibration uses the same reference mode as scoring.
- CLI: `--reference {median,ewma,detrend,diff}` and `--ewma-alpha`.
- Config record: `reference` is now the mode name; `ref_window` / `ewma_alpha` /
  `reference_value` appear only for the mode that uses them.
- Remaining weaknesses documented in the README (false-latch rate about 6% on SYNTHETIC series
  with a 16-block warm-up; `diff` cannot see slow ramps; stationary noise slightly worse).

## 0.1.0.dev0 (unreleased): milestone 1

- `streamguard.score`: 32-value block residual and weighted score
  `S = 0.45*Peak + 0.35*RMS + 0.20*MeanAbs` (ported from oes512q-latch).
- `streamguard.latch`: fail-closed admit/latch decision with reason codes.
- `streamguard.calibrate`: warm-up quantile tau with robust scale (ported from oes512q-latch
  `oes_calibration`), optional bounded drift, fixed tau = 0.50 still available.
- `streamguard.stream`: `StreamGuard` with O(1) memory per stream, `Decision`, JSONL
  `AuditLogWriter`.
- CLI: `python -m streamguard` / `streamguard` reads CSV or stdin, writes JSONL decisions.
- Tests (pytest), CI on Python 3.10 to 3.13, SYNTHETIC demo.
- No benchmarks run yet; all performance figures are TARGET or UNRUN.
