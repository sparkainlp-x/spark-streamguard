# Changelog

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
