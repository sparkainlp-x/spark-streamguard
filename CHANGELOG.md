# Changelog

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
