# Spec notes: changes against product spec v0.1 (draft)

Dated notes on where the implementation departs from the v0.1 draft spec and why. Evidence tags
as in the README.

## 2026-09-29: post-M4 pivot

**What M4 showed (REPORTED).** On the M4 real-data benchmark the spec's core algorithm, the
`S = 0.45*Peak + 0.35*RMS + 0.20*MeanAbs` residual score, lost. On NAB it was significantly worse
than an EWMA control chart (test F1 0.161 against 0.272, p = 0.028). On SMD every method tied at
F1 of about 0.19 with false-latch rates of about 40%. The spec's risk section anticipated this
("may not beat CUSUM/Isolation Forest"), and its mitigation, positioning on auditability and
fail-closed behaviour, is now the product.

**Changes.**

1. Spec section 1 (pitch) now reads: *a fail-closed, auditable gate for any streaming detector*.
   The block score is no longer the product; it is one pluggable scorer (`spark`) among several.
2. Spec section 5 (core algorithm): the residual score is kept as the `spark` scorer. The
   default scorer is now an EWMA control chart (`ewma`, lambda 0.2, trailing-median window of
   288 samples) with quantile 0.995. It was chosen by a rule fixed in advance and applied to the
   M4 tuning splits only (`benchmarks/real/pivot.py choose`). The fixed tau = 0.50 remains
   available.
3. Spec section 6 (components): `streamguard-core` gains a `Scorer` protocol and an optional
   IsolationForest adapter (scikit-learn extra). The core stays numpy-only.
4. Fail-closed coverage grows by one reason code, `SEQUENCE_GAP`, for dropped, repeated or
   reordered blocks when the producer supplies sequence numbers. Without them, a missing block
   cannot be detected (measured: the gate caught 2 to 4 of 48 dropped blocks without `seq`, and
   48 of 48 with it).
5. Spec section 7 (benchmarks) adds two measurements:
   - gate vs raw detector on clean test data: identical decisions (0 disagreements);
   - fault injection (NaN, inf, short and dropped blocks): what raw detectors do versus the
     gate. Results are in the README and `benchmarks/real/pivot_results.md`.

**Still open.** Parity rebuild (M2), MQTT input (M3) and the LLM adapter (M5) are unchanged. The
TARGETs are unchanged and still not met for throughput (27k blocks/s for the `ewma` default) or
for the false-latch rate (3.1% on the NAB test split).
