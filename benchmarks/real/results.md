Date 2026-09-29, StreamGuard 0.1.0.dev0. REPORTED.

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

Hardware: Intel(R) Xeon(R) Processor, 8 logical CPUs, single thread, Python 3.13.5, numpy 2.5.3.

`StreamGuard.process()` streaming throughput on NAB blocks: 43,122 blocks/s (9,141 blocks).

Batch scoring throughput, NAB (blocks x channels per second, scores only, test series, chosen settings; batch numpy implementations):

| Family | per s |
|---|---|
| cusum | 8,260 |
| ewma | 78,414 |
| iforest | 3,330 |
| oes_fixed | 74,391 |
| oes_latch | 76,515 |
| sg_default | 64,177 |
| streamguard_detrend | 55,578 |
| streamguard_diff | 67,538 |
| streamguard_median | 41,152 |
| zscore | 40,995 |

Batch scoring throughput, SMD (blocks x channels per second, scores only, test series, chosen settings; batch numpy implementations):

| Family | per s |
|---|---|
| cusum | 140,511 |
| ewma | 280,765 |
| iforest | 12,258 |
| oes_fixed | 310,047 |
| oes_latch | 311,439 |
| sg_default | 83,103 |
| streamguard_detrend | 62,213 |
| streamguard_diff | 82,006 |
| streamguard_median | 48,286 |
| zscore | 972,910 |

Memory (tracemalloc, longest NAB series realKnownCause/machine_temperature_system_failure.csv, 709 blocks):

| What | Retained | Peak |
|---|---|---|
| StreamGuard.process (per stream) | 1.0 KiB | 53.0 KiB |
| StreamGuard diff (mode=diff) (batch scoring) | - | 383.0 KiB |
| StreamGuard median (mode=median) (batch scoring) | - | 383.8 KiB |
| StreamGuard detrend (mode=detrend) (batch scoring) | - | 382.9 KiB |
| CUSUM (k=0.5) (batch scoring) | - | 367.6 KiB |
| CUSUM (k=1.0) (batch scoring) | - | 367.6 KiB |
| CUSUM (k=2.0) (batch scoring) | - | 367.5 KiB |
| EWMA chart (lam=0.1) (batch scoring) | - | 709.3 KiB |
| EWMA chart (lam=0.2) (batch scoring) | - | 709.3 KiB |
| EWMA chart (lam=0.3) (batch scoring) | - | 709.3 KiB |
| Rolling z-score (window=64) (batch scoring) | - | 8.4 KiB |
| Rolling z-score (window=288) (batch scoring) | - | 10.0 KiB |
| Isolation Forest (batch scoring) | - | 641.4 KiB |
| StreamGuard default (untuned) (mode=diff) (batch scoring) | - | 383.1 KiB |
| oes512q-latch v2 config (q=0.99) (batch scoring) | - | 544.5 KiB |
| oes512q-latch fixed tau=0.50 (batch scoring) | - | 544.5 KiB |

NAB official scorer, 16 NAB test series (normalised; see caveats):

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
