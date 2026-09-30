# Post-M4 pivot results (REPORTED)

## Default-scorer rule (tuning splits only)

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

## M4 test split: gate vs raw baseline

| Dataset | Detector (M4-tuned setting) | Raw F1 | Gate F1 | Raw AP | Gate AP | Flag disagreements |
|---|---|---|---|---|---|---|
| NAB | EWMA chart (lam=0.2) q=0.99 | 0.272 | 0.272 | 0.369 | 0.369 | 0 / 2621 |
| NAB | Isolation Forest q=0.99 | 0.251 | 0.251 | 0.392 | 0.392 | 0 / 2621 |
| NAB | new default: ewma (lam=0.2, ref_window=288) q=0.995 | | 0.246 | | 0.355 | |
| SMD | EWMA chart (lam=0.2) q=0.999 | 0.192 | 0.192 | 0.496 | 0.496 | 0 / 10071 |
| SMD | Isolation Forest q=0.999 | 0.179 | 0.179 | 0.345 | 0.345 | 0 / 10071 |
| SMD | new default: ewma (lam=0.2, ref_window=288) q=0.995 | | 0.190 | | 0.503 | |

## Fault injection on NAB test series

16 series x 3 faults per type.

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

## Throughput and memory (single thread, NAB)

| Gate + scorer | Blocks | Blocks/s | Retained bytes | Peak bytes |
|---|---|---|---|---|
| ewma (default: lam=0.2, ref_window=288, q=0.995) | 9141 | 27,373 | 12,756 | 64,872 |
| spark (diff, q=0.999; 0.1.0.dev0 default) | 9141 | 40,319 | 1,584 | 53,628 |
| isoforest (100 trees, q=0.99) | 9141 | 317 | 375,486 | 1,020,505 |
