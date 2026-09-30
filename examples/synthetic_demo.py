"""SYNTHETIC demo: seeded random walk with injected spikes, run through StreamGuard.

Seed 42. The data are generated here; nothing is measured on real data. Output is SYNTHETIC.

The demo pins the ``spark`` scorer (diff reference), the scorer this demo was written for. The
library default scorer is ``ewma`` (chosen on real-data tuning splits); on this random walk it
latches most post-warm-up blocks because it reads random-walk drift as a level shift (see
README, "Choosing a scorer").
"""

from __future__ import annotations

import numpy as np

from streamguard import StreamGuard

SEED = 42
N_BLOCKS = 80
WARMUP = 16
SPIKE_BLOCKS = (30, 47, 63)  # blocks that get one injected spike each
SPIKE_SIZE = 8.0


def make_series() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    x = np.cumsum(rng.normal(0.0, 0.1, size=N_BLOCKS * 32))  # random walk
    for b in SPIKE_BLOCKS:
        x[b * 32 + int(rng.integers(0, 32))] += SPIKE_SIZE
    x[70 * 32 + 3] = np.nan  # one sensor dropout
    return x


def main() -> None:
    x = make_series()
    guard = StreamGuard(warmup_blocks=WARMUP, scorer="spark", quantile=0.999)
    counts: dict[str, int] = {}
    print(f"SYNTHETIC demo | seed={SEED} blocks={N_BLOCKS} warmup={WARMUP} spikes={SPIKE_BLOCKS}")
    for block in x.reshape(N_BLOCKS, 32):
        d = guard.process(block)
        counts[d.reason.value] = counts.get(d.reason.value, 0) + 1
        if d.event == "CALIBRATED":
            cal = guard.calibration
            print(f"block {d.index:2d}: CALIBRATED tau={cal.tau:.4f} scale={cal.scale:.4f}")
        elif not d.admit and d.reason.value not in ("NOT_CALIBRATED", "NO_REFERENCE"):
            score = "-" if d.score is None else f"{d.score:.4f}"
            print(f"block {d.index:2d}: LATCH reason={d.reason.value} score={score}")
    print("reason counts:", dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
