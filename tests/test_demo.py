"""The SYNTHETIC README demo runs and latches every injected spike and the NaN dropout."""

import importlib.util
from pathlib import Path

from streamguard import StreamGuard

DEMO = Path(__file__).resolve().parents[1] / "examples" / "synthetic_demo.py"


def load_demo():
    spec = importlib.util.spec_from_file_location("synthetic_demo", DEMO)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_demo_latches_spikes_and_nan(capsys):
    demo = load_demo()
    demo.main()
    out = capsys.readouterr().out
    assert out.startswith("SYNTHETIC demo")
    g = StreamGuard(warmup_blocks=demo.WARMUP, scorer="spark", quantile=0.999)
    ds = [g.process(b) for b in demo.make_series().reshape(demo.N_BLOCKS, 32)]
    for b in demo.SPIKE_BLOCKS:
        assert ds[b].reason.value == "SCORE_OVER_TAU"
    assert ds[70].reason.value == "NAN_INPUT"
