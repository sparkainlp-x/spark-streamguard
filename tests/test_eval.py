"""The SYNTHETIC reference-mode numbers quoted in the README are reproducible."""

import importlib.util
from pathlib import Path

EVAL = Path(__file__).resolve().parents[1] / "benchmarks" / "synthetic_reference_eval.py"


def load():
    spec = importlib.util.spec_from_file_location("synthetic_reference_eval", EVAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_readme_numbers_reproduce():
    ev = load()
    res = ev.run()
    demo = ev.summarize(res, ("demo",))
    gen = ev.summarize(res, ev.FAMILIES)
    old, new = "median (rb=4, old default)", "diff"
    assert (demo[old]["caught"], demo[old]["false"]) == (3, 22)
    assert (demo[new]["caught"], demo[new]["false"]) == (3, 3)
    assert (gen[old]["caught"], gen[old]["false"], gen[old]["negatives"]) == (58, 135, 1386)
    assert (gen[new]["caught"], gen[new]["false"]) == (96, 84)
    assert "| diff |" in ev.table(gen, "t")


def test_series_are_seeded():
    ev = load()
    a, b = ev.make_series("step", 3), ev.make_series("step", 3)
    assert a["spikes"] == b["spikes"] and a["step"] == b["step"]
    assert (a["x"] == b["x"]).all()
