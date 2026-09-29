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
    assert (demo[new]["caught"], demo[new]["false"]) == (3, 2)  # 3 before the diff reset
    assert (gen[old]["caught"], gen[old]["false"], gen[old]["negatives"]) == (58, 135, 1386)
    assert (gen[new]["caught"], gen[new]["false"]) == (96, 84)
    assert "| diff |" in ev.table(gen, "t")


def test_series_are_seeded():
    ev = load()
    a, b = ev.make_series("step", 3), ev.make_series("step", 3)
    assert a["spikes"] == b["spikes"] and a["step"] == b["step"]
    assert (a["x"] == b["x"]).all()


SWEEP = Path(__file__).resolve().parents[1] / "benchmarks" / "warmup_sweep.py"


def load_sweep():
    spec = importlib.util.spec_from_file_location("warmup_sweep", SWEEP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_warmup_sweep_numbers_reproduce(capsys):
    sw = load_sweep()
    best, why, results = sw.tune()
    assert best == (64, 0.999) and why.startswith("rule 2")
    assert len(results) == 15
    assert not any(r["false_rate"] <= 0.01 for r in results.values())  # 1% target not met
    r = results[best]
    assert (r["caught"], r["spikes"], r["false"], r["negatives"]) == (234, 240, 57, 3750)
    r16 = results[(16, 0.99)]
    assert (r16["caught"], r16["false"]) == (238, 245)
    ho = sw.heldout(best)["overall"]
    chosen = ho["chosen: diff, W=64, q=0.999"]
    assert (chosen["caught"], chosen["false"], chosen["negatives"]) == (237, 65, 3750)
    sw.main(["x", "tune"])
    assert "Chosen: W=64, q=0.999" in capsys.readouterr().out


def test_sweep_rules_1_and_3():
    sw = load_sweep()
    fake = {
        (16, 0.99): {"recall": 0.95, "false_rate": 0.009},
        (32, 0.99): {"recall": 0.97, "false_rate": 0.008},
        (64, 0.99): {"recall": 0.99, "false_rate": 0.02},
    }
    assert sw.choose(fake) == (
        (32, 0.99),
        "rule 1: max recall with false-latch rate <= 1% (target met)",
    )
    low = {k: {"recall": 0.5 + i / 10, "false_rate": 0.05} for i, k in enumerate(fake)}
    best, why = sw.choose(low)
    assert best == (64, 0.99) and why.startswith("rule 3")


def test_sweep_corpus_is_fair():
    sw = load_sweep()
    for split in (sw.TUNING, sw.HELDOUT):
        for s in sw.corpus(split):
            assert len(s["spikes"]) == sw.N_SPIKES
            assert min(s["spikes"]) >= sw.EVAL_START > max(sw.WARMUPS) + 1
            if s["step"] is not None:
                assert s["step"] >= sw.EVAL_START
    a, b = sw.make_series(sw.TUNING, 0, 0), sw.make_series(sw.HELDOUT, 0, 0)
    assert not (a["x"] == b["x"]).all()
