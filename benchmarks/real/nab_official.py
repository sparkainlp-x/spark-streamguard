"""Optional: NAB's official scorer on StreamGuard's and the baselines' NAB test-split detections.

Needs a NAB checkout at the pinned commit (``--nab``) and a Python with ``pandas>=2,<3`` and
``simplejson`` for NAB's code (``--nab-python``). Reads the chosen settings from results.json,
recomputes detections on the NAB **test** series, writes NAB result files (anomaly_score = 1 on
the last sample of each latched block, 0 elsewhere, 0 during warm-up) and runs NAB's scorer at
threshold 0.5 with **no** NAB threshold optimisation. Published NAB detectors are re-scored on
the same sub-corpus from NAB's own result files with NAB's official thresholds (which NAB
optimised on its full corpus with labels), for context only. Normalised with the null
detector on the same sub-corpus. Not comparable to the public NAB scoreboard.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import protocol as P  # noqa: E402
from data import load_nab  # noqa: E402
from download import cache_dir  # noqa: E402

PUBLISHED = ("numenta", "windowedGaussian", "relativeEntropy", "knncad", "bayesChangePt", "random")
PROFILES = ("standard", "reward_low_FP_rate", "reward_low_FN_rate")


def alias(fam: str) -> str:
    # NAB's normaliser splits detector names on "_": use underscore-free aliases.
    return "sg" + "".join(w.capitalize() for w in fam.split("_"))


def find_method(label: str, fam: str) -> P.Method:
    for m in P.method_grid() + P.FIXED_METHODS:
        if m.label() == label and m.family == fam:
            return m
    raise KeyError(label)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nab", type=Path, required=True)
    p.add_argument("--nab-python", default=sys.executable)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--results", type=Path, default=HERE / "results.json")
    args = p.parse_args(argv)
    report = json.loads(args.results.read_text(encoding="utf-8"))
    nab_commit = json.loads((HERE / "nab_manifest.json").read_text())["nab_commit"]
    got = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=args.nab, capture_output=True, text=True, check=True
    ).stdout.strip()
    if got != nab_commit:
        raise SystemExit(f"NAB checkout at {got}, expected {nab_commit}")
    ds = report["datasets"]["nab"]
    series = {s.name: s for s in load_nab(args.data_dir or cache_dir())}
    test = [series[k] for k in ds["test"]]
    if args.work.exists():
        shutil.rmtree(args.work)
    data, res = args.work / "data", args.work / "results"
    root = args.data_dir or cache_dir()
    for s in test:
        (data / s.name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(root / "nab" / s.name, data / s.name)
    windows = json.loads((root / "nab" / "combined_windows.json").read_text())
    (args.work / "windows.json").write_text(json.dumps({s.name: windows[s.name] for s in test}))

    names = {}
    for fam, r in ds["results"].items():
        m = find_method(r["setting"], fam)
        names[fam] = alias(fam)
        for s in test:
            sc = P.raw_scores(m, s, {})
            ratio, _ = P.ratios(sc, s.channels, r["q"], m.fixed_tau)
            flagged = np.flatnonzero(ratio >= 1.0) + P.FIRST_EVAL
            score = np.zeros(len(s.meta["values"]))
            score[flagged * P.BLOCK + P.BLOCK - 1] = 1.0
            folder, fname = s.name.split("/")
            out = res / names[fam] / folder / f"{names[fam]}_{fname}"
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as fh:
                fh.write("timestamp,value,anomaly_score\n")
                for t, v, a in zip(s.meta["timestamps"], s.meta["values"], score, strict=True):
                    fh.write(f"{t},{v!r},{a:.1f}\n")
    for det in ("null", *PUBLISHED):
        for s in test:
            folder, fname = s.name.split("/")
            dst = res / det / folder / f"{det}_{fname}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(args.nab / "results" / det / folder / f"{det}_{fname}", dst)
    thresholds = json.loads((args.nab / "config" / "thresholds.json").read_text())
    for a in names.values():
        thresholds[a] = {pr: {"threshold": 0.5, "score": None} for pr in PROFILES}
    (args.work / "thresholds.json").write_text(json.dumps(thresholds))
    dets = [*names.values(), "null", *PUBLISHED]
    cmd = [
        args.nab_python,
        "run.py",
        "--score",
        "--normalize",
        "--skipConfirmation",
        "-d",
        *dets,
        "--dataDir",
        str(data),
        "--resultsDir",
        str(res),
        "--windowsFile",
        str(args.work / "windows.json"),
        "--thresholdsFile",
        str(args.work / "thresholds.json"),
        "-n",
        "1",
    ]
    proc = subprocess.run(cmd, cwd=args.nab, capture_output=True, text=True, check=False)
    (args.work / "nab_stdout.txt").write_text(proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0:
        print(proc.stdout[-2000:], proc.stderr[-2000:])
        raise SystemExit(f"NAB scorer failed ({proc.returncode})")
    final = json.loads((res / "final_results.json").read_text())
    inv = {v: k for k, v in names.items()}
    report["nab_official_score"] = {
        "evidence": "REPORTED (NAB official scorer, test-split sub-corpus; see note)",
        "nab_commit": nab_commit,
        "series": len(test),
        "note": __doc__.split("\n\n", 1)[1].strip(),
        "normalized": {
            inv.get(d, d): {pr: round(float(v), 2) for pr, v in final[d].items()}
            for d in dets
            if d != "null"
        },
    }
    args.results.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    for d, v in report["nab_official_score"]["normalized"].items():
        print(d, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
