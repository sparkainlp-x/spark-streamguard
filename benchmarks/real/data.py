"""Loaders for the cached NAB and SMD files (see download.py) into protocol.Series."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from protocol import Series, make_series

HERE = Path(__file__).resolve().parent


def load_nab(root: Path, keys: list[str] | None = None) -> list[Series]:
    """Load NAB series (default: all 47 keys in nab_manifest.json)."""
    if keys is None:
        keys = json.loads((HERE / "nab_manifest.json").read_text(encoding="utf-8"))["series"]
    windows = json.loads((root / "nab" / "combined_windows.json").read_text(encoding="utf-8"))
    out = []
    for key in sorted(keys):
        ts, vals = [], []
        with (root / "nab" / key).open(encoding="utf-8") as fh:
            header = fh.readline().strip().split(",")
            if header != ["timestamp", "value"]:
                raise ValueError(f"{key}: unexpected header {header}")
            for line in fh:
                if line.strip():
                    t, v = line.rsplit(",", 1)
                    ts.append(t.strip())
                    vals.append(float(v))
        t = np.array(ts, dtype="datetime64[s]")
        lab = np.zeros(len(t), dtype=int)
        for a, b in windows[key]:
            lo = np.datetime64(a.replace(" ", "T")[:19])
            hi = np.datetime64(b.replace(" ", "T")[:19])
            lab[(t >= lo) & (t <= hi)] = 1
        ser = make_series(key, "NAB", np.array(vals), lab)
        ser.meta = {"timestamps": ts, "values": vals}
        out.append(ser)
    return out


def load_smd(root: Path) -> list[Series]:
    out = []
    for path in sorted((root / "smd" / "test").glob("machine-*.txt")):
        x = np.loadtxt(path, delimiter=",", dtype=np.float64)
        lab = np.loadtxt(root / "smd" / "test_label" / path.name, dtype=int)
        out.append(make_series(path.stem, "SMD", x, lab))
    return out


LOADERS = {"nab": load_nab, "smd": load_smd}
