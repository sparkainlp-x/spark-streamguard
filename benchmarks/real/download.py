"""Download the pinned real-data benchmark files and verify their SHA-256 checksums.

Data are cached **outside the repository**: ``$STREAMGUARD_BENCH_DATA`` if set, otherwise
``~/.cache/spark-streamguard-bench``. Nothing here runs in CI.

* NAB (Numenta Anomaly Benchmark, MIT): the 47 series in the labelled real-data folders and
  ``labels/combined_windows.json``, at the commit pinned in ``nab_manifest.json``.
* SMD (Server Machine Dataset, MIT, from the OmniAnomaly repository): the 28 ``test`` and
  ``test_label`` files, at the commit pinned in ``smd_manifest.json`` (about 245 MB).
* Yahoo S5 is **not included**: it is distributed by Yahoo Webscope under a licence that needs
  registration, and unofficial mirrors are not used.

Usage::

    python benchmarks/real/download.py            # download + verify both datasets
    python benchmarks/real/download.py --verify   # verify the cache only
    python benchmarks/real/download.py --pin-smd  # (maintainer) rebuild smd_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAB_MANIFEST = HERE / "nab_manifest.json"
SMD_MANIFEST = HERE / "smd_manifest.json"
SMD_REPO = "NetManAIOps/OmniAnomaly"
SMD_COMMIT = "7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4"
SMD_MACHINES = (
    [f"machine-1-{i}" for i in range(1, 9)]
    + [f"machine-2-{i}" for i in range(1, 10)]
    + [f"machine-3-{i}" for i in range(1, 12)]
)


def cache_dir() -> Path:
    env = os.environ.get("STREAMGUARD_BENCH_DATA")
    return Path(env) if env else Path.home() / ".cache" / "spark-streamguard-bench"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def nab_files() -> dict[str, tuple[str, str]]:
    """Local relative path -> (pinned URL, SHA-256)."""
    m = json.loads(NAB_MANIFEST.read_text(encoding="utf-8"))
    raw = f"https://raw.githubusercontent.com/numenta/NAB/{m['nab_commit']}"
    files = {f"nab/{k}": (f"{raw}/data/{k}", sha) for k, sha in sorted(m["series"].items())}
    files["nab/combined_windows.json"] = (f"{raw}/{m['labels']['path']}", m["labels"]["sha256"])
    return files


def smd_urls() -> dict[str, str]:
    raw = f"https://raw.githubusercontent.com/{SMD_REPO}/{SMD_COMMIT}/ServerMachineDataset"
    out = {}
    for mach in SMD_MACHINES:
        for part in ("test", "test_label"):
            out[f"smd/{part}/{mach}.txt"] = f"{raw}/{part}/{mach}.txt"
    return out


def smd_files() -> dict[str, tuple[str, str]]:
    m = json.loads(SMD_MANIFEST.read_text(encoding="utf-8"))
    if m["commit"] != SMD_COMMIT:
        raise ValueError("smd_manifest.json commit does not match SMD_COMMIT")
    urls = smd_urls()
    return {k: (urls[k], sha) for k, sha in sorted(m["files"].items())}


def fetch(url: str, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as fh:  # noqa: S310
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    got = sha256_file(tmp)
    tmp.replace(path)
    return got


def download(files: dict[str, tuple[str, str]], root: Path) -> None:
    for name, (url, expected) in files.items():
        path = root / name
        if path.is_file() and sha256_file(path) == expected:
            continue
        got = fetch(url, path)
        if got != expected:
            path.unlink()
            raise ValueError(f"SHA-256 mismatch for {url}: got {got}, expected {expected}")
        print(f"ok {name}")


def verify(files: dict[str, tuple[str, str]], root: Path) -> None:
    for name, (_, expected) in files.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing; run python benchmarks/real/download.py")
        if sha256_file(path) != expected:
            raise ValueError(f"SHA-256 mismatch for {path}")


def pin_smd(root: Path) -> None:
    files = {}
    for name, url in smd_urls().items():
        files[name] = fetch(url, root / name)
        print(f"pinned {name}")
    SMD_MANIFEST.write_text(
        json.dumps(
            {
                "repo": f"https://github.com/{SMD_REPO}",
                "commit": SMD_COMMIT,
                "path": "ServerMachineDataset",
                "license": "MIT (ServerMachineDataset/LICENSE at commit)",
                "files": files,
            },
            indent=1,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download/verify NAB and SMD (pinned, SHA-256).")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--pin-smd", action="store_true")
    p.add_argument("--only", choices=("nab", "smd"))
    args = p.parse_args(argv)
    root = cache_dir()
    if args.pin_smd:
        pin_smd(root)
        return 0
    sets = {"nab": nab_files, "smd": smd_files}
    for key, fn in sets.items():
        if args.only and key != args.only:
            continue
        files = fn()
        (verify if args.verify else download)(files, root)
        verify(files, root)
        print(f"verified {len(files)} {key} files in {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
