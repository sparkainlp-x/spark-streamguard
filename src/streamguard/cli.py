"""Command-line interface: ``python -m streamguard`` / ``streamguard``.

Reads numbers from a CSV file or stdin, chunks them into 32-value blocks in reading order and
writes one JSON object per line to stdout: a ``config`` record first, then one ``decision``
record per block. Streaming end to end: at most one block is held in memory.

Parsing rules (fail-closed):

* Cells are split on commas (CSV) and on whitespace inside a cell; empty cells are skipped.
* ``--column`` selects one column (0-based index, or a header name with ``--header``).
* A token that is not a number (``abc``) is kept as an invalid value, so its block latches with
  ``BAD_TYPE``. ``nan`` / ``inf`` parse as floats and latch with ``NAN_INPUT`` / ``INF_INPUT``.
* A trailing partial block (fewer than 32 values) is emitted and latches with ``BAD_SHAPE``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable, Iterator
from typing import IO

from streamguard import __version__
from streamguard.calibrate import DEFAULT_QUANTILE, DriftConfig
from streamguard.reference import DEFAULT_REFERENCE, MODES
from streamguard.score import BLOCK
from streamguard.stream import AuditLogWriter, StreamGuard


def _parse_token(tok: str) -> float | None:
    try:
        return float(tok)
    except ValueError:
        return None


def iter_values(
    fh: IO[str], column: str | None = None, header: bool = False
) -> Iterator[float | None]:
    """Yield parsed values (``None`` for an unparseable token) from a CSV/text stream."""
    reader = csv.reader(fh)
    col_idx: int | None = None
    if header:
        names = next(reader, None) or []
        if column is not None:
            names = [n.strip() for n in names]
            if column in names:
                col_idx = names.index(column)
            elif column.lstrip("-").isdigit():
                col_idx = int(column)
            else:
                raise SystemExit(f"streamguard: column {column!r} not found in header {names}")
    elif column is not None:
        if not column.lstrip("-").isdigit():
            raise SystemExit("streamguard: --column by name requires --header")
        col_idx = int(column)

    for row in reader:
        cells: Iterable[str]
        if col_idx is not None:
            try:
                cells = [row[col_idx]]
            except IndexError:
                if not row:  # blank line
                    continue
                cells = ["<missing>"]  # short row: fail closed on that value
        else:
            cells = row
        for cell in cells:
            for tok in cell.split():
                yield _parse_token(tok)


def iter_blocks(values: Iterable[float | None], size: int = BLOCK) -> Iterator[list]:
    """Chunk values into lists of ``size``; the last list may be shorter."""
    buf: list = []
    for v in values:
        buf.append(v)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="streamguard",
        description="Fail-closed streaming guard: read numbers, score 32-value blocks, "
        "write JSONL admit/latch decisions.",
    )
    p.add_argument("input", nargs="?", default="-", help="CSV/text file, or '-' for stdin")
    p.add_argument("--column", help="column to read: 0-based index, or name with --header")
    p.add_argument("--header", action="store_true", help="first CSV row is a header")
    p.add_argument(
        "--tau",
        type=float,
        default=None,
        help="fixed threshold (e.g. 0.5); default: self-calibrate from warm-up",
    )
    p.add_argument("--warmup", type=int, default=32, help="warm-up blocks (default 32)")
    p.add_argument(
        "--quantile",
        type=float,
        default=DEFAULT_QUANTILE,
        help="warm-up score quantile used as tau (default 0.99)",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=None,
        help="fixed residual scale (default: robust warm-up scale, or 1.0 with --tau)",
    )
    p.add_argument(
        "--reference",
        choices=MODES,
        default=DEFAULT_REFERENCE,
        help=f"reference mode (default {DEFAULT_REFERENCE}); see README",
    )
    p.add_argument(
        "--ewma-alpha",
        type=float,
        default=0.05,
        help="smoothing factor for --reference ewma (default 0.05)",
    )
    p.add_argument(
        "--ref-blocks",
        type=int,
        default=4,
        help="trailing-median reference window in blocks (default 4)",
    )
    p.add_argument(
        "--drift-rate",
        type=float,
        default=0.0,
        help="bounded tau drift rate, 0 disables (default 0)",
    )
    p.add_argument(
        "--drift-max-rel",
        type=float,
        default=0.25,
        help="max relative tau drift from the calibrated value (default 0.25)",
    )
    p.add_argument("--audit-log", help="also append the JSONL records to this file")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def run(
    argv: list[str] | None = None,
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    try:
        guard = StreamGuard(
            tau=args.tau,
            warmup_blocks=args.warmup,
            quantile=args.quantile,
            scale=args.scale,
            reference=args.reference,
            ref_blocks=args.ref_blocks,
            ewma_alpha=args.ewma_alpha,
            drift=DriftConfig(rate=args.drift_rate, max_rel=args.drift_max_rel),
        )
    except (TypeError, ValueError) as exc:
        print(f"streamguard: invalid configuration: {exc}", file=stderr)
        return 2

    writers = [AuditLogWriter(stdout, flush=False)]
    if args.audit_log:
        writers.append(AuditLogWriter(args.audit_log))
    fh = stdin if args.input == "-" else open(args.input, newline="", encoding="utf-8")  # noqa: SIM115
    n = admitted = 0
    try:
        cfg = guard.config()
        for w in writers:
            w.write_record(cfg)
        for block in iter_blocks(iter_values(fh, args.column, args.header)):
            d = guard.process(block)
            n += 1
            admitted += d.admit
            for w in writers:
                w.write(d)
    finally:
        if fh is not stdin:
            fh.close()
        for w in writers:
            w.close()
    print(
        f"streamguard: {n} blocks, {admitted} admitted, {n - admitted} latched, tau={guard.tau}",
        file=stderr,
    )
    return 0


def main() -> None:
    sys.exit(run())
