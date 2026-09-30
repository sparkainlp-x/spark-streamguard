"""CLI end-to-end and JSONL format."""

import io
import json
import subprocess
import sys

import numpy as np
import pytest

from streamguard import __version__
from streamguard.cli import iter_blocks, iter_values, run

KEYS = {"type", "v", "i", "admit", "latch", "score", "tau", "reason", "peak", "event", "tau_next"}


def call(argv, stdin_text=""):
    out, err = io.StringIO(), io.StringIO()
    code = run(argv, stdin=io.StringIO(stdin_text), stdout=out, stderr=err)
    recs = [json.loads(line) for line in out.getvalue().splitlines()]
    return code, recs, err.getvalue()


def series(n_blocks, spikes=()):
    x = np.random.default_rng(42).normal(size=n_blocks * 32)
    for i in spikes:
        x[i] += 40.0
    return x


def test_stdin_fixed_tau():
    text = " ".join(["0.01"] * 64)
    code, recs, err = call(["--tau", "0.5", "--ref-blocks", "1"], text)
    assert code == 0
    assert recs[0]["type"] == "config" and recs[0]["mode"] == "fixed"
    assert [r["reason"] for r in recs[1:]] == ["NO_REFERENCE", "OK"]
    assert "2 blocks, 1 admitted, 1 latched" in err


def test_csv_file_calibrated_detects_spike(tmp_path):
    x = series(40, spikes=[37 * 32 + 5])
    p = tmp_path / "in.csv"
    p.write_text("value\n" + "\n".join(f"{v:.6f}" for v in x) + "\n")
    code, recs, _ = call([str(p), "--header", "--column", "value", "--warmup", "16"])
    assert code == 0
    decisions = recs[1:]
    assert len(decisions) == 40
    for r in decisions:
        assert set(r) == KEYS and r["type"] == "decision" and r["latch"] == int(not r["admit"])
    assert decisions[16]["event"] == "CALIBRATED"
    assert decisions[0]["reason"] == "NO_REFERENCE"
    assert all(r["reason"] == "NOT_CALIBRATED" for r in decisions[1:17])
    assert decisions[37]["reason"] == "SCORE_OVER_TAU"
    assert decisions[37]["peak"] > 10


def test_multi_column_rows_are_flattened_row_major():
    rows = "\n".join(",".join(["0.0"] * 8) for _ in range(8))  # 64 values
    code, recs, _ = call(["--tau", "0.5", "--ref-blocks", "1"], rows)
    assert code == 0 and len(recs) == 3


def test_column_by_index_without_header():
    rows = "\n".join(f"{i},0.0" for i in range(64))
    code, recs, _ = call(["--tau", "0.5", "--column", "1", "--ref-blocks", "1"], rows)
    assert [r["reason"] for r in recs[1:]] == ["NO_REFERENCE", "OK"]


def test_column_by_index_with_header():
    rows = "a,b\n" + "\n".join("5,0.0" for _ in range(32))
    _, recs, _ = call(["--header", "--column", "1", "--tau", "0.5"], rows)
    assert len(recs) == 2


def test_fail_closed_tokens_and_partial_block():
    vals = ["0.0"] * 32 + ["0.0"] * 31 + ["abc"] + ["nan"] + ["0.0"] * 31 + ["inf"] + ["0.0"] * 31
    vals += ["0.0"] * 10  # partial trailing block
    code, recs, _ = call(["--tau", "0.5", "--ref-blocks", "1"], "\n".join(vals))
    assert code == 0
    reasons = [r["reason"] for r in recs[1:]]
    assert reasons == ["NO_REFERENCE", "BAD_TYPE", "NAN_INPUT", "INF_INPUT", "BAD_SHAPE"]
    assert all(not r["admit"] for r in recs[1:])


def test_short_rows_with_column_fail_closed():
    rows = "\n".join(["1,0.0"] * 31 + ["1"] + [""])
    _, recs, _ = call(["--tau", "0.5", "--column", "1"], rows)
    assert recs[1]["reason"] == "BAD_TYPE"


def test_empty_input():
    code, recs, err = call([], "")
    assert code == 0 and len(recs) == 1 and "0 blocks" in err


def test_audit_log_file(tmp_path):
    log = tmp_path / "audit.jsonl"
    text = "\n".join(["0.0"] * 64)
    code, recs, _ = call(["--tau", "0.5", "--audit-log", str(log)], text)
    assert code == 0
    file_recs = [json.loads(x) for x in log.read_text().splitlines()]
    assert file_recs == recs


def test_drift_flags():
    x = series(60)
    code, recs, _ = call(
        ["--warmup", "8", "--drift-rate", "0.05", "--drift-max-rel", "0.1"],
        "\n".join(map(str, x)),
    )
    assert code == 0
    assert recs[0]["drift_rate"] == 0.05


@pytest.mark.parametrize(
    "argv",
    [["--warmup", "1"], ["--tau", "-1"], ["--quantile", "2"], ["--drift-rate", "5"]],
)
def test_invalid_configuration_exit_2(argv):
    code, recs, err = call(argv, "")
    assert code == 2 and recs == [] and "invalid configuration" in err


def test_column_name_errors():
    with pytest.raises(SystemExit, match="requires --header"):
        call(["--column", "value"], "1\n")
    with pytest.raises(SystemExit, match="not found"):
        call(["--header", "--column", "zzz"], "a,b\n1,2\n")


def test_iter_helpers():
    assert list(iter_values(io.StringIO("1, 2 3\n\nx\n"))) == [1.0, 2.0, 3.0, None]
    assert [len(b) for b in iter_blocks(range(70))] == [32, 32, 6]
    assert list(iter_blocks([])) == []


def test_python_dash_m_end_to_end(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text(" ".join(["0.0"] * 96))
    proc = subprocess.run(
        [sys.executable, "-m", "streamguard", str(p), "--tau", "0.5"],
        capture_output=True,
        text=True,
        check=True,
    )
    recs = [json.loads(x) for x in proc.stdout.splitlines()]
    assert [r["reason"] for r in recs[1:]] == ["NO_REFERENCE", "OK", "OK"]
    assert "3 blocks" in proc.stderr


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as e:
        run(["--version"])
    assert e.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_blank_lines_skipped_with_column_and_header_without_column():
    rows = "x,y\n" + "\n\n".join(["1,0.0"] * 32)
    _, recs, _ = call(["--header", "--column", "y", "--tau", "0.5"], rows)
    assert len(recs) == 2 and recs[1]["reason"] == "NO_REFERENCE"
    _, recs, _ = call(["--header", "--tau", "0.5"], rows)
    assert len(recs) == 3  # 64 values from both columns


def test_module_entry_point_in_process(monkeypatch, capsys):
    import runpy

    monkeypatch.setattr(sys, "argv", ["streamguard", "--tau", "0.5"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(" ".join(["0"] * 32)))
    with pytest.raises(SystemExit) as e:
        runpy.run_module("streamguard", run_name="__main__")
    assert e.value.code == 0
    out = capsys.readouterr().out.splitlines()
    assert json.loads(out[1])["reason"] == "NO_REFERENCE"


@pytest.mark.parametrize("mode", ["median", "ewma", "detrend", "diff"])
def test_reference_flag(mode):
    x = series(30, spikes=[25 * 32 + 9])
    code, recs, _ = call(
        ["--reference", mode, "--ewma-alpha", "0.2", "--warmup", "16"], "\n".join(map(str, x))
    )
    assert code == 0 and recs[0]["scorer_params"]["reference"] == mode
    assert recs[1 + 25]["reason"] == "SCORE_OVER_TAU"


def test_reference_flag_rejects_unknown():
    with pytest.raises(SystemExit):
        call(["--reference", "mean"], "")
