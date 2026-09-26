"""The bridge banks resume PARTS and guards the RAM floor — the two things a 10 h pass needs.

    PYTHONPATH=D:/Projects/TanitAD/stack;D:/Projects/TanitAD/taniteval \
      C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q tests/test_bridge_resume.py

⛔ WHY THIS EXISTS. The arm loop used to bank nothing until its last line. This programme has
MEASURED that failure twice — navhard's official v2 runner scored all 5,912 scenarios and died in
aggregation, and W3's own final cache pass was killed at 3,164 s leaving a 215-row CSV. On CPU the
navtest pass is ~10 h, so "it ran and then we lost it" is the expected failure, not the exotic one.

Every expectation below is written as a LITERAL, never as an expression over the code under test,
and the two guards carry a MUTATION control: the sustained-RAM rule and the unreadable-part rule
each have a case that must read the OPPOSITE value when the mechanism is removed.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]
CODE = PKG / "code"


def _mod():
    spec = importlib.util.spec_from_file_location("w3_bridge", CODE / "run_bridge_navtest.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["w3_bridge"] = m
    spec.loader.exec_module(m)
    return m


M = _mod()


def _cols(tokens, base):
    n = len(tokens)
    return {"token": np.asarray(tokens),
            "fingerprint": np.asarray([f"fp{t}" for t in tokens]),
            "source": np.asarray(["refcv4b"] * n),
            "poses": np.stack([np.full((8, 3), base + i, dtype=np.float32) for i in range(n)]),
            "knots": np.stack([np.full((8, 2), base + i, dtype=np.float32) for i in range(n)]),
            "device": np.asarray(["cpu"] * n)}


# ── the parts round-trip, in the caller's order and not the file's ───────────────────────────

def test_assemble_returns_the_requested_order_not_the_part_order(tmp_path):
    M.chunk_write(tmp_path, 0, _cols(["t3", "t4"], 30.0))
    M.chunk_write(tmp_path, 1, _cols(["t1", "t2"], 10.0))
    got = M.chunk_assemble(tmp_path, ["t1", "t2", "t3", "t4"])
    assert got["token"].tolist() == ["t1", "t2", "t3", "t4"]
    # LITERALS: t1 came from part 1 (base 10), t3 from part 0 (base 30).
    assert got["poses"][0][0][0] == 10.0
    assert got["poses"][1][0][0] == 11.0
    assert got["poses"][2][0][0] == 30.0
    assert got["poses"][3][0][0] == 31.0
    assert got["fingerprint"].tolist() == ["fpt1", "fpt2", "fpt3", "fpt4"]


def test_one_part_and_two_parts_assemble_bit_identically(tmp_path):
    """Resuming must not change a single number: chunking is bookkeeping, not arithmetic."""
    toks = [f"t{i}" for i in range(6)]
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir()
    two.mkdir()
    M.chunk_write(one, 0, _cols(toks, 100.0))
    M.chunk_write(two, 0, _cols(toks[:2], 100.0))
    M.chunk_write(two, 1, _cols(toks[2:4], 102.0))
    M.chunk_write(two, 2, _cols(toks[4:], 104.0))
    a, b = M.chunk_assemble(one, toks), M.chunk_assemble(two, toks)
    for k in ("token", "fingerprint", "source", "device"):
        assert a[k].tolist() == b[k].tolist()
    assert a["poses"].tobytes() == b["poses"].tobytes()
    assert a["knots"].tobytes() == b["knots"].tobytes()


def test_scan_reports_exactly_the_banked_tokens_so_a_resume_skips_them(tmp_path):
    M.chunk_write(tmp_path, 0, _cols(["a", "b"], 1.0))
    M.chunk_write(tmp_path, 1, _cols(["c"], 3.0))
    banked = M.chunk_scan(tmp_path)
    assert sorted(banked) == ["a", "b", "c"]
    todo = [t for t in ["a", "b", "c", "d"] if t not in banked]
    assert todo == ["d"]                       # LITERAL: only the unbanked token is recomputed


def test_assemble_refuses_a_token_it_does_not_hold(tmp_path):
    M.chunk_write(tmp_path, 0, _cols(["a"], 1.0))
    with pytest.raises(SystemExit):
        M.chunk_assemble(tmp_path, ["a", "missing"])


def test_write_is_atomic_and_leaves_no_temp(tmp_path):
    M.chunk_write(tmp_path, 7, _cols(["a"], 1.0))
    assert [p.name for p in sorted(tmp_path.glob("*.npz"))] == ["part_00007.npz"]
    assert list(tmp_path.glob("*.tmp.npz")) == []


# ── an unreadable part is recomputed, never counted done ─────────────────────────────────────

def test_an_unreadable_part_is_ignored_while_a_good_one_in_the_same_scan_reads(tmp_path):
    """⚠️ SAME-BREATH CONTROL. A scan that returns few tokens because it could not READ a file is
    indistinguishable from an honest small scan — unless a control in the SAME call must read."""
    M.chunk_write(tmp_path, 0, _cols(["good"], 1.0))          # the control: must be found
    (tmp_path / "part_00001.npz").write_bytes(b"not an npz at all")
    banked = M.chunk_scan(tmp_path)
    assert sorted(banked) == ["good"]          # LITERAL — the corrupt part contributes nothing
    assert "good" in banked                    # …and the control proves the scan itself worked


# ── the RAM floor: one low sample is not pressure ────────────────────────────────────────────

def _probe(seq):
    it = iter(seq)
    return lambda: next(it)


def test_a_single_low_sample_that_recovers_is_not_pressure(tmp_path):
    ok, last, unread = M.ram_ok(3000, sustain=3, gap_s=0, probe=_probe([1900, 3413, 3400]))
    assert ok is True and last == 3413 and unread == 0


def test_a_sustained_low_reading_is_pressure(tmp_path):
    ok, last, unread = M.ram_ok(3000, sustain=3, gap_s=0, probe=_probe([1900, 1800, 1700]))
    assert ok is False and last == 1700 and unread == 0


def test_mutation_without_the_sustain_the_recovering_sample_reads_as_pressure(tmp_path):
    """⛔ THE MUTATION CONTROL. sustain=1 is the defect this guard exists to prevent — a single
    1,900 MB reading (MEASURED 2026-09-20; it recovered to 3,413 MB) stopping a healthy chain.
    It must read the OPPOSITE of the test above, or that test proves nothing about the sustain."""
    ok, _, _ = M.ram_ok(3000, sustain=1, gap_s=0, probe=_probe([1900, 3413, 3400]))
    assert ok is False


def test_an_unreadable_probe_fails_open_and_is_counted(tmp_path):
    """A number that could not be read is not a low number — but it is not silence either."""
    ok, _, unread = M.ram_ok(3000, sustain=3, gap_s=0, probe=_probe([None]))
    assert ok is True and unread == 1


def test_the_floor_is_the_briefs_3_gb(tmp_path):
    assert M.RAM_FLOOR_MB == 3000              # LITERAL: the W3 brief's binding floor
