"""Box 2's RA-1 temperature guard (code/box2_ops/ra1runs_guard.sh) must stay armed until ITS chain ends. CPU only, < 1 s.

MEASURED 2026-10-20 22:43: chain_ra1_full5.sh's first log line is "waiting for ZZRA1A1-DONE in .../chain_ra1_full4.out", and the guard's
end test was an UNANCHORED `grep -E 'ZZRA1A1-DONE|...'` over that log, so the guard read the chain's own waiting line as "the chain ended"
and switched itself off one minute into the gamma-200 training (re-armed by hand within about a minute). A polling filter that contains
the pattern it searches for matches itself (TRAPS.md). Pinned here:
  * the guard's end-of-chain regex matches a chain's DONE / STOP marker only at the START of a line;
  * no chain the guard watches writes a non-marker line that the regex matches (its "waiting for <predecessor marker>" line included);
  * every chain that starts the guard has its own markers in the regex (else the guard never exits).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
OPS = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-refav1-hj-mini" / "code" / "box2_ops"
GUARD = OPS / "ra1runs_guard.sh"
pytestmark = pytest.mark.skipif(not GUARD.exists(), reason="the refav1-hj-mini package is not in this checkout")


def _end_regex() -> re.Pattern:
    lines = [ln for ln in GUARD.read_text(encoding="utf-8").splitlines() if "$CHAIN_OUT" in ln and "grep -q -E" in ln]
    assert len(lines) == 1, lines
    return re.compile(re.search(r"grep -q -E '([^']+)'", lines[0]).group(1))


def _chains():
    return sorted(p for p in OPS.glob("chain_ra1_*.sh") if "ra1runs_guard.sh" in p.read_text(encoding="utf-8"))


def test_the_end_regex_matches_a_marker_only_at_the_start_of_a_line():
    rx = _end_regex()
    for m in ("ZZRA1A1-DONE 10-20 19:07 tip 890e12a6: ...", "ZZRA1A2-STOP box 2 unreachable", "ZZRA1A2-DONE 10-21 01:00 tip x: ...",
              "ZZRA1FULLRUNS-DONE 10-19 ..."):
        assert rx.search(m), m
    for m in ("10-20 22:40 waiting for ZZRA1A1-DONE in /c/Users/Admin/box2_ship/logs/chain_ra1_full4.out",
              "10-20 22:40 waiting for ZZRA1A2-DONE in x", "echo ZZRA1A2-DONE", "  ZZRA1A1-STOP indented"):
        assert not rx.search(m), m


def test_no_watched_chain_writes_a_non_marker_line_the_guard_reads_as_its_end():
    rx = _end_regex()
    chains = _chains()
    assert chains, "no chain starts the guard"
    for p in chains:
        src = p.read_text(encoding="utf-8")
        for w in re.findall(r'echo "\$\(date [^)]*\)\s+([^"]*)"', src):     # every timestamped progress line the chain echoes
            line = "10-20 22:40 " + w.replace("$AFTER_DONE", re.search(r"AFTER_DONE:-([A-Z0-9-]+)", src).group(1)
                                               if "AFTER_DONE:-" in src else "X")
            assert not rx.search(line), (p.name, line)


def test_every_chain_that_starts_the_guard_has_its_markers_in_the_regex():
    rx = _end_regex()
    for p in _chains():
        src = p.read_text(encoding="utf-8")
        stops = set(re.findall(r'echo "(ZZ[A-Z0-9]+)-STOP', src))
        assert stops, p.name
        for s in stops:
            assert rx.search(f"{s}-DONE x") and rx.search(f"{s}-STOP x"), (p.name, s)
